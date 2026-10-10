"""
Research Assistant Crew — LangGraph pipeline.

Implements all hardening items:
  1. Source integrity: structured TavilySource data, index-based references, URL validation.
  2. Analyst sees real content: include_raw_content with per-source token budget.
  3. Structured Manager plan (Pydantic), used directly for search queries.
  4. Revision loops: search retry (≤2) if too few cited sources; writer retry (≤1) on reject.
  5. Per-node token limits; detect finish_reason == "length".
  6. No time.sleep() pacing; removed redundant query-extraction LLM call.
"""

import json
import logging
import os
import re
import time
from collections.abc import Generator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError
from tavily import TavilyClient

from agents.prompts import (
    ANALYSIS_AGENT_SYSTEM_PROMPT,
    MANAGER_PLAN_SYSTEM_PROMPT,
    MANAGER_REVIEW_SYSTEM_PROMPT,
    SEARCH_AGENT_SYSTEM_PROMPT,
    WRITER_AGENT_SYSTEM_PROMPT,
)
from agents.state import AgentLog, ManagerPlan, ResearchState, TavilySource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-configurable caps (item 4)
# ---------------------------------------------------------------------------
MAX_SEARCH_RETRIES: int = int(os.getenv("MAX_SEARCH_RETRIES", "2"))
MAX_WRITER_RETRIES: int = int(os.getenv("MAX_WRITER_RETRIES", "1"))
MAX_TOTAL_LLM_CALLS: int = int(os.getenv("MAX_TOTAL_LLM_CALLS", "20"))

# Per-source content token budget (item 2): ~1,500 tokens ≈ 6,000 chars
SOURCE_CONTENT_CHAR_BUDGET: int = int(os.getenv("SOURCE_CONTENT_CHAR_BUDGET", "6000"))

# Per-node max_tokens (item 5)
TOKENS_PLAN = 512
TOKENS_SEARCH = 1024
TOKENS_ANALYSIS = 2048
TOKENS_WRITER = 6000
TOKENS_REVIEW = 6000


# ---------------------------------------------------------------------------
# Pydantic model for Manager plan (item 3)
# ---------------------------------------------------------------------------
class ManagerPlanModel(BaseModel):
    core_objective: str
    sub_tasks: list[str]
    search_queries: list[str]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def get_llm(
    groq_api_key: str,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> ChatGroq:
    """Initialize a Groq Chat LLM with the given per-node token budget."""
    return ChatGroq(  # type: ignore[call-arg]
        groq_api_key=groq_api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def invoke_with_retry(
    llm: ChatGroq,
    messages: list[Any],
    max_retries: int = 5,
    initial_wait: float = 3.0,
) -> Any:
    """Invoke LLM with exponential backoff on 429 rate-limit errors (item 6)."""
    wait_time = initial_wait
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str
                or "rate_limit" in err_str.lower()
                or "tpm" in err_str.lower()
            )
            if is_rate_limit and attempt < max_retries:
                match = re.search(r"try again in ([\d\.]+)s", err_str, re.IGNORECASE)
                parsed_wait = float(match.group(1)) + 1.0 if match else 0.0
                sleep_duration = max(wait_time, parsed_wait)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Waiting %.1fs.",
                    attempt,
                    max_retries,
                    sleep_duration,
                )
                time.sleep(sleep_duration)
                wait_time *= 1.8
            else:
                raise


def _usage_tokens(response: Any) -> int | None:
    """Extract total_tokens safely from LLM response usage metadata."""
    usage = getattr(response, "usage_metadata", None) or {}
    return usage.get("total_tokens")


def _finish_reason(response: Any) -> str:
    """Extract finish_reason safely from LLM response metadata."""
    meta = getattr(response, "response_metadata", None) or {}
    return str(meta.get("finish_reason") or "")


# ---------------------------------------------------------------------------
# Query refinement and source merging for search retries
# ---------------------------------------------------------------------------

def refined_queries(topic: str, attempt: int) -> list[str]:
    """Return refined search queries for retry attempts."""
    if attempt <= 1:
        return [
            f"{topic} primary sources",
            f"{topic} benchmarks and empirical results",
        ]
    return [
        f"{topic} official documentation",
        f"{topic} academic papers",
    ]


def merge_sources(
    existing: list[TavilySource],
    new: list[TavilySource],
    limit: int = 8,
) -> list[TavilySource]:
    """Concatenate existing and new sources, deduplicate by URL preserving order, capped at limit."""
    merged: list[TavilySource] = []
    seen_urls: set[str] = set()
    for s in existing + new:
        url = s.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged.append(s)
            if len(merged) >= limit:
                break
    return merged


# ---------------------------------------------------------------------------
# Tavily search with real content (item 2)
# ---------------------------------------------------------------------------

def execute_tavily_search(
    tavily_api_key: str,
    queries: list[str],
    max_results: int = 3,
) -> list[TavilySource]:
    """Run queries via Tavily with raw content, deduplicate, apply per-source budget."""
    client = TavilyClient(api_key=tavily_api_key)
    all_results: list[TavilySource] = []
    seen_urls: set[str] = set()

    for q in queries[:3]:
        clean_q = q.strip().strip('"').strip("'").strip("`").strip("*")
        # Strip common list prefixes
        if clean_q and clean_q[0] in "-*•" and len(clean_q) > 2:
            clean_q = clean_q[2:].strip().strip('"').strip("'")
        if not clean_q or len(clean_q) < 3:
            continue
        try:
            response = client.search(
                query=clean_q,
                search_depth="advanced",
                max_results=max_results,
                include_raw_content=True,  # item 2: fetch real page content
            )
            for r in response.get("results", []):
                url = r.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                # Prefer raw_content, fall back to content snippet
                raw = r.get("raw_content") or r.get("content") or ""
                if len(raw) > SOURCE_CONTENT_CHAR_BUDGET:
                    raw = raw[:SOURCE_CONTENT_CHAR_BUDGET] + " [TRUNCATED]"
                    logger.info("Source content truncated for URL: %s", url)

                all_results.append(
                    TavilySource(
                        title=r.get("title", "Untitled Source"),
                        url=url,
                        content=raw,
                    )
                )
        except Exception as e:
            err_msg = str(e)
            if (
                "unauthorized" in err_msg.lower()
                or "invalid api key" in err_msg.lower()
                or "401" in err_msg
            ):
                raise ValueError(
                    f"Tavily API Error: Invalid API Key or Unauthorized ({err_msg})"
                )
            logger.warning("Tavily search failed for query '%s': %s", clean_q, e)

    return all_results[:8]


# ---------------------------------------------------------------------------
# URL validation (item 1)
# ---------------------------------------------------------------------------

def extract_urls_from_text(text: str) -> list[str]:
    """Extract all markdown and bare URLs from text."""
    md_urls = re.findall(r"\]\((https?://[^\s\)]+)\)", text)
    bare_urls = re.findall(
        r"(?<!\()(https?://[^\s\)\]\"\'<>]+)(?!\))", text
    )
    return list(set(md_urls + bare_urls))


def validate_report_urls(
    report_text: str, allowed_sources: list[TavilySource]
) -> list[str]:
    """Return list of warning strings for URLs in the report not in allowed_sources."""
    allowed_urls = {s["url"] for s in allowed_sources}
    report_urls = extract_urls_from_text(report_text)
    warnings = []
    for url in report_urls:
        if url not in allowed_urls:
            warnings.append(
                f"URL not in Tavily result set (possibly hallucinated): {url}"
            )
    return warnings


# ---------------------------------------------------------------------------
# Manager plan parsing (item 3)
# ---------------------------------------------------------------------------

def parse_manager_plan(raw_text: str) -> ManagerPlanModel | None:
    """Parse JSON manager plan from raw LLM output; return None on failure."""
    # Strip markdown fences if present
    text = re.sub(r"```[a-z]*\n?", "", raw_text).strip()
    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return ManagerPlanModel(**data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Manager plan parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Manager review verdict parsing (item 4)
# ---------------------------------------------------------------------------

def parse_manager_verdict(review_text: str) -> dict[str, Any]:
    """Extract verdict JSON from manager review output. Default: approve."""
    match = re.search(r"```verdict\s*(\{.*?\})\s*```", review_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"verdict": "approve", "reasons": []}


def strip_verdict_block(review_text: str) -> str:
    """Remove the trailing verdict block from the report text."""
    return re.sub(r"```verdict\s*\{.*?\}\s*```", "", review_text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Count cited source indices in analyst output (item 4)
# ---------------------------------------------------------------------------

def count_cited_source_indices(claims_text: str) -> int:
    """Count how many distinct Source #N references appear in the analyst output."""
    matches = re.findall(r"Source\s*#(\d+)", claims_text, re.IGNORECASE)
    return len(set(matches))


# ---------------------------------------------------------------------------
# Check report structure (single source of truth for structure checks)
# ---------------------------------------------------------------------------

def check_report_structure(report: str) -> list[str]:
    """Check report structure against formatting rules; return list of problem descriptions (empty if valid)."""
    problems: list[str] = []

    # Missing "## Executive Summary" (case-insensitive)
    has_summary = bool(
        re.search(r"^##\s+Executive\s+Summary", report, re.IGNORECASE | re.MULTILINE)
    )
    if not has_summary:
        problems.append("Missing '## Executive Summary' section")

    # The number of other "## " headers that are not Executive Summary or Sources must be between 3 and 5 inclusive
    all_headers = re.findall(r"^##\s+(.+)$", report, re.MULTILINE)
    content_headers = [
        h.strip()
        for h in all_headers
        if not re.match(r"^Executive\s+Summary\b", h.strip(), re.IGNORECASE)
        and not re.match(r"^Sources?\b", h.strip(), re.IGNORECASE)
    ]
    content_count = len(content_headers)
    if not (3 <= content_count <= 5):
        problems.append(
            f"Expected between 3 and 5 content sections, found {content_count}"
        )

    # Missing "## Sources" section
    has_sources = bool(
        re.search(r"^##\s+Sources?", report, re.IGNORECASE | re.MULTILINE)
    )
    if not has_sources:
        problems.append("Missing '## Sources' section")

    return problems


# ---------------------------------------------------------------------------
# Module-level routing functions (pure functions, no state mutations)
# ---------------------------------------------------------------------------

def route_after_analysis(state: ResearchState) -> str:
    """Route to search_agent if fewer than 3 cited sources and within retry limit; else writer_agent."""
    cited_count = count_cited_source_indices(state.get("claims_analysis") or "")
    retry_count = state.get("search_retry_count", 0)

    if cited_count < 3 and retry_count < MAX_SEARCH_RETRIES:
        logger.info(
            "Only %d cited sources found (need ≥3). Routing back to search (retry %d/%d).",
            cited_count,
            retry_count + 1,
            MAX_SEARCH_RETRIES,
        )
        return "search_agent"
    return "writer_agent"


def route_after_review(state: ResearchState) -> str:
    """Route to writer_agent if review was rejected and within retry limit; else END."""
    verdict = state.get("review_verdict")
    retry_count = state.get("writer_retry_count", 0)

    if verdict == "reject" and retry_count < MAX_WRITER_RETRIES:
        logger.info(
            "Manager rejected draft (retry %d/%d). Reasons: %s",
            retry_count + 1,
            MAX_WRITER_RETRIES,
            state.get("review_reasons", []),
        )
        return "writer_agent"
    return END


# ---------------------------------------------------------------------------
# Initial state builder
# ---------------------------------------------------------------------------

def make_initial_state(topic: str) -> ResearchState:
    """Construct an initial ResearchState with all fields defaulted."""
    return {
        "topic": topic,
        "plan": None,
        "parsed_plan": None,
        "sources": None,
        "raw_sources_list": [],
        "claims_analysis": None,
        "analyst_claims": [],
        "draft_report": None,
        "final_report": None,
        "url_validation_warnings": [],
        "agent_logs": [],
        "error": None,
        "search_retry_count": 0,
        "writer_retry_count": 0,
        "manager_review_truncated": False,
        "review_verdict": None,
        "review_reasons": [],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_research_graph(
    groq_api_key: str,
    tavily_api_key: str,
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0.2,
):
    """Construct the LangGraph state graph for the multiagent research crew."""

    # Per-node LLMs with tuned token budgets (item 5)
    llm_plan = get_llm(groq_api_key, model_name, temperature, max_tokens=TOKENS_PLAN)
    llm_search = get_llm(groq_api_key, model_name, temperature, max_tokens=TOKENS_SEARCH)
    llm_analysis = get_llm(groq_api_key, model_name, temperature, max_tokens=TOKENS_ANALYSIS)
    llm_writer = get_llm(groq_api_key, model_name, temperature, max_tokens=TOKENS_WRITER)
    llm_review = get_llm(groq_api_key, model_name, temperature, max_tokens=TOKENS_REVIEW)

    # Shared LLM call counter to enforce cap (item 4)
    call_counter: dict[str, int] = {"n": 0}

    def _call(llm: ChatGroq, messages: list[Any]) -> Any:
        call_counter["n"] += 1
        max_calls = MAX_TOTAL_LLM_CALLS
        if call_counter["n"] > max_calls:
            raise RuntimeError(
                f"Exceeded MAX_TOTAL_LLM_CALLS={max_calls}. Aborting."
            )
        return invoke_with_retry(llm, messages)

    # -----------------------------------------------------------------------
    # 1. MANAGER PLAN NODE
    # -----------------------------------------------------------------------
    def manager_plan_node(state: ResearchState) -> dict[str, Any]:
        t0 = time.monotonic()
        topic = state["topic"]
        messages = [
            SystemMessage(content=MANAGER_PLAN_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Research Question/Topic: {topic}\n\n"
                "Respond with the JSON plan object only."
            ),
        ]
        response = _call(llm_plan, messages)
        raw_text = response.content

        # Parse structured plan (item 3)
        parsed_plan_dict: ManagerPlan | None = None
        parsed = parse_manager_plan(raw_text)
        if parsed:
            plan_display = (
                f"**Core Objective:** {parsed.core_objective}\n\n"
                f"**Sub-Tasks:**\n" + "\n".join(f"- {t}" for t in parsed.sub_tasks) + "\n\n"
                "**Search Queries:**\n" + "\n".join(f"- `{q}`" for q in parsed.search_queries)
            )
            parsed_plan_dict = {
                "core_objective": parsed.core_objective,
                "sub_tasks": parsed.sub_tasks,
                "search_queries": parsed.search_queries,
            }
        else:
            logger.warning("Manager plan JSON parse failed; using raw text as plan.")
            plan_display = raw_text

        duration = time.monotonic() - t0
        log_entry: AgentLog = {
            "agent": "Manager Agent",
            "status": "Plan Created",
            "message": "Deconstructed topic into sub-tasks and delegated search objectives.",
            "duration_seconds": round(duration, 2),
            "token_count": _usage_tokens(response),
            "data": plan_display,
        }
        return {
            "plan": plan_display,
            "parsed_plan": parsed_plan_dict,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # -----------------------------------------------------------------------
    # 2. SEARCH AGENT NODE
    # -----------------------------------------------------------------------
    def search_node(state: ResearchState) -> dict[str, Any]:
        t0 = time.monotonic()
        topic = state["topic"]
        is_retry = state.get("claims_analysis") is not None
        if is_retry:
            retry_count = state.get("search_retry_count", 0) + 1
            queries = refined_queries(topic, retry_count)
        else:
            retry_count = state.get("search_retry_count", 0)
            parsed_plan: ManagerPlan | None = state.get("parsed_plan")
            if parsed_plan and parsed_plan.get("search_queries"):
                queries = parsed_plan["search_queries"]
            else:
                queries = [topic]

        new_results = execute_tavily_search(tavily_api_key, queries, max_results=3)

        if is_retry:
            existing = state.get("raw_sources_list") or []
            search_results = merge_sources(existing, new_results, limit=8)
        else:
            search_results = new_results

        if not search_results:
            # Fallback: retry with topic directly (no LLM call needed)
            search_results = execute_tavily_search(tavily_api_key, [topic], max_results=4)

        # Build indexed block for the LLM (item 1: LLM never sees bare URLs)
        if search_results:
            indexed_block = "\n\n".join(
                f"Source #{i + 1}:\n"
                f"- Title: {r['title']}\n"
                f"- Content:\n{r['content']}"
                for i, r in enumerate(search_results)
            )
        else:
            indexed_block = f"No web results found for topic: '{topic}'"

        curation_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Candidate Sources (retrieved via Tavily Web Search):\n{indexed_block}\n\n"
            "Curate the above sources. Refer to each source by its Source #N index only — "
            "do NOT write URLs. Present the curated source list in Markdown."
        )

        curation_response = _call(
            llm_search,
            [
                SystemMessage(content=SEARCH_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=curation_prompt),
            ],
        )
        sources_text = curation_response.content.strip()

        # Detect if the model returned a tool-call JSON instead of Markdown (item 1)
        if sources_text.startswith("{") and ("tool" in sources_text or "arguments" in sources_text):
            # Surface an explicit labeled notice — no silent fallback (item 1)
            sources_text = (
                "⚠️ **[UNCURATED — curation model returned a tool-call instead of Markdown]**\n\n"
                + indexed_block
            )
            logger.warning("Curation LLM returned tool-call JSON; surfacing uncurated notice.")

        duration = time.monotonic() - t0
        log_entry: AgentLog = {
            "agent": "Search Agent",
            "status": "Sources Curated",
            "message": (
                f"Retrieved {len(search_results)} sources via Tavily. "
                f"Retry count: {retry_count}"
            ),
            "duration_seconds": round(duration, 2),
            "token_count": _usage_tokens(curation_response),
            "data": sources_text,
        }
        return {
            "sources": sources_text,
            "raw_sources_list": search_results,
            "search_retry_count": retry_count,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # -----------------------------------------------------------------------
    # 3. ANALYSIS AGENT NODE
    # -----------------------------------------------------------------------
    def analysis_node(state: ResearchState) -> dict[str, Any]:
        t0 = time.monotonic()
        topic = state["topic"]
        sources = state.get("sources", "")
        raw_sources = state.get("raw_sources_list", [])

        # Provide the full indexed source content to the analyst (item 2)
        if raw_sources:
            evidence_block = "\n\n".join(
                f"Source #{i + 1} — {r['title']}:\n{r['content']}"
                for i, r in enumerate(raw_sources)
            )
        else:
            evidence_block = sources or "(no sources available)"

        analysis_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Curated Source List (index references):\n{sources}\n\n"
            f"Full Source Content for Fact-Checking:\n{evidence_block}\n\n"
            "Extract the 3–5 strongest empirical claims. Follow the CRITICAL RULE: "
            "every fact must appear in the source content above."
        )
        response = _call(
            llm_analysis,
            [
                SystemMessage(content=ANALYSIS_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=analysis_prompt),
            ],
        )
        claims_text = response.content

        # Extract individual claim sentences for revision-loop check (item 4)
        claim_lines = [
            line.strip()
            for line in claims_text.split("\n")
            if line.strip().startswith("**Claim**") or line.strip().startswith("- **Claim**")
        ]

        duration = time.monotonic() - t0
        log_entry: AgentLog = {
            "agent": "Analysis Agent",
            "status": "Claims Extracted",
            "message": "Extracted empirical claims, cross-referenced source evidence.",
            "duration_seconds": round(duration, 2),
            "token_count": _usage_tokens(response),
            "data": claims_text,
        }
        return {
            "claims_analysis": claims_text,
            "analyst_claims": claim_lines,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # -----------------------------------------------------------------------
    # 4. WRITER AGENT NODE
    # -----------------------------------------------------------------------
    def writer_node(state: ResearchState) -> dict[str, Any]:
        t0 = time.monotonic()
        topic = state["topic"]
        claims = state.get("claims_analysis", "")
        raw_sources = state.get("raw_sources_list", [])

        is_rewrite = state.get("draft_report") is not None
        if is_rewrite:
            writer_retry_count = state.get("writer_retry_count", 0) + 1
        else:
            writer_retry_count = state.get("writer_retry_count", 0)

        # Give writer the structured source list so it can hyperlink correctly
        source_list_block = "\n".join(
            f"{i + 1}. [{r['title']}]({r['url']})"
            for i, r in enumerate(raw_sources)
        )

        writer_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Critical Analyst's Claims & Evidence:\n{claims}\n\n"
            f"Authorised Source List (use ONLY these URLs in the Sources section):\n"
            f"{source_list_block}\n\n"
            "Write the structured technical report: 2-sentence Executive Summary, "
            "3–5 headed sections with deep technical evidence, and the complete sources list. "
            "No filler intros. Do NOT add URLs beyond the authorised list above."
        )

        if is_rewrite:
            reasons = state.get("review_reasons", [])
            if reasons:
                feedback_lines = "\n".join(f"- {r}" for r in reasons)
                writer_prompt += f"\n\nManager feedback to address in this revision:\n{feedback_lines}"
            else:
                writer_prompt += "\n\nManager feedback to address in this revision:\n"

        response = _call(
            llm_writer,
            [
                SystemMessage(content=WRITER_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=writer_prompt),
            ],
        )
        draft_text = response.content

        # Detect truncation (item 5)
        finish_reason = _finish_reason(response)
        if finish_reason == "length":
            logger.warning("Writer output was truncated (finish_reason=length).")
            draft_text += (
                "\n\n> ⚠️ **Writer output was truncated due to token limits.** "
                "The report may be incomplete."
            )

        duration = time.monotonic() - t0
        log_entry: AgentLog = {
            "agent": "Writer Agent",
            "status": "Draft Report Written",
            "message": (
                "Synthesized findings into a structured technical briefing. "
                f"Writer retry: {writer_retry_count}"
            ),
            "duration_seconds": round(duration, 2),
            "token_count": _usage_tokens(response),
            "data": draft_text,
        }
        return {
            "draft_report": draft_text,
            "writer_retry_count": writer_retry_count,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # -----------------------------------------------------------------------
    # 5. MANAGER REVIEW NODE
    # -----------------------------------------------------------------------
    def manager_review_node(state: ResearchState) -> dict[str, Any]:
        t0 = time.monotonic()
        topic = state["topic"]
        draft = state.get("draft_report", "")
        claims = state.get("claims_analysis", "")
        raw_sources = state.get("raw_sources_list", [])

        review_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Analyst's Claims (verify these appear in the report):\n{claims}\n\n"
            f"Writer's Draft Report:\n{draft}\n\n"
            "Perform final QA: check the 2-sentence summary, section depth, claim coverage, "
            "and source accuracy. Edit in-place as needed. Return the finalized report followed "
            "by the verdict JSON block."
        )
        response = _call(
            llm_review,
            [
                SystemMessage(content=MANAGER_REVIEW_SYSTEM_PROMPT),
                HumanMessage(content=review_prompt),
            ],
        )
        raw_review = response.content

        # Detect truncation (item 5)
        finish_reason = _finish_reason(response)
        truncated = finish_reason == "length"
        if truncated:
            logger.warning("Manager review output was truncated (finish_reason=length).")

        # Parse verdict (item 4)
        verdict_data = parse_manager_verdict(raw_review)
        raw_verdict = verdict_data.get("verdict", "approve")
        verdict = str(raw_verdict).lower() if isinstance(raw_verdict, str) else "approve"
        if verdict not in ("approve", "reject"):
            verdict = "approve"
        reasons = [str(r) for r in verdict_data.get("reasons", [])] if verdict == "reject" else []

        final_text = strip_verdict_block(raw_review)

        if truncated:
            final_text += (
                "\n\n> ⚠️ **Manager review output was truncated.** "
                "The report may be incomplete."
            )

        # URL validation (item 1)
        url_warnings = validate_report_urls(final_text, raw_sources)

        duration = time.monotonic() - t0
        log_entry: AgentLog = {
            "agent": "Manager Agent (Review)",
            "status": f"Review complete — verdict: {verdict}",
            "message": "Manager verified the report against claims and source list.",
            "duration_seconds": round(duration, 2),
            "token_count": _usage_tokens(response),
            "data": final_text,
        }
        return {
            "final_report": final_text,
            "url_validation_warnings": url_warnings,
            "manager_review_truncated": truncated,
            "review_verdict": verdict,
            "review_reasons": reasons,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # -----------------------------------------------------------------------
    # Assemble graph
    # -----------------------------------------------------------------------
    workflow = StateGraph(ResearchState)

    workflow.add_node("manager_plan", manager_plan_node)
    workflow.add_node("search_agent", search_node)
    workflow.add_node("analysis_agent", analysis_node)
    workflow.add_node("writer_agent", writer_node)
    workflow.add_node("manager_review", manager_review_node)

    workflow.add_edge(START, "manager_plan")
    workflow.add_edge("manager_plan", "search_agent")
    workflow.add_edge("search_agent", "analysis_agent")
    workflow.add_conditional_edges(
        "analysis_agent",
        route_after_analysis,
        {"search_agent": "search_agent", "writer_agent": "writer_agent"},
    )
    workflow.add_edge("writer_agent", "manager_review")
    workflow.add_conditional_edges(
        "manager_review",
        route_after_review,
        {"writer_agent": "writer_agent", END: END},
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public streaming interface
# ---------------------------------------------------------------------------

def run_research_stream(
    topic: str,
    groq_api_key: str,
    tavily_api_key: str,
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0.2,
) -> Generator[dict[str, Any], None, None]:
    """Execute the research workflow, yielding state updates after each node."""
    app = build_research_graph(groq_api_key, tavily_api_key, model_name, temperature)

    initial_state: ResearchState = make_initial_state(topic)

    try:
        yield from app.stream(initial_state)
    except Exception as e:
        yield {"error": str(e)}
