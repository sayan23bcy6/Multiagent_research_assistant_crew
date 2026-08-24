import time
import re
from typing import Dict, Any, Generator, List
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from agents.state import ResearchState
from agents.prompts import (
    MANAGER_PLAN_SYSTEM_PROMPT,
    MANAGER_REVIEW_SYSTEM_PROMPT,
    SEARCH_AGENT_SYSTEM_PROMPT,
    ANALYSIS_AGENT_SYSTEM_PROMPT,
    WRITER_AGENT_SYSTEM_PROMPT,
)


def get_llm(groq_api_key: str, model: str = "openai/gpt-oss-120b", temperature: float = 0.2) -> ChatGroq:
    """Initialize Groq Chat LLM with bounded output tokens."""
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=2048,
    )


def invoke_with_retry(llm: ChatGroq, messages: List[Any], max_retries: int = 5, initial_wait: float = 3.0) -> Any:
    """Invoke LLM with automatic exponential backoff on 429 TPM/RPM rate limits."""
    wait_time = initial_wait
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "rate_limit" in err_str.lower() or "tpm" in err_str.lower()) and attempt < max_retries:
                # Extract wait time from error if present (e.g. "try again in 5.2s")
                match = re.search(r"try again in ([\d\.]+)s", err_str, re.IGNORECASE)
                if match:
                    parsed_wait = float(match.group(1)) + 1.0
                    sleep_duration = max(wait_time, parsed_wait)
                else:
                    sleep_duration = wait_time
                time.sleep(sleep_duration)
                wait_time *= 1.8
            else:
                raise e


def execute_tavily_search(tavily_api_key: str, queries: list[str], max_results: int = 3) -> list[Dict[str, Any]]:
    """Execute search across queries via Tavily client with compact snippets to respect TPM limits."""
    client = TavilyClient(api_key=tavily_api_key)
    all_results = []
    seen_urls = set()

    for q in queries[:2]:
        clean_q = q.strip().strip('"').strip("'").strip("`").strip("*")
        if clean_q.startswith(("- ", "* ", "1. ", "2. ", "3. ", "• ")):
            clean_q = clean_q[2:].strip().strip('"').strip("'")
        if not clean_q or len(clean_q) < 3:
            continue
        try:
            response = client.search(
                query=clean_q,
                search_depth="basic",
                max_results=max_results,
                include_raw_content=False,
            )
            results = response.get("results", [])
            for r in results:
                url = r.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    # Keep snippet very compact (~200 chars) to strictly preserve TPM budget
                    snippet = r.get("content", "")
                    if len(snippet) > 220:
                        snippet = snippet[:220] + "..."
                    all_results.append({
                        "title": r.get("title", "Untitled Source"),
                        "url": url,
                        "content": snippet,
                    })
        except Exception as e:
            err_msg = str(e)
            if "unauthorized" in err_msg.lower() or "invalid api key" in err_msg.lower() or "401" in err_msg:
                raise ValueError(f"Tavily API Error: Invalid API Key or Unauthorized ({err_msg})")
            continue

    return all_results[:6]


def build_research_graph(groq_api_key: str, tavily_api_key: str, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.2):
    """Construct the LangGraph state graph for the multiagent crew."""
    llm = get_llm(groq_api_key, model=model_name, temperature=temperature)

    # 1. MANAGER PLAN NODE
    def manager_plan_node(state: ResearchState) -> Dict[str, Any]:
        topic = state["topic"]
        messages = [
            SystemMessage(content=MANAGER_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=f"Research Question/Topic: {topic}\n\nPlease formulate a concise research delegation plan and 2 recommended search queries."),
        ]
        response = invoke_with_retry(llm, messages)
        plan_text = response.content

        log_entry = {
            "agent": "Manager Agent",
            "status": "Plan Created",
            "message": "Deconstructed topic into sub-tasks and delegated search objectives.",
            "data": plan_text,
        }
        return {
            "plan": plan_text,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # 2. SEARCH AGENT NODE (Senior Research Librarian)
    def search_node(state: ResearchState) -> Dict[str, Any]:
        topic = state["topic"]
        plan = state.get("plan", "")

        # Pacing delay to avoid TPM bursts on rapid Groq responses
        time.sleep(1.5)

        # Extract search queries from topic + plan
        query_extraction_prompt = (
            f"Topic: {topic}\nPlan:\n{plan}\n\n"
            "Extract 2 concise search keywords or phrases suitable for a web search engine.\n"
            "Return ONLY the 2 search queries, one per line, with NO other text or numbers."
        )
        query_response = invoke_with_retry(llm, [HumanMessage(content=query_extraction_prompt)])
        raw_lines = [line.strip().strip('"').strip("'") for line in query_response.content.split("\n") if line.strip()]
        
        candidate_queries = [topic] + [l for l in raw_lines if not l.lower().startswith("here") and not l.startswith("#")]
        
        # Execute Tavily Search
        search_results = execute_tavily_search(tavily_api_key, candidate_queries[:2], max_results=3)

        if not search_results:
            search_results = execute_tavily_search(tavily_api_key, [topic], max_results=4)

        if search_results:
            formatted_results_text = "\n\n".join([
                f"Source #{i+1}:\n- Title: {r['title']}\n- URL: {r['url']}\n- Snippet: {r['content']}"
                for i, r in enumerate(search_results)
            ])
        else:
            formatted_results_text = f"Primary Topic Overview for '{topic}'."

        time.sleep(1.5)

        # Senior Research Librarian curation
        curation_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Candidate Sources Found via Web Search:\n{formatted_results_text}\n\n"
            "As the Senior Research Librarian, evaluate the sources above and formulate the curated list of 5–8 credible sources/references in Markdown. "
            "For each source, provide Title, URL, a 1-line relevance note, and key factual extracts. Do NOT output a JSON tool call."
        )
        curation_response = invoke_with_retry(llm, [
            SystemMessage(content=SEARCH_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=curation_prompt),
        ])
        sources_text = curation_response.content.strip()

        if sources_text.startswith("{") and ("tool" in sources_text or "arguments" in sources_text):
            sources_text = f"### Curated Primary Sources\n\n{formatted_results_text}"

        log_entry = {
            "agent": "Search Agent",
            "status": "Sources Curated",
            "message": f"Found and verified {len(search_results)} candidate sources with Tavily. Curated top primary references.",
            "data": sources_text,
        }
        return {
            "sources": sources_text,
            "raw_sources_list": search_results,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # 3. ANALYSIS AGENT NODE (Critical Research Analyst)
    def analysis_node(state: ResearchState) -> Dict[str, Any]:
        topic = state["topic"]
        sources = state.get("sources", "")

        time.sleep(1.5)

        analysis_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Curated Sources & Evidence from Senior Research Librarian:\n{sources}\n\n"
            "Perform critical analysis: extract the 3–5 strongest claims with empirical supporting evidence, "
            "exact source citations, and critical verification notes."
        )
        response = invoke_with_retry(llm, [
            SystemMessage(content=ANALYSIS_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=analysis_prompt),
        ])
        claims_text = response.content

        log_entry = {
            "agent": "Analysis Agent",
            "status": "Claims Extracted",
            "message": "Extracted key empirical claims, cross-referenced evidence, and verified source credibility.",
            "data": claims_text,
        }
        return {
            "claims_analysis": claims_text,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # 4. WRITER AGENT NODE (Technical Report Writer)
    def writer_node(state: ResearchState) -> Dict[str, Any]:
        topic = state["topic"]
        claims = state.get("claims_analysis", "")

        time.sleep(1.5)

        writer_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Critical Analyst's Claims & Evidence:\n{claims}\n\n"
            "Write the structured technical report: 2-sentence summary, 3–5 headed sections with deep technical evidence, "
            "and the complete sources list. No filler intros."
        )
        response = invoke_with_retry(llm, [
            SystemMessage(content=WRITER_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=writer_prompt),
        ])
        draft_text = response.content

        log_entry = {
            "agent": "Writer Agent",
            "status": "Draft Report Written",
            "message": "Synthesized findings into a structured technical briefing with 2-sentence summary, headed sections, and source list.",
            "data": draft_text,
        }
        return {
            "draft_report": draft_text,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # 5. MANAGER REVIEW NODE (Final Report & Polish)
    def manager_review_node(state: ResearchState) -> Dict[str, Any]:
        topic = state["topic"]
        draft = state.get("draft_report", "")

        time.sleep(1.5)

        review_prompt = (
            f"Research Topic: {topic}\n\n"
            f"Writer's Draft Report:\n{draft}\n\n"
            "Perform final quality assurance review. Ensure the 2-sentence summary is impactful, sections are logically structured, "
            "claims are properly cited, and output the final, polished publication-ready report in Markdown."
        )
        response = invoke_with_retry(llm, [
            SystemMessage(content=MANAGER_REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=review_prompt),
        ])
        final_text = response.content

        log_entry = {
            "agent": "Final Report",
            "status": "Report Finalized",
            "message": "Manager verified and signed off on the comprehensive research report.",
            "data": final_text,
        }
        return {
            "final_report": final_text,
            "agent_logs": state.get("agent_logs", []) + [log_entry],
        }

    # Construct Graph
    workflow = StateGraph(ResearchState)

    workflow.add_node("manager_plan", manager_plan_node)
    workflow.add_node("search_agent", search_node)
    workflow.add_node("analysis_agent", analysis_node)
    workflow.add_node("writer_agent", writer_node)
    workflow.add_node("manager_review", manager_review_node)

    workflow.add_edge(START, "manager_plan")
    workflow.add_edge("manager_plan", "search_agent")
    workflow.add_edge("search_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "writer_agent")
    workflow.add_edge("writer_agent", "manager_review")
    workflow.add_edge("manager_review", END)

    return workflow.compile()


def run_research_stream(
    topic: str,
    groq_api_key: str,
    tavily_api_key: str,
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0.2,
) -> Generator[Dict[str, Any], None, None]:
    """Execute research workflow yielding state updates after each agent node."""
    app = build_research_graph(groq_api_key, tavily_api_key, model_name, temperature)
    
    initial_state: ResearchState = {
        "topic": topic,
        "plan": None,
        "sources": None,
        "raw_sources_list": [],
        "claims_analysis": None,
        "draft_report": None,
        "final_report": None,
        "agent_logs": [],
        "error": None,
    }

    try:
        for output in app.stream(initial_state):
            # output is a dict like {'manager_plan': {...}} or {'search_agent': {...}}
            yield output
    except Exception as e:
        yield {"error": str(e)}
