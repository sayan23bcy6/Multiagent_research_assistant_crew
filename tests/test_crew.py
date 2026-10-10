"""
Tests for the Research Assistant Crew.

Covers:
- Retry logic in invoke_with_retry
- URL validation
- Manager plan parsing
- Manager review verdict parsing
- Conditional routing and counters
- Tavily error handling and deduplication
- check_report_structure checks
- merge_sources behavior
- LangGraph graph integration tests (mocked LLM + search)
"""

import collections
import json
import types
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END

from agents.crew import (
    build_research_graph,
    check_report_structure,
    count_cited_source_indices,
    execute_tavily_search,
    extract_urls_from_text,
    invoke_with_retry,
    make_initial_state,
    merge_sources,
    parse_manager_plan,
    parse_manager_verdict,
    route_after_analysis,
    route_after_review,
    run_research_stream,
    strip_verdict_block,
    validate_report_urls,
)
from agents.state import TavilySource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_sources() -> list[TavilySource]:
    return [
        TavilySource(
            title="Alpha Paper",
            url="https://alpha.example.com/paper",
            content="Quantum error rate reduced to 0.1%...",
        ),
        TavilySource(
            title="Beta Blog",
            url="https://beta.example.com/post",
            content="New benchmark shows 99.9% fidelity...",
        ),
        TavilySource(
            title="Gamma Docs",
            url="https://gamma.example.com/docs",
            content="Algorithm X achieves O(n log n)...",
        ),
    ]


@pytest.fixture()
def sample_plan_json() -> str:
    return json.dumps({
        "core_objective": "Analyse quantum error correction advances.",
        "sub_tasks": [
            "Review hardware benchmarks",
            "Examine error rates",
            "Assess commercial timelines",
        ],
        "search_queries": [
            "quantum error correction 2025",
            "qubit fidelity benchmarks",
            "quantum computing commercial roadmap",
        ],
    })


# ---------------------------------------------------------------------------
# Item 1: URL validation
# ---------------------------------------------------------------------------

class TestUrlValidation:
    def test_extract_markdown_url(self):
        text = "See [Alpha Paper](https://alpha.example.com/paper) for details."
        urls = extract_urls_from_text(text)
        assert "https://alpha.example.com/paper" in urls

    def test_extract_bare_url(self):
        text = "Visit https://bare.example.com for info."
        urls = extract_urls_from_text(text)
        assert "https://bare.example.com" in urls

    def test_validate_report_urls_all_valid(self, sample_sources):
        report = (
            "See [Alpha Paper](https://alpha.example.com/paper) and "
            "[Beta Blog](https://beta.example.com/post)."
        )
        warnings = validate_report_urls(report, sample_sources)
        assert warnings == []

    def test_validate_report_urls_hallucinated(self, sample_sources):
        report = "Citing [Fake Source](https://hallucinated.example.com/fake)."
        warnings = validate_report_urls(report, sample_sources)
        assert len(warnings) == 1
        assert "hallucinated.example.com" in warnings[0]

    def test_validate_report_urls_mixed(self, sample_sources):
        report = (
            "[Alpha Paper](https://alpha.example.com/paper) is good, "
            "but also [Fake](https://evil.example.com/x)."
        )
        warnings = validate_report_urls(report, sample_sources)
        assert len(warnings) == 1
        assert "evil.example.com" in warnings[0]


# ---------------------------------------------------------------------------
# Item 3: Manager plan parsing
# ---------------------------------------------------------------------------

class TestManagerPlanParsing:
    def test_parse_valid_json(self, sample_plan_json):
        plan = parse_manager_plan(sample_plan_json)
        assert plan is not None
        assert plan.core_objective == "Analyse quantum error correction advances."
        assert len(plan.sub_tasks) == 3
        assert len(plan.search_queries) == 3

    def test_parse_json_with_markdown_fences(self, sample_plan_json):
        fenced = f"```json\n{sample_plan_json}\n```"
        plan = parse_manager_plan(fenced)
        assert plan is not None
        assert plan.search_queries[0] == "quantum error correction 2025"

    def test_parse_invalid_json_returns_none(self):
        plan = parse_manager_plan("This is not JSON at all.")
        assert plan is None

    def test_parse_missing_field_returns_none(self):
        bad = json.dumps({"core_objective": "x", "sub_tasks": []})  # missing search_queries
        plan = parse_manager_plan(bad)
        assert plan is None

    def test_parse_json_embedded_in_text(self, sample_plan_json):
        text = f"Here is the plan:\n{sample_plan_json}\nEnd of plan."
        plan = parse_manager_plan(text)
        assert plan is not None


# ---------------------------------------------------------------------------
# Item 4: Retry logic — invoke_with_retry
# ---------------------------------------------------------------------------

class TestInvokeWithRetry:
    def test_succeeds_on_first_attempt(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="ok")
        result = invoke_with_retry(llm, ["msg"], max_retries=3)
        assert result.content == "ok"
        assert llm.invoke.call_count == 1

    def test_retries_on_429_then_succeeds(self):
        llm = MagicMock()
        rate_error = Exception("429 rate_limit exceeded, try again in 2s")
        llm.invoke.side_effect = [rate_error, rate_error, MagicMock(content="ok")]

        with patch("agents.crew.time.sleep"):  # don't actually sleep
            result = invoke_with_retry(llm, ["msg"], max_retries=5, initial_wait=0.1)

        assert result.content == "ok"
        assert llm.invoke.call_count == 3

    def test_raises_after_max_retries(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("429 rate_limit hit")

        with patch("agents.crew.time.sleep"):
            with pytest.raises(Exception, match="429"):
                invoke_with_retry(llm, ["msg"], max_retries=3, initial_wait=0.1)

    def test_raises_non_rate_limit_immediately(self):
        llm = MagicMock()
        llm.invoke.side_effect = ValueError("Some other error")

        with pytest.raises(ValueError):
            invoke_with_retry(llm, ["msg"], max_retries=5)

        assert llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
# Item 4: Conditional routing helpers
# ---------------------------------------------------------------------------

class TestConditionalRouting:
    def test_count_cited_indices_none(self):
        assert count_cited_source_indices("No sources cited here.") == 0

    def test_count_cited_indices_multiple(self):
        text = "Source #1 confirms this. Source #3 adds detail. Source #1 repeated."
        assert count_cited_source_indices(text) == 2  # unique: 1, 3

    def test_count_cited_indices_case_insensitive(self):
        text = "source #2 and SOURCE #4 both confirm."
        assert count_cited_source_indices(text) == 2


# ---------------------------------------------------------------------------
# Item 4: Manager verdict parsing
# ---------------------------------------------------------------------------

class TestManagerVerdict:
    def test_parse_approve_verdict(self):
        review = 'Great report.\n```verdict\n{"verdict": "approve", "reasons": []}\n```'
        v = parse_manager_verdict(review)
        assert v["verdict"] == "approve"

    def test_parse_reject_verdict(self):
        review = 'Needs work.\n```verdict\n{"verdict": "reject", "reasons": ["Missing evidence"]}\n```'
        v = parse_manager_verdict(review)
        assert v["verdict"] == "reject"
        assert "Missing evidence" in v["reasons"]

    def test_missing_verdict_defaults_approve(self):
        review = "Report looks good."
        v = parse_manager_verdict(review)
        assert v["verdict"] == "approve"

    def test_bad_json_verdict_defaults_approve(self):
        review = "Report text.\n```verdict\nnot-valid-json\n```"
        v = parse_manager_verdict(review)
        assert v["verdict"] == "approve"
        assert v["reasons"] == []

    def test_strip_verdict_block(self):
        review = 'Report text.\n```verdict\n{"verdict": "approve", "reasons": []}\n```'
        stripped = strip_verdict_block(review)
        assert "```verdict" not in stripped
        assert "Report text." in stripped


# ---------------------------------------------------------------------------
# Item 1: No-silent-fallback — Tavily search error handling
# ---------------------------------------------------------------------------

class TestTavilySearchFallback:
    def test_invalid_api_key_raises(self):
        with patch("agents.crew.TavilyClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.search.side_effect = Exception("401 unauthorized: invalid api key")
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Tavily API Error"):
                execute_tavily_search("bad-key", ["quantum computing"])

    def test_other_error_is_swallowed_and_empty_returned(self):
        with patch("agents.crew.TavilyClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.search.side_effect = Exception("Network timeout")
            mock_client_cls.return_value = mock_client

            results = execute_tavily_search("good-key", ["quantum computing"])
            assert results == []

    def test_deduplicates_by_url(self):
        with patch("agents.crew.TavilyClient") as mock_client_cls:
            mock_client = MagicMock()
            dup_result = {
                "results": [
                    {"url": "https://dup.example.com", "title": "Dup", "content": "x", "raw_content": "x"},
                    {"url": "https://dup.example.com", "title": "Dup", "content": "x", "raw_content": "x"},
                    {"url": "https://unique.example.com", "title": "Unique", "content": "y", "raw_content": "y"},
                ]
            }
            mock_client.search.return_value = dup_result
            mock_client_cls.return_value = mock_client

            results = execute_tavily_search("key", ["test"])
            urls = [r["url"] for r in results]
            assert len(urls) == len(set(urls))  # no duplicates


# ---------------------------------------------------------------------------
# Structure verification tests (replacing TestGoldenOutputStructure)
# ---------------------------------------------------------------------------

GOOD_REPORT = """## Executive Summary

Quantum computing has achieved error rates below 1% through surface codes.
Commercial deployment of fault-tolerant systems is expected within 5 years.

## Hardware Benchmarks

Details about qubits and fidelity...

## Error Correction Approaches

Surface codes are dominant...

## Commercial Timeline

Companies plan deployment by 2029...

## Sources

1. [Alpha Paper](https://alpha.example.com/paper)
2. [Beta Blog](https://beta.example.com/post)
"""

REPORT_NO_SUMMARY = """## Hardware Benchmarks

Details about qubits and fidelity...

## Error Correction Approaches

Surface codes are dominant...

## Commercial Timeline

Companies plan deployment by 2029...

## Sources

1. [Alpha Paper](https://alpha.example.com/paper)
"""

REPORT_TWO_SECTIONS = """## Executive Summary

Quantum computing has achieved error rates below 1% through surface codes.
Commercial deployment of fault-tolerant systems is expected within 5 years.

## Hardware Benchmarks

Details about qubits and fidelity...

## Error Correction Approaches

Surface codes are dominant...

## Sources

1. [Alpha Paper](https://alpha.example.com/paper)
"""

REPORT_NO_SOURCES = """## Executive Summary

Quantum computing has achieved error rates below 1% through surface codes.
Commercial deployment of fault-tolerant systems is expected within 5 years.

## Hardware Benchmarks

Details about qubits and fidelity...

## Error Correction Approaches

Surface codes are dominant...

## Commercial Timeline

Companies plan deployment by 2029...
"""


class TestReportStructure:
    def test_good_report_passes(self):
        problems = check_report_structure(GOOD_REPORT)
        assert problems == []

    def test_missing_summary_detected(self):
        problems = check_report_structure(REPORT_NO_SUMMARY)
        assert any("Executive Summary" in p for p in problems)

    def test_two_content_sections_detected(self):
        problems = check_report_structure(REPORT_TWO_SECTIONS)
        assert any("Expected between 3 and 5 content sections" in p and "2" in p for p in problems)

    def test_missing_sources_detected(self):
        problems = check_report_structure(REPORT_NO_SOURCES)
        assert any("Sources" in p for p in problems)


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

class TestRouting:
    def test_route_after_analysis_too_few_sources(self):
        state = make_initial_state("quantum")
        state["claims_analysis"] = "Source #1 confirms finding."
        state["search_retry_count"] = 0
        assert route_after_analysis(state) == "search_agent"

    def test_route_after_analysis_retry_cap_reached(self):
        state = make_initial_state("quantum")
        state["claims_analysis"] = "Source #1 confirms finding."
        state["search_retry_count"] = 2
        assert route_after_analysis(state) == "writer_agent"

    def test_route_after_analysis_sufficient_sources(self):
        state = make_initial_state("quantum")
        state["claims_analysis"] = "Source #1, Source #2, Source #3 all confirm."
        state["search_retry_count"] = 0
        assert route_after_analysis(state) == "writer_agent"

    def test_route_after_review_reject_retry_zero(self):
        state = make_initial_state("quantum")
        state["review_verdict"] = "reject"
        state["writer_retry_count"] = 0
        assert route_after_review(state) == "writer_agent"

    def test_route_after_review_reject_retry_cap(self):
        state = make_initial_state("quantum")
        state["review_verdict"] = "reject"
        state["writer_retry_count"] = 1
        assert route_after_review(state) == END

    def test_route_after_review_approve(self):
        state = make_initial_state("quantum")
        state["review_verdict"] = "approve"
        state["writer_retry_count"] = 0
        assert route_after_review(state) == END


# ---------------------------------------------------------------------------
# merge_sources tests
# ---------------------------------------------------------------------------

class TestMergeSources:
    def test_dedupe_by_url(self):
        s1 = TavilySource(title="S1", url="https://example.com/1", content="c1")
        s2 = TavilySource(title="S2", url="https://example.com/1", content="c2")
        merged = merge_sources([s1], [s2])
        assert len(merged) == 1
        assert merged[0]["title"] == "S1"

    def test_order_preserved(self):
        s1 = TavilySource(title="S1", url="https://example.com/1", content="c1")
        s2 = TavilySource(title="S2", url="https://example.com/2", content="c2")
        s3 = TavilySource(title="S3", url="https://example.com/3", content="c3")
        merged = merge_sources([s1, s2], [s3, s1])
        assert [s["url"] for s in merged] == [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]

    def test_cap_respected(self):
        sources = [
            TavilySource(title=f"S{i}", url=f"https://example.com/{i}", content=f"c{i}")
            for i in range(10)
        ]
        merged = merge_sources(sources[:5], sources[5:], limit=8)
        assert len(merged) == 8


# ---------------------------------------------------------------------------
# Graph integration tests with FakeLLM
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, cited_sources: int = 3, always_reject: bool = False):
        self.cited_sources = cited_sources
        self.always_reject = always_reject
        self.writer_prompts: list[str] = []
        self.role_calls: collections.Counter = collections.Counter()

    def invoke(self, messages):
        system = messages[0].content
        if "core_objective" in system:
            self.role_calls["plan"] += 1
            content = json.dumps({
                "core_objective": "Understand the quantum landscape.",
                "sub_tasks": [
                    "Hardware benchmarks",
                    "Error correction",
                    "Commercial roadmaps",
                ],
                "search_queries": [
                    "quantum computing benchmarks",
                    "surface codes",
                    "quantum hardware roadmaps",
                ],
            })
        elif "Critical Research Analyst" in system:
            self.role_calls["analysis"] += 1
            claim_blocks = []
            for i in range(1, self.cited_sources + 1):
                claim_blocks.append(
                    f"**Claim**: Empirical finding {i}.\n"
                    f"- Supporting Evidence: Evidence from Source #{i}.\n"
                    f"- Source Citation: Source #{i}\n"
                )
            content = "\n\n".join(claim_blocks)
        elif "Senior Research Librarian" in system:
            self.role_calls["search"] += 1
            content = "**Source #1** — Alpha ...\n**Source #2** — Beta ...\n**Source #3** — Gamma ..."
        elif "Technical Report Writer" in system:
            self.role_calls["writer"] += 1
            self.writer_prompts.append(messages[-1].content)
            content = (
                "## Executive Summary\n\n"
                "Quantum computing has achieved critical error mitigation milestones. "
                "Hardware roadmap targets point toward fault tolerance by 2030.\n\n"
                "## Hardware Benchmarks\n\n"
                "Qubit fidelity exceeds thresholds.\n\n"
                "## Error Correction Approaches\n\n"
                "Surface code implementations demonstrate logical qubit gain.\n\n"
                "## Commercial Timeline\n\n"
                "Leading vendors target utility scale by end of decade.\n\n"
                "## Sources\n\n"
                "1. [Alpha](https://alpha.example.com/a)\n"
                "2. [Beta](https://beta.example.com/b)\n"
            )
        elif "```verdict" in system:
            self.role_calls["review"] += 1
            report_body = (
                "## Executive Summary\n\n"
                "Quantum computing has achieved critical error mitigation milestones. "
                "Hardware roadmap targets point toward fault tolerance by 2030.\n\n"
                "## Hardware Benchmarks\n\n"
                "Qubit fidelity exceeds thresholds.\n\n"
                "## Error Correction Approaches\n\n"
                "Surface code implementations demonstrate logical qubit gain.\n\n"
                "## Commercial Timeline\n\n"
                "Leading vendors target utility scale by end of decade.\n\n"
                "## Sources\n\n"
                "1. [Alpha](https://alpha.example.com/a)\n"
                "2. [Beta](https://beta.example.com/b)\n"
            )
            if self.always_reject:
                verdict_json = json.dumps({
                    "verdict": "reject",
                    "reasons": ["needs more evidence"],
                })
            else:
                verdict_json = json.dumps({
                    "verdict": "approve",
                    "reasons": [],
                })
            content = f"{report_body}\n\n```verdict\n{verdict_json}\n```"
        else:
            raise ValueError(f"Unknown system message: {system[:100]}")

        return types.SimpleNamespace(
            content=content,
            usage_metadata={"total_tokens": 10},
            response_metadata={"finish_reason": "stop"},
        )


class TestGraphIntegration:
    def test_retry_path_always_reject_cited_sources_1(self):
        """Test A (retry path, always reject, cited_sources=1)."""
        fake = FakeLLM(cited_sources=1, always_reject=True)
        with patch("agents.crew.get_llm", return_value=fake), patch(
            "agents.crew.TavilyClient"
        ) as mock_tavily_cls:
            mock_tavily = MagicMock()
            mock_tavily.search.return_value = {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://alpha.example.com/a",
                        "content": "Alpha content",
                        "raw_content": "Alpha raw",
                    },
                    {
                        "title": "Beta",
                        "url": "https://beta.example.com/b",
                        "content": "Beta content",
                        "raw_content": "Beta raw",
                    },
                    {
                        "title": "Gamma",
                        "url": "https://gamma.example.com/c",
                        "content": "Gamma content",
                        "raw_content": "Gamma raw",
                    },
                ]
            }
            mock_tavily_cls.return_value = mock_tavily

            app = build_research_graph("dummy_groq", "dummy_tavily")
            final = app.invoke(make_initial_state("quantum"))

            assert fake.role_calls["search"] == 3
            assert fake.role_calls["analysis"] == 3
            assert fake.role_calls["writer"] == 2
            assert fake.role_calls["review"] == 2
            assert final["search_retry_count"] == 2
            assert final["writer_retry_count"] == 1
            assert final["final_report"]
            assert "```verdict" not in final["final_report"]
            assert "needs more evidence" in fake.writer_prompts[1]

    def test_approve_path_cited_sources_3(self):
        """Test B (approve path, cited_sources=3)."""
        fake = FakeLLM(cited_sources=3, always_reject=False)
        with patch("agents.crew.get_llm", return_value=fake), patch(
            "agents.crew.TavilyClient"
        ) as mock_tavily_cls:
            mock_tavily = MagicMock()
            mock_tavily.search.return_value = {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://alpha.example.com/a",
                        "content": "Alpha content",
                        "raw_content": "Alpha raw",
                    },
                    {
                        "title": "Beta",
                        "url": "https://beta.example.com/b",
                        "content": "Beta content",
                        "raw_content": "Beta raw",
                    },
                    {
                        "title": "Gamma",
                        "url": "https://gamma.example.com/c",
                        "content": "Gamma content",
                        "raw_content": "Gamma raw",
                    },
                ]
            }
            mock_tavily_cls.return_value = mock_tavily

            app = build_research_graph("dummy_groq", "dummy_tavily")
            final = app.invoke(make_initial_state("quantum"))

            assert fake.role_calls["search"] == 1
            assert fake.role_calls["analysis"] == 1
            assert fake.role_calls["writer"] == 1
            assert fake.role_calls["review"] == 1
            assert final["search_retry_count"] == 0
            assert final["writer_retry_count"] == 0

    def test_call_cap_aborts(self, monkeypatch):
        """Test C (cap): monkeypatch MAX_TOTAL_LLM_CALLS to 3, stream aborts with error."""
        monkeypatch.setattr("agents.crew.MAX_TOTAL_LLM_CALLS", 3)
        fake = FakeLLM(cited_sources=3, always_reject=False)
        with patch("agents.crew.get_llm", return_value=fake), patch(
            "agents.crew.TavilyClient"
        ) as mock_tavily_cls:
            mock_tavily = MagicMock()
            mock_tavily.search.return_value = {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://alpha.example.com/a",
                        "content": "Alpha content",
                        "raw_content": "Alpha raw",
                    },
                    {
                        "title": "Beta",
                        "url": "https://beta.example.com/b",
                        "content": "Beta content",
                        "raw_content": "Beta raw",
                    },
                    {
                        "title": "Gamma",
                        "url": "https://gamma.example.com/c",
                        "content": "Gamma content",
                        "raw_content": "Gamma raw",
                    },
                ]
            }
            mock_tavily_cls.return_value = mock_tavily

            events = list(run_research_stream("quantum", "dummy_k", "dummy_t"))
            assert "error" in events[-1]
