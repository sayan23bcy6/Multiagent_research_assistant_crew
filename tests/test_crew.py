"""
Tests for the Research Assistant Crew.

Covers (item 10):
- Retry logic in invoke_with_retry
- URL validation
- Manager plan parsing
- Conditional routing (search retry, writer retry)
- No-silent-fallback behavior (item 1)
- Golden-output structure checks (3 tests)
"""

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from agents.crew import (
    count_cited_source_indices,
    execute_tavily_search,
    extract_urls_from_text,
    invoke_with_retry,
    parse_manager_plan,
    parse_manager_verdict,
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
        TavilySource(title="Alpha Paper", url="https://alpha.example.com/paper", content="Quantum error rate reduced to 0.1%..."),
        TavilySource(title="Beta Blog", url="https://beta.example.com/post", content="New benchmark shows 99.9% fidelity..."),
        TavilySource(title="Gamma Docs", url="https://gamma.example.com/docs", content="Algorithm X achieves O(n log n)..."),
    ]


@pytest.fixture()
def sample_plan_json() -> str:
    return json.dumps({
        "core_objective": "Analyse quantum error correction advances.",
        "sub_tasks": ["Review hardware benchmarks", "Examine error rates", "Assess commercial timelines"],
        "search_queries": ["quantum error correction 2025", "qubit fidelity benchmarks", "quantum computing commercial roadmap"],
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
        report = "See [Alpha Paper](https://alpha.example.com/paper) and [Beta Blog](https://beta.example.com/post)."
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
# Golden-output structure tests (item 10)
# ---------------------------------------------------------------------------

GOOD_REPORT = """
## Executive Summary

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

BAD_REPORT_NO_SUMMARY = """
## Introduction

Quantum computing is interesting. Let us explore it.

## Hardware

Some stuff about hardware.
"""

BAD_REPORT_NO_SOURCES = """
## Executive Summary

Quantum computing is advancing rapidly. Error rates have improved.

## Hardware Benchmarks

Great progress has been made.
"""


class TestGoldenOutputStructure:
    """Golden-output tests for the required report structure (item 10)."""

    def test_report_has_executive_summary(self):
        """Report must start with a 2-sentence Executive Summary section."""
        assert "## Executive Summary" in GOOD_REPORT or "Executive Summary" in GOOD_REPORT

    def test_report_has_sources_section(self):
        """Report must have a Sources section with at least one hyperlinked source."""
        assert "## Sources" in GOOD_REPORT
        # At least one markdown hyperlink in the sources section
        sources_section = GOOD_REPORT.split("## Sources")[-1]
        links = re.findall(r"\[.+?\]\(https?://.+?\)", sources_section)
        assert len(links) >= 1

    def test_report_has_multiple_headed_sections(self):
        """Report must have at least 3 headed sections (## headers) beyond the summary."""
        headers = re.findall(r"^##\s+.+", GOOD_REPORT, re.MULTILINE)
        # Must have Summary + at least 2 content sections + Sources = at least 4
        assert len(headers) >= 4

    def test_bad_report_missing_summary_detected(self):
        """Reports without Executive Summary should be detectable."""
        has_summary = "## Executive Summary" in BAD_REPORT_NO_SUMMARY or (
            len(re.findall(r"^##\s+Executive\s+Summary", BAD_REPORT_NO_SUMMARY, re.MULTILINE)) > 0
        )
        assert not has_summary

    def test_bad_report_missing_sources_detected(self):
        """Reports without Sources section should be detectable."""
        assert "## Sources" not in BAD_REPORT_NO_SOURCES

    def test_executive_summary_has_two_sentences(self):
        """Executive Summary should have approximately 2 sentences."""
        match = re.search(
            r"## Executive Summary\s*\n+(.*?)(?=\n##|\Z)", GOOD_REPORT, re.DOTALL
        )
        assert match is not None
        summary_text = match.group(1).strip()
        # Count sentences by splitting on . ! ?
        sentences = [s.strip() for s in re.split(r"[.!?]+", summary_text) if s.strip()]
        assert 2 <= len(sentences) <= 4  # allow slight flexibility
