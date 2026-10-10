"""
Evaluation harness for the Research Assistant Crew (item 12).

Usage:
    python evals/run_evals.py

Scores each question on:
  (a) URL integrity: % of cited URLs that match Tavily result set.
  (b) Report structure: has Executive Summary, ≥2 content sections, Sources section.
  (c) Key-point coverage: % of expected key points found (simple keyword match).

Results are written to evals/results.md.
Do NOT run in CI — requires live API keys.
"""

import datetime
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow running from repo root or evals/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agents.crew import check_report_structure, run_research_stream  # noqa: E402

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"
RESULTS_FILE = Path(__file__).parent / "results.md"


def load_questions() -> list[dict[str, Any]]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def score_url_integrity(final_report: str, raw_sources: list[dict]) -> float:
    """(a) % of cited URLs that exist in the Tavily result set."""
    from agents.crew import extract_urls_from_text
    report_urls = extract_urls_from_text(final_report)
    if not report_urls:
        return 1.0  # no URLs cited — nothing to check
    allowed_urls = {s["url"] for s in raw_sources}
    matched = sum(1 for u in report_urls if u in allowed_urls)
    return matched / len(report_urls)


def score_structure(final_report: str) -> bool:
    """(b) True if report has Executive Summary, 3–5 content sections, and Sources."""
    return not check_report_structure(final_report)


def score_key_points(final_report: str, key_points: list[str]) -> float:
    """(c) % of expected key points found via simple keyword match (documented heuristic)."""
    if not key_points:
        return 1.0
    report_lower = final_report.lower()
    found = sum(1 for kp in key_points if kp.lower() in report_lower)
    return found / len(key_points)


def run_eval(question: dict[str, Any], groq_key: str, tavily_key: str, model: str) -> dict[str, Any]:
    topic = question["question"]
    key_points = question.get("key_points", [])

    state_store: dict[str, Any] = {
        "final_report": "",
        "raw_sources_list": [],
        "url_validation_warnings": [],
    }

    print(f"\n{'='*60}")
    print(f"  Q: {topic[:80]}")

    for step_output in run_research_stream(topic, groq_key, tavily_key, model):
        if "error" in step_output:
            return {"question": topic, "error": step_output["error"]}
        if "search_agent" in step_output:
            state_store["raw_sources_list"] = step_output["search_agent"].get("raw_sources_list", [])
        if "manager_review" in step_output:
            state_store["final_report"] = step_output["manager_review"].get("final_report", "")
            state_store["url_validation_warnings"] = step_output["manager_review"].get("url_validation_warnings", [])

    report = state_store["final_report"]
    raw_sources = state_store["raw_sources_list"]

    url_score = score_url_integrity(report, raw_sources)
    struct_ok = score_structure(report)
    kp_score = score_key_points(report, key_points)

    return {
        "question": topic,
        "url_integrity": round(url_score, 3),
        "structure_ok": struct_ok,
        "key_point_coverage": round(kp_score, 3),
        "url_warnings": len(state_store["url_validation_warnings"]),
    }


def write_results(results: list[dict[str, Any]], model: str) -> None:
    date_str = datetime.date.today().isoformat()
    lines = [
        "# Evaluation Results",
        "",
        f"**Date:** {date_str}  ",
        f"**Model:** `{model}`",
        "",
        "> ⚠️ Key-point coverage uses simple keyword matching — it is a rough heuristic, not a semantic similarity score.",
        "",
        "| # | Question (truncated) | URL Integrity | Structure OK | Key-Point Coverage | URL Warnings |",
        "|---|---------------------|:---:|:---:|:---:|:---:|",
    ]
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"| {i} | {r['question'][:60]} | ERROR | ERROR | ERROR | — |")
        else:
            struct_icon = "✅" if r["structure_ok"] else "❌"
            lines.append(
                f"| {i} | {r['question'][:60]} "
                f"| {r['url_integrity']:.0%} "
                f"| {struct_icon} "
                f"| {r['key_point_coverage']:.0%} "
                f"| {r['url_warnings']} |"
            )

    n = len(results)
    struct_count = sum(1 for r in results if r.get("structure_ok"))
    good = [r for r in results if "error" not in r]
    lines += [
        "",
        "## Summary",
        f"- **Avg URL Integrity:** {(sum(r.get('url_integrity', 0) for r in good) / max(len(good), 1)):.0%}",
        f"- **Structure Pass Rate:** {struct_count}/{n}",
        f"- **Avg Key-Point Coverage:** {(sum(r.get('key_point_coverage', 0) for r in good) / max(len(good), 1)):.0%}",
    ]

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {RESULTS_FILE}")


def main() -> None:
    groq_key = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    if not groq_key or not tavily_key:
        print("ERROR: GROQ_API_KEY and TAVILY_API_KEY must be set in environment.")
        sys.exit(1)

    questions = load_questions()
    results = []
    for q in questions:
        result = run_eval(q, groq_key, tavily_key, model)
        print(
            f"  URL: {result.get('url_integrity','ERR'):.0%}  "
            f"Struct: {'✅' if result.get('structure_ok') else '❌'}  "
            f"KP: {result.get('key_point_coverage','ERR'):.0%}"
            if "error" not in result
            else f"  ERROR: {result['error']}"
        )
        results.append(result)

    write_results(results, model)


if __name__ == "__main__":
    main()
