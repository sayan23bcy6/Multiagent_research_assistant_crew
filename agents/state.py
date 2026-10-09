"""
State definitions for the Research Assistant Crew.

ResearchState is the single source of truth flowing through the LangGraph pipeline.
AgentLog is now used to type the agent_logs list (item 7 / item 8).
"""

from typing import Any, TypedDict


class AgentLog(TypedDict):
    """Structured log entry produced by each agent node."""

    agent: str
    status: str
    message: str
    duration_seconds: float | None
    token_count: int | None
    data: Any | None


class ManagerPlan(TypedDict):
    """Parsed structured plan from the Manager Agent (item 3)."""

    core_objective: str
    sub_tasks: list[str]
    search_queries: list[str]


class TavilySource(TypedDict):
    """Single Tavily search result kept as structured data (item 1)."""

    title: str
    url: str
    content: str  # truncated to per-source token budget


class ResearchState(TypedDict):
    topic: str
    plan: str | None  # raw markdown plan text for display
    parsed_plan: ManagerPlan | None  # structured plan (item 3)
    sources: str | None  # curated markdown for display
    raw_sources_list: list[TavilySource]  # structured Tavily results (item 1)
    claims_analysis: str | None
    analyst_claims: list[str]  # individual claim strings extracted (item 3 / 4)
    draft_report: str | None
    final_report: str | None
    url_validation_warnings: list[str]  # unmatched URLs found in report (item 1)
    agent_logs: list[AgentLog]  # typed log entries (item 8)
    error: str | None
    # Revision-loop counters (item 4)
    search_retry_count: int
    writer_retry_count: int
    # Finish-reason tracking (item 5)
    manager_review_truncated: bool
