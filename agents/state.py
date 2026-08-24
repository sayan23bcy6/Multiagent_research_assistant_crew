from typing import TypedDict, List, Dict, Any, Optional

class AgentLog(TypedDict):
    agent: str
    status: str
    message: str
    data: Optional[Any]

class ResearchState(TypedDict):
    topic: str
    plan: Optional[str]
    sources: Optional[str]
    raw_sources_list: Optional[List[Dict[str, Any]]]
    claims_analysis: Optional[str]
    draft_report: Optional[str]
    final_report: Optional[str]
    agent_logs: List[Dict[str, Any]]
    error: Optional[str]
