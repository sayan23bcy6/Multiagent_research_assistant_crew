"""
Multiagent Research Assistant Crew package.

Exports only state types at package import time to keep __init__ lightweight.
The full graph (build_research_graph, run_research_stream) is imported lazily
in app.py to avoid pulling in heavy dependencies (langchain_groq, langgraph)
when only type hints are needed.
"""

from agents.state import AgentLog, ManagerPlan, ResearchState, TavilySource

__all__ = ["ResearchState", "AgentLog", "ManagerPlan", "TavilySource"]
