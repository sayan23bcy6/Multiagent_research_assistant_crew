import os
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from agents.crew import run_research_stream
from agents.prompts import (
    SEARCH_AGENT_ROLE,
    SEARCH_AGENT_GOAL,
    SEARCH_AGENT_BACKSTORY,
    ANALYSIS_AGENT_ROLE,
    ANALYSIS_AGENT_GOAL,
    ANALYSIS_AGENT_BACKSTORY,
    WRITER_AGENT_ROLE,
    WRITER_AGENT_GOAL,
    WRITER_AGENT_BACKSTORY,
    MANAGER_AGENT_ROLE,
    MANAGER_AGENT_GOAL,
    MANAGER_AGENT_BACKSTORY,
)

# Page configuration
st.set_page_config(
    page_title="The Research Assistant Crew",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich dark cyber grid theme and card styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0c0f17;
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 36px 36px;
    }
    
    /* Header typography */
    .crew-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }
    
    .crew-subtext {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }
    
    /* Architecture Diagram Grid */
    .arch-container {
        background: #0f1422;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 28px;
        position: relative;
    }
    
    .agent-card {
        background: #0b0e17;
        border-radius: 8px;
        padding: 16px 18px;
        font-family: 'JetBrains Mono', monospace;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-2px);
    }
    
    .card-manager {
        border: 1.5px solid #ea580c;
        box-shadow: 0 0 15px rgba(234, 88, 12, 0.15);
    }
    
    .card-search {
        border: 1.5px solid #06b6d4;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.15);
    }
    
    .card-analysis {
        border: 1.5px solid #84cc16;
        box-shadow: 0 0 15px rgba(132, 204, 22, 0.15);
    }
    
    .card-writer {
        border: 1.5px solid #eab308;
        box-shadow: 0 0 15px rgba(234, 179, 8, 0.15);
    }
    
    .card-final {
        border: 1.5px solid #ec4899;
        box-shadow: 0 0 15px rgba(236, 72, 153, 0.15);
    }
    
    .card-title-manager { color: #fb923c; font-weight: 700; font-size: 0.95rem; }
    .card-title-search { color: #22d3ee; font-weight: 700; font-size: 0.95rem; }
    .card-title-analysis { color: #a3e635; font-weight: 700; font-size: 0.95rem; }
    .card-title-writer { color: #fde047; font-weight: 700; font-size: 0.95rem; }
    .card-title-final { color: #f472b6; font-weight: 700; font-size: 0.95rem; }
    
    .card-desc {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 6px;
        line-height: 1.4;
        font-family: 'Inter', sans-serif;
    }
    
    .arch-flow-text {
        text-align: center;
        color: #f97316;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 8px 0;
        letter-spacing: 0.08em;
    }
    
    /* Result Cards */
    .report-box {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 24px;
        color: #e5e7eb;
        line-height: 1.7;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Keys
    default_groq = os.getenv("GROQ_API_KEY", "")
    default_tavily = os.getenv("TAVILY_API_KEY", "")
    default_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_groq,
        type="password",
        help="Required for fast LLM inference via Groq.",
    )
    
    tavily_api_key = st.text_input(
        "Tavily API Key",
        value=default_tavily,
        type="password",
        help="Required for high-signal web search.",
    )
    
    st.divider()
    
    st.markdown("### 🤖 LLM Model Settings")
    model_options = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "qwen-qwq-32b",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "Custom...",
    ]
    
    selected_model = st.selectbox(
        "Groq Model",
        options=model_options,
        index=0 if default_model not in model_options else model_options.index(default_model),
        help="openai/gpt-oss-120b is selected by default. If you hit Groq TPM limits, switch to llama-3.3-70b-versatile.",
    )
    
    if selected_model == "Custom...":
        model_name = st.text_input("Custom Model Identifier", value=default_model)
    else:
        model_name = selected_model
        
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    
    st.caption("💡 *Tip: If you encounter 413 TPM limits on Groq free tier, select `llama-3.3-70b-versatile` for high token throughput.*")

    
    st.divider()
    
    st.markdown("### 📚 Prompt Specifications")
    with st.expander("🔍 Search Agent Prompt"):
        st.markdown(f"**Role:** {SEARCH_AGENT_ROLE}")
        st.markdown(f"**Goal:** {SEARCH_AGENT_GOAL}")
        st.markdown(f"**Backstory:** *\"{SEARCH_AGENT_BACKSTORY}\"*")
        st.markdown("**Tools:** `Tavily Web Search`")
        
    with st.expander("🔬 Analysis Agent Prompt"):
        st.markdown(f"**Role:** {ANALYSIS_AGENT_ROLE}")
        st.markdown(f"**Goal:** {ANALYSIS_AGENT_GOAL}")
        st.markdown(f"**Backstory:** *\"{ANALYSIS_AGENT_BACKSTORY}\"*")
        st.markdown("**Tools:** `None — Pure Reasoning`")
        
    with st.expander("✍️ Writer Agent Prompt"):
        st.markdown(f"**Role:** {WRITER_AGENT_ROLE}")
        st.markdown(f"**Goal:** {WRITER_AGENT_GOAL}")
        st.markdown(f"**Backstory:** *\"{WRITER_AGENT_BACKSTORY}\"*")
        st.markdown("**Tools:** `None — Pure Synthesis`")
        
    with st.expander("👔 Manager Agent Prompt"):
        st.markdown(f"**Role:** {MANAGER_AGENT_ROLE}")
        st.markdown(f"**Goal:** {MANAGER_AGENT_GOAL}")
        st.markdown(f"**Backstory:** *\"{MANAGER_AGENT_BACKSTORY}\"*")


# ----------------- MAIN UI -----------------

st.markdown('<div class="crew-header">THE RESEARCH ASSISTANT CREW</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="crew-subtext">A manager agent delegates a research question to three specialist agents, then synthesizes their findings into one report — no human stitching required.</div>',
    unsafe_allow_html=True,
)


# Quick preset suggestions
if "topic_query" not in st.session_state:
    st.session_state.topic_query = ""

def set_topic(text: str):
    st.session_state.topic_query = text

st.markdown("##### ⚡ Quick Research Topics")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)

col_p1.button("⚛️ Quantum Computing 2026", on_click=set_topic, args=("Current state of quantum computing hardware benchmarks and error correction breakthroughs in 2026",), use_container_width=True)
col_p2.button("🧠 DeepSeek-R1 Architecture", on_click=set_topic, args=("DeepSeek-R1 reasoning architecture, training methodology, and reinforcement learning innovations",), use_container_width=True)
col_p3.button("🔋 Solid-State Batteries", on_click=set_topic, args=("Solid-state battery commercialization roadmap, energy density gains, and automotive adoption timeline",), use_container_width=True)
col_p4.button("⚡ Small Language Models", on_click=set_topic, args=("Advancements in Small Language Models (SLMs) running on edge devices vs frontier LLMs",), use_container_width=True)

# Topic input
user_topic = st.text_area(
    "Enter your research question or topic:",
    key="topic_query",
    placeholder="e.g. Compare post-training reinforcement learning techniques across recent frontier reasoning models...",
    height=90,
)



# Run Crew Button
col_btn, col_info = st.columns([1, 4])
with col_btn:
    start_research = st.button("🚀 Launch Research Crew", type="primary", use_container_width=True)

# State placeholders for session
if "crew_results" not in st.session_state:
    st.session_state.crew_results = None

if start_research:
    if not groq_api_key:
        st.error("⚠️ Please provide a valid **Groq API Key** in the sidebar or `.env` file.")
    elif not tavily_api_key:
        st.error("⚠️ Please provide a valid **Tavily API Key** in the sidebar or `.env` file.")
    elif not user_topic.strip():
        st.warning("⚠️ Please enter a research question or topic to investigate.")
    else:
        # Containers for live streaming status
        progress_box = st.container()
        
        with progress_box:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            manager_plan_exp = st.expander("📋 1. Manager Delegation Plan", expanded=True)
            search_exp = st.expander("🔍 2. Search Agent (Librarian Sources)", expanded=True)
            analysis_exp = st.expander("🔬 3. Analysis Agent (Key Claims & Evidence)", expanded=True)
            writer_exp = st.expander("✍️ 4. Writer Agent (Draft Report)", expanded=True)
            final_exp = st.expander("🏆 5. Final Verified Report", expanded=True)


            state_store = {
                "plan": None,
                "sources": None,
                "claims_analysis": None,
                "draft_report": None,
                "final_report": None,
            }

            status_text.info("👔 **Manager Agent** is analyzing research objectives and preparing sub-tasks...")
            progress_bar.progress(10)

            stream_gen = run_research_stream(
                topic=user_topic.strip(),
                groq_api_key=groq_api_key,
                tavily_api_key=tavily_api_key,
                model_name=model_name,
                temperature=temperature,
            )

            for step_output in stream_gen:
                if "error" in step_output:
                    st.error(f"❌ Error during workflow execution: {step_output['error']}")
                    break
                
                # Manager Plan Node
                if "manager_plan" in step_output:
                    node_data = step_output["manager_plan"]
                    state_store["plan"] = node_data.get("plan")
                    with manager_plan_exp:
                        st.markdown(state_store["plan"])
                    status_text.info("🔍 **Search Agent (Senior Research Librarian)** is searching Tavily for primary sources...")
                    progress_bar.progress(30)
                
                # Search Agent Node
                elif "search_agent" in step_output:
                    node_data = step_output["search_agent"]
                    state_store["sources"] = node_data.get("sources")
                    with search_exp:
                        st.markdown(state_store["sources"])
                    status_text.info("🔬 **Analysis Agent (Critical Analyst)** is extracting empirical claims and validating evidence...")
                    progress_bar.progress(55)
                
                # Analysis Agent Node
                elif "analysis_agent" in step_output:
                    node_data = step_output["analysis_agent"]
                    state_store["claims_analysis"] = node_data.get("claims_analysis")
                    with analysis_exp:
                        st.markdown(state_store["claims_analysis"])
                    status_text.info("✍️ **Writer Agent (Technical Writer)** is drafting the structured report...")
                    progress_bar.progress(75)
                
                # Writer Agent Node
                elif "writer_agent" in step_output:
                    node_data = step_output["writer_agent"]
                    state_store["draft_report"] = node_data.get("draft_report")
                    with writer_exp:
                        st.markdown(state_store["draft_report"])
                    status_text.info("👔 **Manager Agent** is conducting final QA review and polish...")
                    progress_bar.progress(90)
                
                # Manager Review Node
                elif "manager_review" in step_output:
                    node_data = step_output["manager_review"]
                    state_store["final_report"] = node_data.get("final_report")
                    with final_exp:
                        st.markdown(state_store["final_report"])
                    progress_bar.progress(100)
                    status_text.success("✅ **Research Crew finished execution! Final report ready.**")

            st.session_state.crew_results = state_store

# ----------------- DISPLAY SAVED RESULTS -----------------
if st.session_state.crew_results and st.session_state.crew_results.get("final_report"):
    results = st.session_state.crew_results
    st.divider()
    
    st.markdown("## 📄 Final Research Intelligence Briefing")
    
    tab_report, tab_sources, tab_claims, tab_plan = st.tabs([
        "🏆 Final Report",
        "🔍 Sources & Citations",
        "🔬 Claims & Evidence",
        "📋 Manager Plan",
    ])
    
    with tab_report:
        st.markdown(results["final_report"])
        st.download_button(
            label="📥 Download Report (.md)",
            data=results["final_report"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=False,
        )
        
    with tab_sources:
        if results.get("sources"):
            st.markdown(results["sources"])
            
    with tab_claims:
        if results.get("claims_analysis"):
            st.markdown(results["claims_analysis"])
            
    with tab_plan:
        if results.get("plan"):
            st.markdown(results["plan"])
