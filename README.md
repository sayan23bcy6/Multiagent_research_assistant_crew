# The Research Assistant Crew

> A manager agent delegates a research question to three specialist agents, then synthesizes their findings into one report — no human stitching required.

Built with **Streamlit**, **LangGraph**, **LangChain**, **Groq** (`openai/gpt-oss-120b`), and **Tavily Web Search**.

---

## 🏛️ Architecture & Agent Team

```
                        ┌────────────────────────┐
                        │     MANAGER AGENT      │
                        │  (Breaks into subtasks)│
                        └───────────┬────────────┘
                                    │
                                    │ delegates
                                    ▼
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│      SEARCH AGENT       │  │     ANALYSIS AGENT      │  │      WRITER AGENT       │
│ Senior Research Librar. │  │ Critical Research Anal. │  │ Technical Report Writer │
│  (Tavily Web Search)    │  │  (Claims & Evidence)    │  │  (Structured Synthesis) │
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    │
                                    │ reports back
                                    ▼
                        ┌────────────────────────┐
                        │      FINAL REPORT      │
                        │ (Manager review & QA)  │
                        └────────────────────────┘
```

### Specialist Agent Personas & Goals

1. **Manager Agent**:
   - **Role**: Research Manager & Orchestrator
   - **Goal**: Breaks down research inquiry into targeted sub-tasks and directs the search librarian, then performs final QA on the writer's report.
2. **Search Agent**:
   - **Role**: Senior Research Librarian
   - **Goal**: Find 5–8 credible, recent sources on the given topic and return title, URL, and a 1-line relevance note for each.
   - **Backstory**: *"You've spent 15 years finding primary sources for investigative journalists. You distrust SEO content farms and always prefer official docs, papers, or first-party blogs."*
   - **Tools**: Tavily Web Search.
3. **Analysis Agent**:
   - **Role**: Critical Research Analyst
   - **Goal**: Extract the 3–5 strongest claims from the Search Agent's sources, each with supporting evidence and source citations.
   - **Backstory**: *"You've reviewed thousands of papers for a research lab. You flag unsupported claims instead of repeating them, and you never merge two sources' claims into one without saying so."*
   - **Tools**: None — pure reasoning.
4. **Writer Agent**:
   - **Role**: Technical Report Writer
   - **Goal**: Turn the Analyst's claims into a structured report: a 2-sentence summary, 3–5 headed sections, and a sources list.
   - **Backstory**: *"You write for busy engineers. No filler intros, no restating the question — you open with the answer and back it with the evidence you were given."*
   - **Tools**: None — pure synthesis.

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create or edit your `.env` file (see `.env.example`):

```ini
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

*(You can also input your API keys directly in the Streamlit web interface sidebar!)*

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 🌟 Key Features

- **Multi-Agent Orchestration**: Stateful graph execution using `langgraph.graph.StateGraph`.
- **High-Speed Inference**: Powered by Groq LLMs (default `openai/gpt-oss-120b` with fallbacks like `llama-3.3-70b-versatile`, `deepseek-r1-distill-llama-70b`, `qwen-qwq-32b`).
- **Precision Primary Search**: Real-time web searching via Tavily API.
- **Cyberpunk Dark Grid UI**: Matching the custom multi-agent architecture diagram styling.
- **Live Execution Feedback**: Real-time progress indicators for every agent step.
- **Export Capabilities**: 1-click Markdown download of verified research briefings.
