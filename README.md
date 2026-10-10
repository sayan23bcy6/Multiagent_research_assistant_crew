# The Research Assistant Crew

> A manager agent delegates a research question to three specialist agents, then synthesizes their findings into one reviewed report — no human stitching required.

Built with **Streamlit**, **LangGraph**, **LangChain-Groq**, **Groq** (`openai/gpt-oss-120b`), and **Tavily Web Search**.

---

## 🏛️ Architecture

```
manager_plan → search_agent → analysis_agent ─┬→ writer_agent → manager_review ─┬→ END
                    ↑                          │        ↑                         │
                    └── (search retry ≤2) ─────┘        └── (writer retry ≤1) ───┘
```

The **Manager** produces a structured JSON plan (Pydantic-validated). The **Search Agent** uses
those queries directly — no redundant LLM call. If fewer than 3 sources are cited after analysis,
the pipeline retries search (up to 2 times). After the Manager reviews the draft, a reject verdict
routes the **Writer** back for one revision.

The architecture diagram is also rendered interactively in the **🏛️ Architecture** tab of the app.

### Specialist Agent Personas & Goals

| Agent | Role | Tools |
|-------|------|-------|
| **Manager** | Research Manager — plans sub-tasks, reviews final report | None |
| **Search** | Senior Research Librarian — finds credible sources | Tavily Web Search (with raw content) |
| **Analyst** | Critical Research Analyst — extracts 3–5 empirical claims | None — pure reasoning |
| **Writer** | Technical Report Writer — structures the report | None — pure synthesis |

---

## 🔒 Source Integrity

- Tavily results are kept as **structured data** (`TavilySource`) through the entire pipeline.
- The Search Agent refers to sources by **index** (`Source #1`, `Source #2`, …) — the LLM never
  writes raw URLs.
- URLs in the final report are **validated** against the Tavily result set. Any unmatched URL
  triggers a visible warning in the UI.
- If LLM curation fails, an explicit **[UNCURATED]** notice is shown — there is no silent fallback.

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or with make:
make install
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```ini
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

*(API keys can also be entered in the Streamlit sidebar.)*

### 3. Run the App

```bash
streamlit run app.py
# or:
make run
```

### 4. Docker

```bash
make docker
# or:
docker build -t research-crew:latest .
docker run -p 8501:8501 \
  -e GROQ_API_KEY=... \
  -e TAVILY_API_KEY=... \
  research-crew:latest
```

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `TAVILY_API_KEY` | *(required)* | Tavily API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model identifier |
| `MAX_SEARCH_RETRIES` | `2` | Max search-agent retries if <3 sources cited |
| `MAX_WRITER_RETRIES` | `1` | Max writer retries on manager reject |
| `MAX_TOTAL_LLM_CALLS` | `20` | Hard cap on total LLM calls per run |
| `SOURCE_CONTENT_CHAR_BUDGET` | `6000` | Max chars fetched per source (~1,500 tokens) |

---

## 🧪 Tests & CI

```bash
make test    # pytest (mocks Groq and Tavily — no API keys needed)
make lint    # ruff + mypy
```

CI runs on every push and pull request via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## 📊 Evaluation Harness

```bash
make evals   # requires live API keys — NOT run in CI
```

Results are written to [`evals/results.md`](evals/results.md).
Scores: URL integrity, report structure, and key-point coverage (keyword heuristic).

> **Latest eval scores**: Not yet run. Execute `make evals` locally after configuring API keys.

---

## 🌟 Key Features

- **Structured Manager Plan**: JSON-validated `core_objective`, `sub_tasks`, and `search_queries`.
- **Real Source Content**: Tavily `include_raw_content` with per-source 1,500-token budget.
- **Source Integrity**: Index-based references, URL validation, no silent fallbacks.
- **Revision Loops**: Search retries (≤2) and writer retries (≤1) based on quality signals.
- **Per-Node Token Budgets**: Writer/Review get 6,000 tokens; Plan node gets 512.
- **Agent Timeline**: Execution timeline with duration and token counts in the UI.
- **Live Execution Feedback**: Real-time progress for every agent step.
- **Export**: 1-click Markdown download of reviewed research briefings.

---

## 📁 Project Structure

```
.
├── agents/
│   ├── __init__.py      # lightweight — only state type exports
│   ├── crew.py          # LangGraph graph, all node functions
│   ├── prompts.py       # agent system prompts
│   └── state.py         # ResearchState TypedDict + AgentLog, ManagerPlan, TavilySource
├── tests/               # pytest unit tests (no API keys required)
├── evals/               # evaluation harness (requires API keys)
│   ├── questions.yaml
│   ├── run_evals.py
│   └── results.md
├── .github/workflows/ci.yml
├── app.py               # Streamlit UI
├── Dockerfile
├── Makefile
├── CONTRIBUTING.md
├── pyproject.toml
└── requirements.txt
```

---

## 🔮 Future Work

- Model fallback: try next model in a list on 429 or model error.
- Streaming token-by-token output per agent node.
- Persistent run history and comparison across topics.
- Semantic key-point scoring in the evaluation harness (instead of keyword match).
