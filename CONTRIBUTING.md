# Contributing to The Research Assistant Crew

Thank you for contributing! This guide covers setup, testing, and the PR process.

---

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/sayan23bcy6/Multiagent_research_assistant_crew.git
   cd Multiagent_research_assistant_crew
   ```

2. **Create a virtual environment (Python 3.11+)**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install runtime + dev dependencies**

   ```bash
   make install
   pip install ruff mypy pytest pytest-cov
   # Or install the dev extras from pyproject.toml:
   pip install -e ".[dev]"
   ```

4. **Configure environment variables**

   Copy `.env.example` to `.env` and fill in your API keys:

   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

---

## Running the App

```bash
make run
# or directly:
streamlit run app.py
```

---

## Running Tests

```bash
make test
# or:
pytest tests/ -v
```

Tests in `tests/` are **unit tests** — they mock all Groq and Tavily clients and do **not** require live API keys.

---

## Linting and Type-Checking

```bash
make lint
# or individually:
ruff check agents/ app.py tests/ evals/
mypy agents/ --ignore-missing-imports --no-strict-optional
```

---

## Running Evaluations

Evaluations are **not** run in CI because they require live API keys and cost money.

```bash
make evals
# or:
python evals/run_evals.py
```

Results are written to `evals/results.md`. Update the README with the new scores and the date/model used.

---

## Pull Request Instructions

1. **Branch naming**: `improvements/<short-description>` or `fix/<short-description>`
2. **One logical change per commit** — keep commits small and focused.
3. **Before opening a PR**:
   - All tests pass: `make test`
   - Linting passes: `make lint`
   - The app starts: `streamlit run app.py`
4. **PR description** must include:
   - Summary of what changed and why
   - Files changed
   - Any ambiguous decisions you made and how you resolved them
   - Anything you could not verify (e.g., live API behavior)
5. Do **not** commit directly to `main`. Open a PR and request a review.

---

## Code Style

- **Python 3.11+**
- **Ruff** for linting (see `pyproject.toml` for rules)
- **Mypy** for type checking (loose mode — `no_strict_optional = true`)
- Docstrings on all public functions and classes
- Type hints on all function signatures

---

## Architecture Overview

```
manager_plan → search_agent → analysis_agent ─┬→ writer_agent → manager_review ─┬→ END
                    ↑                          │        ↑                         │
                    └── (search retry ≤2) ─────┘        └── (writer retry ≤1) ───┘
```

See `agents/crew.py` for the full LangGraph graph definition and `agents/state.py` for the `ResearchState` TypedDict.
