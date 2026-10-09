.PHONY: install run test lint docker evals

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check agents/ app.py tests/ evals/
	mypy agents/ --ignore-missing-imports --no-strict-optional

docker:
	docker build -t research-crew:latest .
	docker run --rm -p 8501:8501 \
		-e GROQ_API_KEY=$(GROQ_API_KEY) \
		-e TAVILY_API_KEY=$(TAVILY_API_KEY) \
		research-crew:latest

evals:
	@echo "Running evaluation harness (requires live API keys)..."
	python evals/run_evals.py
