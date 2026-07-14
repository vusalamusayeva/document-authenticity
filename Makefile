.PHONY: install test lint format run docker-build docker-run

install:
	pip install -e ".[dev]"

test:
	pytest -m "not evaluation"

test-eval:
	pytest -m evaluation

lint:
	ruff check src api tests
	mypy src api

format:
	ruff format src api tests

run:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t document-authenticity .

docker-run:
	docker run -p 8000:8000 --env-file .env document-authenticity
