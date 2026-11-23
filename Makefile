.PHONY: help install up down logs db reset-db run train check test test-integration lint

PY ?= python
-include .env
export

help:
	@echo "make install           install Python dependencies (local dev)"
	@echo "make up                build and start postgres + app with Docker Compose"
	@echo "make down              stop containers"
	@echo "make db                start only PostgreSQL (for local development)"
	@echo "make reset-db          wipe and re-seed the database (moves sample dates to today)"
	@echo "make run               run Streamlit locally"
	@echo "make check | train     connection check / (re)train the knowledge base"
	@echo "make test              unit tests (no database or API key needed)"
	@echo "make test-integration  all tests against the compose database"

install:
	$(PY) -m pip install -r requirements-dev.txt

up:
	docker compose up -d --build
	@echo "Open http://localhost:8501"

down:
	docker compose down

logs:
	docker compose logs -f app

db:
	docker compose up -d postgres

reset-db:
	docker compose down -v
	docker compose up -d postgres

run:
	streamlit run streamlit_app.py

check:
	$(PY) -m app.cli check

train:
	$(PY) -m app.cli train --reset

test:
	$(PY) -m pytest -m "not integration"

test-integration:
	TEST_DATABASE_URL="postgresql+psycopg://$(READONLY_USER):$(READONLY_PASSWORD)@localhost:$(POSTGRES_PORT)/$(POSTGRES_DB)" \
		$(PY) -m pytest

lint:
	ruff check app tests streamlit_app.py
