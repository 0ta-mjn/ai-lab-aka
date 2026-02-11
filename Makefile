.PHONY: lint format fix setup

setup:
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .
