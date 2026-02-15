.PHONY: lint format fix setup typecheck

setup:
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .

typecheck:
	uv run mypy src
