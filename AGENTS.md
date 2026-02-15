# AGENTS.md

Guidance for coding agents working in this repository.

## Scope
- Keep changes minimal, focused, and consistent with existing patterns.
- Prefer root-cause fixes over surface-level patches.
- Do not add unrelated refactors, features, or dependency changes.

## Project Snapshot
- Language: Python (requires 3.12+).
- Package/build tool: `uv` (project is configured in `pyproject.toml` and `uv.lock`).
- Main source tree: `src/`.
- CLI entrypoint: `cli = "src.cli:main"`.

## Environment Setup
1. Use Python 3.12 (`.python-version` is `3.12`).
2. Install/sync dependencies with `uv`.
3. Configure local environment variables via `.env` (see `.env.sample`).

## Required Commands (Before Handoff)
- `make lint` — Ruff lint check.
- `make format` — Ruff format.
- `make fix` — Ruff auto-fix.
- `make typecheck` — Mypy check for `src`.

If pre-commit is needed:
- `make setup` — installs pre-commit hooks via `uv run pre-commit install`.

## Coding Conventions
- Use type hints for public and internal functions.
- Use Pydantic models for structured I/O and schema definitions.
- Keep imports organized (Ruff rules apply).
- Use module-level logger pattern:
  - `logger = logging.getLogger(__name__)`
- Handle external-call failures explicitly and log actionable context.

## Observability and Tracing
- Follow existing Langfuse patterns already used in the codebase:
  - `@observe(...)` decorator for span/generation boundaries.
  - `with_langfuse_span(...)` context manager for nested workflow spans.
- When adding a new LLM/tool step, ensure inputs/outputs/usage are captured consistently.

## External Dependencies and Secrets
Use `.env.sample` as the source of truth for required keys:
- `JINA_AI_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_BASE_URL`

Rules:
- Never hardcode secrets.
- Never commit real secrets.
- If new env vars are introduced, update `.env.sample` with placeholder values.

## Safe Editing Rules
- Do not edit generated or environment-managed artifacts unless explicitly asked:
  - `.venv/`, `activate/`, `*.egg-info/`, caches.
- Avoid changing lock/config files unless required by the requested task.
- Preserve public interfaces unless the task explicitly requires API changes.

## Network/LLM Call Caution
- This project includes real network and model calls (Jina Reader, LiteLLM, Langfuse).
- Prefer validating logic with static checks first (`lint`, `typecheck`) before any expensive or secret-dependent runtime calls.

## File Organization Guidance
- Domain workflow code lives under `src/company_detail/`.
- Infra adapters/utilities live under `src/infra/`.
- New modules should follow this separation (domain vs infra) and existing naming style.