# AI-Lab-aka

This is a repository to experiment with several ideas for AI agents.

## Current Use Cases

### Company Detail Extraction Workflow

Extracts detailed information about companies. This use case currently implements and compares two approaches:
1. **Agents SDK Workflow**: An approach utilizing an AI agents framework.
2. **Code-Based Workflow**: A deterministic, code-driven workflow.

*More use cases will be added in the future.*

## Setup

1. **Prerequisites**: Python 3.12+ and [`uv`](https://github.com/astral-sh/uv) are required.
2. **Install Dependencies**:
   ```bash
   uv sync
   ```
3. **Environment Variables**:
   Copy `.env.sample` to `.env` and configure your API keys (e.g., Jina AI, OpenAI, Gemini, Langfuse).
   ```bash
   cp .env.sample .env
   ```

## Usage

This project provides a CLI entrypoint for running the workflows. You can execute it via `uv run cli`.

### Available Commands

- **Run Company Detail Workflow for a single company**
  ```bash
  uv run cli company-detail --company_name "Example Corp" --company_url "https://example.com"
  ```

- **Run Company Detail Workflow in batch (from CSV)**
  ```bash
  uv run cli company-detail-csv path/to/input.csv --output_path path/to/output.jsonl --workflow_type workflow
  ```
  *(Note: `workflow_type` can be `agents` or `workflow`)*

- **Fetch a page using Jina AI Reader**
  ```bash
  uv run cli fetch-jina "https://example.com"
  ```