# NeuralResearcher

NeuralResearcher is a terminal-based, multi-agent research planner for ML/CS topics. Given a research topic, neuralresearcher uses a multi-agent pipeline to ground in the literature, identify concrete gaps, and generate a structured, executable research plan and SOP for a publishable paper.

## Features

- **Topic & Scope Agent**: Refines raw ideas into structured scope specifications.
- **Retrieval Agent**: Searches arXiv and uses Semantic Scholar as an optional enrichment/fallback source.
- **Reading Agent**: Extracts metadata and semantic claims from abstracts (abstract-only for v0.1).
- **Coverage Agent**: Dynamically groups papers into sub-domains to highlight coverage gaps.
- **Gaps & Directions Agents**: Identifies novel opportunities and constructs hypotheses.
- **Planner Agent**: Generates dependency-validated experimental steps.
- **Reviewer Agent**: Checks plan for logical cycles and performs claim consistency review.

## Installation

```bash
pip install -e .
```
To include the MCP server support, install with the `mcp` optional dependency:
```bash
pip install -e ".[mcp]"
```

## Usage

Set your API keys depending on the provider you want to use:
```bash
export GROQ_API_KEY=your_key
# OR
export OPENAI_API_KEY=your_key
# OR 
export ANTHROPIC_API_KEY=your_key
```

Run the researcher:
```bash
neuralresearcher run "Mamba architectures for edge devices" --provider groq
```

### Outputs

The pipeline will emit artifacts and robust, versioned JSON state in the `research/runs/{run_id}/` directory. The final research plan is located at `research/runs/{run_id}/research_plan.md`.

## Evaluation Harness

NeuralResearcher includes a powerful evaluation harness to test its behavior across topics and providers.

```bash
neuralresearcher-evals --suite core --providers openai,anthropic
```

## MCP Server

NeuralResearcher includes a robust Model Context Protocol (MCP) server for integration with MCP clients (e.g., Claude Desktop).

### Available Features
- **Tools**: `start_run` (initiate research), `resume_run` (restarts a failed/interrupted run from `INIT` and clears incompatible artifacts), `cancel_run`, `get_run_status`, `list_runs`, `get_run_result`.
- **Resources**:
  - `research://runs/{run_id}/status`: Live JSON status of a run.
  - `research://runs/{run_id}/plan`: The final generated research plan (Markdown).
  - `research://runs/{run_id}/result`: The complete terminal result metadata.
  - `research://runs/{run_id}/artifacts/{artifact_name}`: Additional artifacts generated during a run.
- **Prompts**: `explore_topic` (guided prompt to kick off a research task).

### Configuration & Limits
- **Environment Variables**:
  - `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY` (provider selection).
  - `NEURALRESEARCHER_DATA_DIR`: Base directory for run state (default: `./research`).
  - `NEURALRESEARCHER_MCP_AUTH_TOKEN`: Bearer token for HTTP authentication.
  - `NEURALRESEARCHER_MCP_CONCURRENCY`: Max concurrent background runs (default: 2).
  - `NEURALRESEARCHER_MCP_MAX_RESULT_BYTES`: Max bytes per artifact read (default: 10MB).
- **Transports**:
  - **stdio**: Default mode. Best for local GUI clients.
  - **streamable-http**: Standalone HTTP server (`neuralresearcher-mcp --transport streamable-http`). Requires `NEURALRESEARCHER_MCP_AUTH_TOKEN` when bound to external interfaces.
- **Cancellation**: Cancellation is cooperative. Background tasks gracefully halt between pipeline stages.

### Example MCP Host Configuration (Claude Desktop)
```json
{
  "mcpServers": {
    "neuralresearcher": {
      "command": "neuralresearcher-mcp",
      "env": {
        "GROQ_API_KEY": "gsk_...",
        "NEURALRESEARCHER_DATA_DIR": "/path/to/my/workspace"
      }
    }
  }
}
```

## Architecture

NeuralResearcher is a state-machine driven multi-agent orchestrator. Agents act on a strictly-typed shared state (via Pydantic) to avoid the typical failure modes of raw prompt chains. State changes are backed up safely with built-in migration hooks.