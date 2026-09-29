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

NeuralResearcher includes a Model Context Protocol (MCP) server.

- **stdio**: Default transport mode.
- **HTTP**: Requires authentication `NEURALRESEARCHER_MCP_AUTH_TOKEN`. Can be started with `neuralresearcher-mcp --transport streamable-http`.
- **Cancellation**: Cancellation is cooperative. Provider and HTTP timeouts are finite.
- **Environment**: Use environment variables like `NEURALRESEARCHER_MCP_AUTH_TOKEN`, `NEURALRESEARCHER_DATA_DIR` to configure the MCP server.

## Architecture

NeuralResearcher is a state-machine driven multi-agent orchestrator. Agents act on a strictly-typed shared state (via Pydantic) to avoid the typical failure modes of raw prompt chains. State changes are backed up safely with built-in migration hooks.