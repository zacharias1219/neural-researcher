# NeuralResearcher

NeuralResearcher is a terminal-based, multi-agent research planner for ML/CS topics. Given a research topic, neuralresearcher uses a multi-agent pipeline to ground in the literature, identify concrete gaps, and generate a structured, executable research plan and SOP for a publishable paper.

## Features

- **Topic & Scope Agent**: Refines raw ideas into structured scope specifications.
- **Retrieval Agent**: Multi-source discovery across arXiv and Semantic Scholar.
- **Reading Agent**: Extracts metadata and semantic claims from full text / abstracts.
- **Coverage Agent**: Dynamically groups papers into sub-domains to highlight coverage gaps.
- **Gaps & Directions Agents**: Identifies novel opportunities and constructs hypotheses.
- **Planner Agent**: Generates dependency-validated experimental steps.
- **Reviewer Agent**: Checks plan for logical cycles and verifies claims.

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
neuralresearcher "Mamba architectures for edge devices" --provider groq
```

### Outputs

The pipeline will emit a final `research_plan.md` in the current directory, along with a `research/` directory containing the robust, versioned JSON state of all artifacts.

## Evaluation Harness

NeuralResearcher includes a powerful evaluation harness to test its behavior across topics and providers.

```bash
neuralresearcher-eval --suite core --providers openai,anthropic
```

## Architecture

NeuralResearcher is a state-machine driven multi-agent orchestrator. Agents act on a strictly-typed shared state (via Pydantic) to avoid the typical failure modes of raw prompt chains. State changes are backed up safely with built-in migration hooks.