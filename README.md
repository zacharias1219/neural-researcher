# NeuralResearcher

**Terminal-based, PyPI-distributed multi-agent research planner for ML/CS topics.**

Given a research topic, NeuralResearcher uses a deterministic multi-agent pipeline to ground in the literature (via arXiv + Semantic Scholar), identify concrete gaps, and generate a structured, executable research plan.

## What it is

- A **research planning assistant** that produces structured plans and SOPs from a topic string.
- A **CLI tool** for interactive or scripted research planning.
- An **MCP server** for integration with Claude Desktop, Cursor, and other MCP-capable hosts.
- A **PyPI-distributable package** with optional extras.

## What it is NOT

- Not a full-text paper reader (v0.1 uses abstract-only evidence; `content_level` is tracked per paper).
- Not a checkpoint-resume system (resume creates a new run with lineage from the prior run's config).
- Not a multi-source canonical research verifier (evidence is extracted from abstracts, not cross-validated against full text).
- Not an agent framework or LangChain wrapper.

## Installation

### From PyPI

```bash
pip install neuralresearcher
```

### With MCP support

```bash
pip install "neuralresearcher[mcp]"
```

### With all extras (MCP + dev tools)

```bash
pip install "neuralresearcher[all]"
```

### Development install

```bash
git clone https://github.com/zacharias1219/neural-researcher.git
cd neural-researcher
pip install -e ".[all]"
```

## Provider Configuration

NeuralResearcher supports four LLM providers. Set the appropriate API key:

| Provider | Env Variable | Default Model |
|----------|-------------|---------------|
| **Groq** (default) | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |

```bash
export GROQ_API_KEY=your_key_here
```

## CLI Commands

```bash
# Show help
neuralresearcher --help

# Show version
neuralresearcher version

# List providers and API key status
neuralresearcher providers

# Run a research plan
neuralresearcher run "Mamba architectures for edge inference"
neuralresearcher run "Mamba architectures for edge inference" --provider groq
neuralresearcher run "graph neural networks" -p openai -m gpt-4o --strict
neuralresearcher run "topic" --no-interactive  # non-interactive mode

# Resume a failed/interrupted run (creates new run with lineage)
neuralresearcher run --resume <run_id>

# List all runs
neuralresearcher runs
neuralresearcher runs --status HALTED --provider groq

# Check run status
neuralresearcher status <run_id>

# List artifacts for a run
neuralresearcher artifacts <run_id>

# Start the MCP server
neuralresearcher mcp --transport stdio
neuralresearcher mcp --transport streamable-http --host 127.0.0.1 --port 8000 --path /mcp
```

## MCP Server

NeuralResearcher exposes a full MCP (Model Context Protocol) server for integration with MCP clients.

### Launch

```bash
# stdio (default, for local GUI clients like Claude Desktop)
neuralresearcher mcp --transport stdio
# or via the standalone entry point:
neuralresearcher-mcp --transport stdio

# Streamable HTTP (for remote/network access)
neuralresearcher mcp --transport streamable-http --host 127.0.0.1 --port 8000 --path /mcp
```

### Tools

| Tool | Description |
|------|-------------|
| `research_start` | Start a new research run asynchronously |
| `research_resume` | Resume (restart from INIT) a failed run |
| `research_status` | Get current or terminal status |
| `research_cancel` | Request cooperative cancellation |
| `research_result` | Get structured result of a completed run |
| `research_list_runs` | List recent runs with optional filters |
| `research_list_artifacts` | List artifacts for a run |

### Resources

| URI | Type | Description |
|-----|------|-------------|
| `research://runs/{run_id}/manifest` | JSON | Sanitized public manifest |
| `research://runs/{run_id}/status` | JSON | Current status |
| `research://runs/{run_id}/plan` | Markdown | Generated research plan |
| `research://runs/{run_id}/result` | JSON | Terminal result |
| `research://runs/{run_id}/papers` | JSON | Retrieved papers |
| `research://runs/{run_id}/claims` | JSON | Extracted claims |
| `research://runs/{run_id}/gaps` | JSON | Identified gaps |
| `research://runs/{run_id}/directions` | JSON | Proposed directions |
| `research://runs/{run_id}/review` | JSON | Plan review result |
| `research://runs/{run_id}/coverage` | JSON | Coverage report |
| `research://runs/{run_id}/artifacts/{name}` | varies | Generated artifacts |

### Prompts

| Prompt | Description |
|--------|-------------|
| `create_research_plan` | Guide the host through a full research planning workflow |
| `review_research_plan` | Guide the host to review an existing plan |
| `explore_research_gap` | Guide the host to explore a specific gap |

### Authentication

For HTTP transport, set bearer authentication:

```bash
export NEURALRESEARCHER_MCP_AUTH_TOKEN=your_secret_token
```

Authentication is **required** when binding to a non-loopback interface. The middleware returns `401 Unauthorized` for missing or invalid tokens.

### MCP Host Configuration (Claude Desktop)

```json
{
  "mcpServers": {
    "neuralresearcher": {
      "command": "neuralresearcher-mcp",
      "env": {
        "GROQ_API_KEY": "gsk_...",
        "NEURALRESEARCHER_DATA_DIR": "/path/to/workspace"
      }
    }
  }
}
```

## Data Directory & Run Layout

All state is stored under the configured data directory (default: `./research`):

```
research/
  runs/
    <run_id>/
      manifest.json       # Run metadata (provider, topic, status, etc.)
      state.json          # Pydantic-typed pipeline state
      research_plan.md    # Generated research plan
      papers/             # (reserved)
      analysis/           # (reserved)
      transcripts/        # LLM call transcripts (not publicly exposed)
      sources/            # (reserved)
      logs/               # Internal logs (not publicly exposed)
      backups/            # State backups (not publicly exposed)
```

Internal files (`transcripts/`, `logs/`, `backups/`, raw `state.json`) are **not exposed** through MCP resources. Only allowlisted public files are accessible. The manifest is sanitized before exposure (internal paths are stripped).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `NEURALRESEARCHER_DATA_DIR` | `research` | Base directory for run state |
| `NEURALRESEARCHER_MCP_TRANSPORT` | `stdio` | MCP transport mode |
| `NEURALRESEARCHER_MCP_HOST` | `127.0.0.1` | HTTP bind host |
| `NEURALRESEARCHER_MCP_PORT` | `8000` | HTTP bind port |
| `NEURALRESEARCHER_MCP_PATH` | `/mcp` | HTTP mount path |
| `NEURALRESEARCHER_MCP_AUTH_TOKEN` | — | Bearer token for HTTP auth |
| `NEURALRESEARCHER_MAX_CONCURRENT_RUNS` | `2` | Max concurrent background runs |
| `NEURALRESEARCHER_MCP_LOG_LEVEL` | `INFO` | MCP server log level |
| `NEURALRESEARCHER_MCP_MAX_RESULT_BYTES` | `10485760` | Max bytes per artifact read |

## Resume Semantics

The `--resume <run_id>` flag does **not** perform true checkpoint resume. It:

1. Reads the prior run's topic, provider, model, and config.
2. Creates a **new run** with a fresh run ID.
3. Sets `resumed_from` in the new manifest pointing to the original run ID.
4. Restarts the pipeline from `INIT`.

The original run is preserved and unmodified.

## Cancellation Semantics

Cancellation is **cooperative**. When cancelled:

1. The `cancellation_requested` flag is set in the manifest.
2. The cancellation token is signalled.
3. The pipeline checks the token between stages and raises `CancelledError`.
4. The run transitions to `HALTED` with halt code `CANCELLED`.

During server shutdown:
1. All active tokens are signalled.
2. Tasks get a grace period to finish cooperatively.
3. Any still-active runs are marked `INTERRUPTED`.

## Evaluation Harness

```bash
# Run the evaluation suite with the default provider (Groq)
neuralresearcher-evals run-suite --suite-name core --providers groq

# Or evaluate across multiple providers
neuralresearcher-evals run-suite --suite-name core --providers groq,openai,anthropic
```

The eval harness runs predefined tasks across providers and grades on completion, correctness, and efficiency. Results are saved to `research/evals/eval_results.json`.

## Security & Privacy Notes

- **No arbitrary filesystem access**: MCP resources are restricted to allowlisted files within run directories.
- **Manifest sanitization**: Public manifests strip internal paths; only filenames are exposed.
- **Path traversal protection**: All artifact reads verify containment within the run directory.
- **Bearer auth**: Required for non-loopback HTTP transport.
- **No secret leaking**: CLI error messages sanitize API keys from output.
- **stdio safety**: In stdio mode, the Rich console is redirected to stderr to avoid contaminating the JSON-RPC protocol stream on stdout.

## Limitations

- Evidence is **abstract-only** (no full-text PDF ingestion in v0.1).
- Claims are extracted from abstracts, not cross-validated against full text.
- Resume is restart-from-INIT, not true checkpoint resume.
- Evaluation harness is functional but not exhaustive.
- Concurrent run limit is enforced via semaphore (default: 2).

## Development

```bash
# Install with dev extras
pip install -e ".[all]"

# Run tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ -v --cov=neuralresearcher --cov-report=term-missing

# Lint
python -m ruff check neuralresearcher/

# Type check
python -m mypy neuralresearcher/

# Build
python -m build

# Clean install verification
pip install dist/neuralresearcher-0.1.0-py3-none-any.whl --force-reinstall
neuralresearcher --help
neuralresearcher version
```

## Release Verification

```bash
# Build wheel and sdist
python -m build

# Install wheel in clean environment
python -m venv venv-verify
venv-verify/Scripts/activate  # Windows
# or: source venv-verify/bin/activate  # Unix

# Install local wheel
pip install dist/neuralresearcher-0.1.0-py3-none-any.whl
# Then optionally install MCP dependencies
pip install "neuralresearcher[mcp]"
# (Alternatively, test extras from source: pip install ".[mcp]")

# Verify CLI
neuralresearcher --help
neuralresearcher version
neuralresearcher providers

# Verify MCP transport
neuralresearcher-mcp --help
# See RELEASE.md for the full release checklist, including stdio and HTTP smoke tests.
# Publish to TestPyPI
python -m twine upload --repository testpypi dist/*

# Publish to PyPI
python -m twine upload dist/*
```

## Architecture

NeuralResearcher is a state-machine driven multi-agent orchestrator. Agents act on strictly-typed shared state (via Pydantic) to avoid the typical failure modes of raw prompt chains. State changes are backed up with atomic writes and built-in migration hooks.

### Pipeline Stages

```
INIT → SCOPED → RETRIEVED → READ → MAPPED → GAPS_IDENTIFIED
     → DIRECTIONS_PROPOSED → PLAN_DRAFTED → PLAN_REVIEWED → REPORT_READY
```

At any point, the pipeline may transition to `HALTED` with a typed halt code indicating the reason.

### Halt Codes

| Code | Meaning |
|------|---------|
| `OUT_OF_SCOPE` | Topic is outside supported domains |
| `INSUFFICIENT_EVIDENCE` | Too few papers retrieved |
| `LOW_RETRIEVAL_RELEVANCE` | Retrieved papers don't match topic |
| `COVERAGE_FAILURE` | Literature coverage is insufficient |
| `INVALID_MODEL_OUTPUT` | LLM returned unparseable output |
| `PROVIDER_ERROR` | LLM provider returned an error |
| `TOOL_ERROR` | External tool (arXiv, Semantic Scholar) failed |
| `STORAGE_FAILURE` | State file I/O error |
| `PLAN_VALIDATION_FAILURE` | Generated plan has no steps |
| `REVIEW_FAILURE` | Plan failed review in strict mode |
| `CANCELLED` | Run was cancelled by user/host |
| `INTERRUPTED` | Run was interrupted by server shutdown |
| `INTERNAL_ERROR` | Unexpected internal error |

## License

MIT