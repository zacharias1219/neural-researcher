# Release Checklist

Before tagging and pushing a new release (e.g., `v0.1.1`), verify the following checklist. This documents the exact passing outputs expected for a clean release.

## 1. Build and Verify Package
```bash
python -m build
twine check dist/*
```
**Expected Output:**
```
Successfully built neuralresearcher-0.1.x.tar.gz and neuralresearcher-0.1.x-py3-none-any.whl
Checking dist\neuralresearcher-0.1.x-py3-none-any.whl: PASSED
Checking dist\neuralresearcher-0.1.x.tar.gz: PASSED
```

## 2. Static Analysis & Tests
```bash
pytest -q
python -m ruff check neuralresearcher tests
python -m mypy neuralresearcher tests
```
**Expected Output:**
```
# Pytest
182 passed in ~24.00s

# Ruff
All checks passed!

# Mypy
Success: no issues found in 62 source files
```

## 3. Wheel Smoke Tests
```bash
python -m venv venv-verify
venv-verify/Scripts/activate
pip install "dist/neuralresearcher-0.1.x-py3-none-any.whl"
```
**Expected Output:**
```
Successfully installed neuralresearcher-0.1.x ...
```

## 4. CLI Smoke Tests
```bash
neuralresearcher --help
neuralresearcher version
neuralresearcher providers
```
**Expected Output:**
```
Usage: neuralresearcher [OPTIONS] COMMAND [ARGS]...
...
neuralresearcher v0.1.x
...
Supported LLM Providers (Groq, OpenAI, Anthropic, DeepSeek)
```

## 5. MCP Transport Smoke Tests

### Stdio Transport & HTTP Transport Auth Check
Run the standalone verification script to test both Stdio and Streamable HTTP auth:
```bash
python scripts/verify_mcp_transport.py
```
**Expected Output:**
```
Verifying Stdio Transport...
  Available tools: ['search_papers', 'fetch_paper', ...]
  Successfully invoked `...`.
  Stdio OK.

Verifying Streamable HTTP Auth...
  [x] Rejected absent token successfully.
  [x] Rejected invalid token successfully.
  [x] Valid token initialized successfully.
  [x] Successfully invoked `...` over HTTP.
  HTTP Auth OK.

All MCP transport smoke tests PASSED.
```
