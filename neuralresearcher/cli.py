import os
from typing import Optional

import typer
from rich import box
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from neuralresearcher.application.run_manager import RunManager
from neuralresearcher.config import (
    API_KEY_ENV_VARS,
    DEFAULT_MODELS,
    PROVIDER_DISPLAY_NAMES,
    LLMProvider,
)
from neuralresearcher.logging import (
    console,
    log_error,
    log_info,
    log_success,
    print_banner,
)

app = typer.Typer(
    name="neuralresearcher",
    help="Terminal-based multi-agent research planner for ML/CS topics.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helper: interactive provider + key selection
# ---------------------------------------------------------------------------

def _select_provider_interactive() -> LLMProvider:
    """Prompt the user to pick an LLM provider."""
    table = Table(
        title="[bold cyan]Available LLM Providers[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        title_justify="left",
        show_header=True,
        header_style="bold bright_white",
        border_style="cyan",
    )
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Provider", style="bold white")
    table.add_column("Default Model", style="dim white")
    table.add_column("Env Variable", style="dim yellow")

    providers = list(LLMProvider)
    for i, p in enumerate(providers, 1):
        env_var = API_KEY_ENV_VARS[p]
        key_set = "[green]*[/green]" if os.environ.get(env_var) else "[red]-[/red]"
        table.add_row(
            str(i),
            f"{PROVIDER_DISPLAY_NAMES[p]} {key_set}",
            DEFAULT_MODELS[p],
            env_var,
        )

    console.print()
    console.print(table)
    console.print()

    choice = Prompt.ask(
        "[bold cyan]Select provider[/bold cyan]",
        choices=[str(i) for i in range(1, len(providers) + 1)],
        default="1",
    )
    return providers[int(choice) - 1]


def _ensure_api_key(provider: LLMProvider) -> None:
    """Check for the key in env; if missing, prompt the user to paste it."""
    env_var = API_KEY_ENV_VARS[provider]
    if os.environ.get(env_var):
        log_info(f"{env_var} detected in environment.")
        return

    console.print(
        f"\n  [warning]! {env_var} is not set.[/warning]\n"
    )
    key = Prompt.ask(
        f"  [bold]Paste your {PROVIDER_DISPLAY_NAMES[provider]} API key[/bold]",
        password=True,
    )
    if not key.strip():
        log_error("No API key provided. Aborting.")
        raise typer.Exit(code=1)

    os.environ[env_var] = key.strip()
    log_success(f"{env_var} set for this session.")


def _select_model_interactive(provider: LLMProvider) -> str:
    """Optionally let the user override the default model."""
    default = DEFAULT_MODELS[provider]
    console.print(
        f"\n  [dim]Default model:[/dim] [bold bright_white]{default}[/bold bright_white]"
    )
    use_default = Confirm.ask(
        "  [bold cyan]Use the default model?[/bold cyan]", default=True
    )
    if use_default:
        return default

    model = Prompt.ask("  [bold]Enter model name[/bold]")
    return model.strip() or default


def _sanitize_error(msg: str) -> str:
    """Remove potential secrets from error messages."""
    import re
    # Redact anything that looks like an API key
    sanitized = re.sub(r'(sk-|gsk_|key-)[A-Za-z0-9_-]{10,}', '[REDACTED]', msg)
    # Redact env var values that match known key patterns
    for env_var in API_KEY_ENV_VARS.values():
        val = os.environ.get(env_var, "")
        if val and len(val) > 8 and val in sanitized:
            sanitized = sanitized.replace(val, "[REDACTED]")
    return sanitized


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    topic: Optional[str] = typer.Argument(None, help="The research topic to investigate."),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="LLM provider: groq, openai, anthropic, deepseek.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override the default model for the chosen provider.",
    ),
    strict: bool = typer.Option(
        False, "--strict", "-s", help="Enable strict reviewer mode.",
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", "-i/-I",
        help="Interactive provider & model selection (default: on).",
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume",
        help="Restart a failed/interrupted run from INIT using the prior run's topic and config. "
             "This creates a new run with lineage, not a true checkpoint resume.",
    ),
):
    """
    [bold green]Run[/bold green] the multi-agent research pipeline on a topic.

    Examples:\n
      neuralresearcher run "mamba model optimization"\n
      neuralresearcher run "graph neural networks" --provider openai\n
      neuralresearcher run "vision transformers" -p anthropic -m claude-3-5-sonnet-latest
    """
    print_banner()

    if bool(topic) == bool(resume):
        log_error("You must provide exactly one of: topic or --resume")
        raise typer.Exit(code=1)

    # --- Resolve provider ---
    resolved_provider: Optional[LLMProvider] = None
    if provider:
        try:
            resolved_provider = LLMProvider(provider.lower())
        except ValueError:
            log_error(f"Unknown provider '{provider}'. Choose from: groq, openai, anthropic, deepseek.")
            raise typer.Exit(code=1)
    elif not resume and interactive:
        resolved_provider = _select_provider_interactive()
    elif not resume:
        resolved_provider = LLMProvider.GROQ

    # --- Ensure API key ---
    if resolved_provider:
        _ensure_api_key(resolved_provider)

    # --- Resolve model ---
    resolved_model: Optional[str] = None
    if model:
        resolved_model = model
    elif interactive and not provider and not resume:
        assert resolved_provider is not None
        resolved_model = _select_model_interactive(resolved_provider)
    elif not resume:
        assert resolved_provider is not None
        resolved_model = DEFAULT_MODELS[resolved_provider]

    # --- Summary panel ---
    summary = Table.grid(padding=(0, 2))
    summary.add_row("[dim]Topic:[/dim]", f"[bold bright_white]{topic or 'From stored run'}[/bold bright_white]")

    prov_str = PROVIDER_DISPLAY_NAMES[resolved_provider] if resolved_provider else "From stored run"
    mod_str = resolved_model if resolved_model else "From stored run"
    summary.add_row("[dim]Provider:[/dim]", f"[bold cyan]{prov_str}[/bold cyan]")
    summary.add_row("[dim]Model:[/dim]", f"[bold]{mod_str}[/bold]")
    summary.add_row("[dim]Strict Mode:[/dim]", f"[bold]{'On' if strict else 'Off'}[/bold]")
    console.print(Panel(summary, title="[bold]Run Configuration[/bold]", border_style="cyan", box=box.SIMPLE))
    console.print()

    # --- Run via ResearchService ---
    import asyncio

    from neuralresearcher.application.models import ResumeResearchRequest, StartResearchRequest

    manager = RunManager(data_dir="research")

    async def execute_cli_run():
        if resume:
            try:
                handle = await manager.resume_run(ResumeResearchRequest(run_id=resume, provider=resolved_provider, model=resolved_model))
                run_id = handle.run_id
            except Exception as e:
                log_error(f"Cannot resume run: {_sanitize_error(str(e))}")
                raise typer.Exit(code=1)
        else:
            req = StartResearchRequest(topic=topic, provider=resolved_provider, model=resolved_model, strict=strict)
            handle = await manager.start_run(req)
            run_id = handle.run_id

        console.rule(f"[bold cyan]Pipeline Start (Run ID: {run_id})[/bold cyan]")

        # Poll for completion
        while True:
            status = await manager.get_status(run_id)
            if status.terminal:
                break
            await asyncio.sleep(1)

        return await manager.get_result(run_id), run_id

    try:
        result, run_id = asyncio.run(execute_cli_run())
    except SystemExit:
        raise
    except Exception as e:
        log_error(f"Run failed: {_sanitize_error(str(e))}")
        raise typer.Exit(code=1)

    if result.success:
        console.rule("[bold green]Pipeline Complete[/bold green]")
        # artifact paths dict format is from MCP result
        plan_uri = result.artifact_resource_uris.get("plan", "research_plan.md")
        # Extract filename from URI
        plan_file = plan_uri.split("/")[-1]
        log_success(f"Research plan generated -> research/runs/{run_id}/{plan_file}")
        log_info(f"Run ID: {run_id}")
        raise typer.Exit(code=0)
    else:
        console.rule("[bold red]Pipeline Halted[/bold red]")
        halt_code_val = result.halt_code if result.halt_code else "UNKNOWN"
        log_error(f"Run {result.run_id} failed during {result.failed_stage} with code {halt_code_val}")
        console.print("[dim]Artifacts preserved at:[/dim]")
        for k, v in result.artifact_resource_uris.items():
            console.print(f"  [dim]{k}: {v}[/dim]")
        raise typer.Exit(code=1)


@app.command()
def runs(
    status_filter: Optional[str] = typer.Option(None, "--status", help="Filter by status (ACTIVE, HALTED, etc.)"),
    provider_filter: Optional[str] = typer.Option(None, "--provider", help="Filter by provider."),
    topic_filter: Optional[str] = typer.Option(None, "--topic", help="Filter by topic substring."),
    limit: int = typer.Option(50, "--limit", help="Maximum number of runs to display."),
):
    """
    [bold blue]List[/bold blue] all research runs.
    """
    print_banner()

    import asyncio
    manager = RunManager(data_dir="research")

    async def _list():
        return await manager.list_runs(
            status=status_filter,
            provider=provider_filter,
            topic=topic_filter,
            limit=limit,
        )

    run_list = asyncio.run(_list())

    if not run_list:
        console.print("  [dim]No runs found.[/dim]")
        return

    table = Table(
        title="[bold cyan]Research Runs[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_white",
        border_style="cyan",
    )
    table.add_column("Run ID", style="bold white", max_width=40)
    table.add_column("Topic", style="white", max_width=40)
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Created", style="dim white")

    for r in run_list:
        status_style = "[green]" if r.status == "REPORT_READY" else "[red]" if r.status in ("HALTED", "CORRUPTED") else "[yellow]"
        table.add_row(
            r.run_id[:36] + ("..." if len(r.run_id) > 36 else ""),
            r.topic[:38] + ("..." if len(r.topic) > 38 else ""),
            r.provider,
            f"{status_style}{r.status}[/]",
            r.created_at or "N/A",
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def status(
    run_id: str = typer.Argument(..., help="The run ID to check."),
):
    """
    [bold yellow]Show[/bold yellow] the status of a specific run.
    """
    import asyncio

    from neuralresearcher.errors import StateCorruptionError

    manager = RunManager(data_dir="research")

    async def _status():
        return await manager.get_status(run_id)

    try:
        s = asyncio.run(_status())
    except StateCorruptionError as e:
        log_error(f"Run {run_id} has a corrupted manifest: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        log_error(f"Run {run_id}: {e}")
        raise typer.Exit(code=1)

    summary = Table.grid(padding=(0, 2))
    summary.add_row("[dim]Run ID:[/dim]", f"[bold bright_white]{s.run_id}[/bold bright_white]")
    summary.add_row("[dim]Topic:[/dim]", f"[bold]{s.topic}[/bold]")
    summary.add_row("[dim]Provider:[/dim]", f"[bold cyan]{s.provider}[/bold cyan]")
    summary.add_row("[dim]Model:[/dim]", f"[bold]{s.model}[/bold]")

    status_color = "green" if s.success else "red" if s.terminal else "yellow"
    summary.add_row("[dim]Status:[/dim]", f"[bold {status_color}]{s.status}[/bold {status_color}]")
    summary.add_row("[dim]Current Stage:[/dim]", f"[bold]{s.current_stage or 'N/A'}[/bold]")
    summary.add_row("[dim]Terminal:[/dim]", f"[bold]{'Yes' if s.terminal else 'No'}[/bold]")
    summary.add_row("[dim]Success:[/dim]", f"[bold {'green' if s.success else 'red'}]{'Yes' if s.success else 'No'}[/bold {'green' if s.success else 'red'}]")

    if s.halt_code:
        summary.add_row("[dim]Halt Code:[/dim]", f"[bold red]{s.halt_code}[/bold red]")
    if s.failed_stage:
        summary.add_row("[dim]Failed Stage:[/dim]", f"[bold red]{s.failed_stage}[/bold red]")
    if s.error_message:
        summary.add_row("[dim]Error:[/dim]", f"[red]{s.error_message}[/red]")

    summary.add_row("[dim]Created:[/dim]", f"{s.created_at or 'N/A'}")
    if s.started_at:
        summary.add_row("[dim]Started:[/dim]", f"{s.started_at}")
    if s.finished_at:
        summary.add_row("[dim]Finished:[/dim]", f"{s.finished_at}")
    summary.add_row("[dim]Duration:[/dim]", f"{s.duration_seconds:.1f}s")
    summary.add_row("[dim]Artifacts:[/dim]", f"{s.artifact_count}")
    if s.cancellation_requested:
        summary.add_row("[dim]Cancellation:[/dim]", "[bold yellow]Requested[/bold yellow]")

    console.print()
    console.print(Panel(summary, title=f"[bold]Run Status: {run_id}[/bold]", border_style="cyan", box=box.SIMPLE))
    console.print()


@app.command()
def artifacts(
    run_id: str = typer.Argument(..., help="The run ID to list artifacts for."),
):
    """
    [bold magenta]List[/bold magenta] artifacts generated by a specific run.
    """
    import asyncio

    from neuralresearcher.errors import StateCorruptionError

    manager = RunManager(data_dir="research")

    async def _artifacts():
        return await manager.list_artifacts(run_id)

    try:
        artifact_list = asyncio.run(_artifacts())
    except StateCorruptionError as e:
        log_error(f"Run {run_id} has a corrupted manifest: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        log_error(f"Run {run_id}: {e}")
        raise typer.Exit(code=1)

    if not artifact_list:
        console.print(f"  [dim]No artifacts found for run {run_id}.[/dim]")
        return

    table = Table(
        title=f"[bold cyan]Artifacts for {run_id}[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_white",
        border_style="cyan",
    )
    table.add_column("Name", style="bold white")
    table.add_column("File", style="white")
    table.add_column("Type", style="dim cyan")
    table.add_column("Size", style="dim white", justify="right")
    table.add_column("Resource URI", style="dim yellow")

    for a in artifact_list:
        size_str = f"{a.size_bytes:,} B"
        if a.size_bytes > 1024:
            size_str = f"{a.size_bytes / 1024:.1f} KB"
        table.add_row(a.name, a.path, a.mime_type, size_str, a.resource_uri)

    console.print()
    console.print(table)
    console.print()


@app.command()
def providers():
    """
    [bold blue]List[/bold blue] available LLM providers and their default models.
    """
    print_banner()

    table = Table(
        title="[bold cyan]Supported LLM Providers[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_white",
        border_style="cyan",
    )
    table.add_column("Provider", style="bold white")
    table.add_column("Default Model", style="dim white")
    table.add_column("Env Variable", style="dim yellow")
    table.add_column("Key Set?", justify="center")

    for p in LLMProvider:
        env_var = API_KEY_ENV_VARS[p]
        key_set = "[green]Yes[/green]" if os.environ.get(env_var) else "[red]No[/red]"
        table.add_row(
            PROVIDER_DISPLAY_NAMES[p],
            DEFAULT_MODELS[p],
            env_var,
            key_set,
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def version():
    """
    [bold yellow]Show[/bold yellow] the current version.
    """
    from importlib.metadata import version as pkg_version
    try:
        v = pkg_version("neuralresearcher")
    except Exception:
        from neuralresearcher import __version__
        v = f"{__version__} (dev)"
    console.print(f"  [bold cyan]neuralresearcher[/bold cyan] v{v}")

@app.command()
def mcp(
    transport: str = typer.Option("stdio", help="Transport mode: stdio or streamable-http"),
    host: str = typer.Option("127.0.0.1", help="Host for streamable-http transport"),
    port: int = typer.Option(8000, help="Port for streamable-http transport"),
    path: str = typer.Option("/mcp", help="Path for streamable-http transport"),
):
    """
    [bold magenta]Start[/bold magenta] the MCP server.
    """
    try:
        from neuralresearcher.adapters.mcp.server import start_mcp_server
        start_mcp_server(transport=transport, host=host, port=port, path=path)
    except ImportError as e:
        log_error(f"MCP components are not installed. Install with 'pip install neuralresearcher[mcp]'. Detail: {e}")
        raise typer.Exit(code=1)



if __name__ == "__main__":
    app()
