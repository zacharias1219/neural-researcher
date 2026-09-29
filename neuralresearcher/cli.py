import os
import typer
from typing import Optional

from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

from neuralresearcher.config import (
    LLMProvider, load_config,
    DEFAULT_MODELS, API_KEY_ENV_VARS, PROVIDER_DISPLAY_NAMES,
)
from neuralresearcher.io.store import StateStore
from neuralresearcher.orchestrator import Orchestrator
from neuralresearcher.logging import (
    console, log_info, log_error, log_warning, log_success, print_banner,
)
from neuralresearcher.application.run_manager import RunManager

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
        None, "--resume", help="Resume an existing run ID."
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
        resolved_model = _select_model_interactive(resolved_provider)
    elif not resume:
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
    from neuralresearcher.application.models import StartResearchRequest, ResumeResearchRequest

    manager = RunManager(data_dir="research")
    
    async def execute_cli_run():
        if resume:
            try:
                handle = await manager.resume_run(ResumeResearchRequest(run_id=resume, provider=resolved_provider, model=resolved_model))
                store_run_id = resume
            except Exception as e:
                log_error(f"Cannot resume run: {e}")
                raise typer.Exit(code=1)
        else:
            req = StartResearchRequest(topic=topic, provider=resolved_provider, model=resolved_model, strict=strict)
            handle = await manager.start_run(req)
            store_run_id = handle.run_id

        console.rule(f"[bold cyan]Pipeline Start (Run ID: {store_run_id})[/bold cyan]")
        
        # Poll for completion
        while True:
            status = await manager.get_status(store_run_id)
            if status.terminal:
                break
            await asyncio.sleep(1)
            
        return await manager.get_result(store_run_id), store_run_id

    result, run_id = asyncio.run(execute_cli_run())
    
    if result.success:
        console.rule("[bold green]Pipeline Complete[/bold green]")
        # artifact paths dict format is from MCP result
        plan_uri = result.artifact_resource_uris.get("plan", "research_plan.md")
        # Extract filename from URI
        plan_file = plan_uri.split("/")[-1]
        log_success(f"Research plan generated -> research/runs/{run_id}/{plan_file}")
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
        v = "0.1.0 (dev)"
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
