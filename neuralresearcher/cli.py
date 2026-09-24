import os
import typer
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich import box

from neuralresearcher.config import (
    Config, LLMProvider, load_config,
    DEFAULT_MODELS, API_KEY_ENV_VARS, PROVIDER_DISPLAY_NAMES,
)
from neuralresearcher.io.store import StateStore
from neuralresearcher.orchestrator import Orchestrator
from neuralresearcher.logging import (
    console, log_info, log_error, log_success, print_banner,
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


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    topic: str = typer.Argument(..., help="The research topic to investigate."),
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
):
    """
    [bold green]Run[/bold green] the multi-agent research pipeline on a topic.

    Examples:\n
      neuralresearcher run "mamba model optimization"\n
      neuralresearcher run "graph neural networks" --provider openai\n
      neuralresearcher run "vision transformers" -p anthropic -m claude-3-5-sonnet-latest
    """
    print_banner()

    # --- Resolve provider ---
    resolved_provider: LLMProvider
    if provider:
        try:
            resolved_provider = LLMProvider(provider.lower())
        except ValueError:
            log_error(f"Unknown provider '{provider}'. Choose from: groq, openai, anthropic, deepseek.")
            raise typer.Exit(code=1)
    elif interactive:
        resolved_provider = _select_provider_interactive()
    else:
        resolved_provider = LLMProvider.GROQ

    # --- Ensure API key ---
    _ensure_api_key(resolved_provider)

    # --- Resolve model ---
    resolved_model: str
    if model:
        resolved_model = model
    elif interactive and not provider:
        resolved_model = _select_model_interactive(resolved_provider)
    else:
        resolved_model = DEFAULT_MODELS[resolved_provider]

    # --- Summary panel ---
    summary = Table.grid(padding=(0, 2))
    summary.add_row("[dim]Topic:[/dim]", f"[bold bright_white]{topic}[/bold bright_white]")
    summary.add_row("[dim]Provider:[/dim]", f"[bold cyan]{PROVIDER_DISPLAY_NAMES[resolved_provider]}[/bold cyan]")
    summary.add_row("[dim]Model:[/dim]", f"[bold]{resolved_model}[/bold]")
    summary.add_row("[dim]Strict Mode:[/dim]", f"[bold]{'On' if strict else 'Off'}[/bold]")
    console.print(Panel(summary, title="[bold]Run Configuration[/bold]", border_style="cyan", box=box.SIMPLE))
    console.print()

    # --- Build & run ---
    try:
        config = load_config(
            provider_override=resolved_provider,
            model_override=resolved_model,
            strict_override=strict,
        )
    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)

    store = StateStore(directory="research")
    orchestrator = Orchestrator(topic=topic, config=config, store=store)

    console.rule("[bold cyan]Pipeline Start[/bold cyan]")
    orchestrator.run()
    console.rule("[bold green]Pipeline Complete[/bold green]")

    log_success("Research plan generated -> research/research_plan.md")


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


if __name__ == "__main__":
    app()
