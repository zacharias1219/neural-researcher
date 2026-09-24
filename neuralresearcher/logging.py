import sys
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
    "agent": "magenta",
    "tool": "blue",
    "dim": "dim white",
})

# Force UTF-8 on Windows to avoid cp1252 encoding errors with special chars
_force_terminal = None
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        _force_terminal = True  # fallback: let Rich handle it

console = Console(theme=custom_theme, force_terminal=_force_terminal)


def log_state_transition(old_state: str, new_state: str) -> None:
    console.print(
        f"  [dim]|[/dim] [info]Transitioning from [bold]{old_state}[/bold] -> [bold]{new_state}[/bold][/info]"
    )


def log_agent_start(agent_name: str) -> None:
    console.print(f"  [agent]> Starting agent: [bold]{agent_name}[/bold][/agent]")


def log_agent_end(agent_name: str) -> None:
    console.print(f"  [success]* Completed agent: [bold]{agent_name}[/bold][/success]")


def log_tool_call(tool_name: str, args: dict | None = None) -> None:
    args_str = str(args) if args else "{}"
    console.print(f"    [tool]~ Tool call: [bold]{tool_name}[/bold] {args_str}[/tool]")


def log_error(msg: str) -> None:
    console.print(f"  [error]x Error: {msg}[/error]")


def log_info(msg: str) -> None:
    console.print(f"  [info]> {msg}[/info]")


def log_success(msg: str) -> None:
    console.print(f"  [success]* {msg}[/success]")


def log_warning(msg: str) -> None:
    console.print(f"  [warning]! {msg}[/warning]")


def print_banner() -> None:
    """Print the startup banner for NeuralResearcher."""

    banner_lines = [
        "+---------------------------------------------+",
        "|                                             |",
        "|    N E U R A L   R E S E A R C H E R        |",
        "|                                             |",
        "|    Multi-Agent Research Plan Generator       |",
        "|                                             |",
        "+---------------------------------------------+",
    ]

    console.print()
    for i, line in enumerate(banner_lines):
        if i == 2:
            console.print(f"  [bold bright_white]{line}[/bold bright_white]")
        elif i == 4:
            console.print(f"  [dim]{line}[/dim]")
        else:
            console.print(f"  [bold cyan]{line}[/bold cyan]")
    console.print()
