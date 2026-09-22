from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
    "agent": "magenta",
    "tool": "blue"
})

console = Console(theme=custom_theme)

def log_state_transition(old_state: str, new_state: str) -> None:
    console.print(f"[info]Transitioning from [bold]{old_state}[/bold] to [bold]{new_state}[/bold][/info]")

def log_agent_start(agent_name: str) -> None:
    console.print(f"[agent]> Starting agent: {agent_name}[/agent]")

def log_agent_end(agent_name: str) -> None:
    console.print(f"[agent]* Completed agent: {agent_name}[/agent]")

def log_tool_call(tool_name: str, args: dict | None = None) -> None:
    args_str = str(args) if args else "{}"
    console.print(f"[tool]  - Calling tool: {tool_name} with {args_str}[/tool]")

def log_error(msg: str) -> None:
    console.print(f"[error]x Error: {msg}[/error]")

def log_info(msg: str) -> None:
    console.print(f"[info]{msg}[/info]")

def log_success(msg: str) -> None:
    console.print(f"[success]* {msg}[/success]")
