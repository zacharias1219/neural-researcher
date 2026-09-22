import typer
from neuralresearcher.config import load_config
from neuralresearcher.io.store import StateStore
from neuralresearcher.orchestrator import Orchestrator
from neuralresearcher.logging import log_info, log_error

app = typer.Typer(help="Terminal-based multi-agent research planner.")

@app.command()
def main(
    topic: str = typer.Argument(..., help="The research topic to investigate."),
    model: str = typer.Option(None, "--model", "-m", help="Override the default Groq model."),
    strict: bool = typer.Option(False, "--strict", "-s", help="Enable strict review mode.")
):
    """
    Given a research topic, neuralresearcher uses a multi-agent pipeline
    to output a detailed research_plan.md.
    """
    log_info(f"Starting neuralresearcher for topic: '{topic}'")
    
    try:
        config = load_config(model_override=model, strict_override=strict)
    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)
        
    store = StateStore(directory="research")
    orchestrator = Orchestrator(topic=topic, config=config, store=store)
    
    orchestrator.run()
    
if __name__ == "__main__":
    app()
