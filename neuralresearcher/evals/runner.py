import typer
import time
import uuid
import json
import shutil
from pathlib import Path
from neuralresearcher.config import Config
from neuralresearcher.io.store import StateStore
from neuralresearcher.orchestrator import Orchestrator
from neuralresearcher.evals.tasks import core_suite
from neuralresearcher.evals.core import Trial
from neuralresearcher.evals.graders import grade_completion, grade_correctness
from neuralresearcher.logging import log_info, log_error

app = typer.Typer()

@app.command()
def run_suite(
    suite_name: str = "core", 
    trials_per_task: int = 1, 
    seed: int = 42,
    provider: str = "openai",
    model: str = typer.Option(None)
):
    if suite_name != "core":
        log_error(f"Unknown suite: {suite_name}")
        raise typer.Exit(code=1)
        
    suite = core_suite
    results = []
    
    # We will write everything to a global evals directory
    eval_dir = Path.cwd() / "research" / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    from neuralresearcher.config import LLMProvider, load_config
    try:
        resolved_provider = LLMProvider(provider.lower())
    except ValueError:
        log_error(f"Unknown provider '{provider}'")
        raise typer.Exit(code=1)
    
    for task in suite.tasks:
        log_info(f"Running EvalTask: {task.id} (topic: {task.topic})")
        
        for i in range(trials_per_task):
            run_id = f"{task.id}_run_{i}_{uuid.uuid4().hex[:6]}"
            trial_dir = eval_dir / run_id
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            
            store = StateStore(directory=str(trial_dir))
            
            try:
                config = load_config(
                    provider_override=resolved_provider,
                    model_override=model
                )
                config.temperature = 0.0
                config.seed = seed
                config.max_retries = 3
            except Exception as e:
                log_error(str(e))
                raise typer.Exit(code=1)
            
            orchestrator = Orchestrator(topic=task.topic, config=config, store=store, task_id=run_id)
            
            start_time = time.time()
            orchestrator.run()
            duration = time.time() - start_time
            
            # Count total tokens from transcripts
            total_tokens = 0
            transcripts = store.load_transcripts(run_id)
            for t in transcripts:
                total_tokens += t.get("usage", {}).get("total_tokens", 0)
                
            completion_outcome = grade_completion(orchestrator, task)
            correctness_outcome = grade_correctness(orchestrator, task)
            
            success = completion_outcome.passed and correctness_outcome.passed
            
            trial = Trial(
                task_id=task.id,
                run_id=run_id,
                topic=task.topic,
                success=success,
                outcomes={
                    "completion": completion_outcome,
                    "correctness": correctness_outcome
                },
                duration_sec=duration,
                total_tokens=total_tokens
            )
            
            results.append(trial.model_dump())
            log_info(f"  Trial {i} success: {success}")
            
    # Save aggregate results
    results_file = eval_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        
    log_info(f"Saved eval results to {results_file}")
    
    # Return non-zero exit code if any trial failed
    if any(not r["success"] for r in results):
        log_error("Some eval tasks failed.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
