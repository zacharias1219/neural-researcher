import typer
import time
import uuid
import json
import shutil
from neuralresearcher.evals.core import Outcome
from pathlib import Path
from neuralresearcher.io.store import StateStore
from neuralresearcher.orchestrator import Orchestrator
from neuralresearcher.evals.tasks import core_suite
from neuralresearcher.evals.core import Trial
from neuralresearcher.evals.graders import grade_completion, grade_correctness, grade_efficiency
from neuralresearcher.logging import log_info, log_error, log_warning

app = typer.Typer()

@app.command()
def run_suite(
    suite_name: str = "core", 
    trials_per_task: int = 1, 
    seed: int = 42,
    providers: str = "openai,anthropic"
):
    if suite_name != "core":
        log_error(f"Unknown suite: {suite_name}")
        raise typer.Exit(code=1)
        
    suite = core_suite
    results = []
    
    # We will write everything to a global evals directory
    eval_dir = Path.cwd() / "research" / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    
    provider_names = [p.strip() for p in providers.split(",")]
    resolved_providers = []
    from neuralresearcher.config import LLMProvider, load_config
    from neuralresearcher.cli import _ensure_api_key
    
    for p in provider_names:
        try:
            rp = LLMProvider(p.lower())
            _ensure_api_key(rp)
            resolved_providers.append(rp)
        except ValueError:
            log_warning(f"Unknown provider '{p}', skipping.")
        except Exception as e:
            log_warning(f"Failed to setup API key for {p}: {e}, skipping.")
            
    if not resolved_providers:
        log_error("No valid providers configured.")
        raise typer.Exit(code=1)
    
    for task in suite.tasks:
        log_info(f"Running EvalTask: {task.id} (topic: {task.topic})")
        task_trials = []
        
        for rp in resolved_providers:
            log_info(f"  Provider: {rp.value}")
            for i in range(trials_per_task):
                run_id = f"{task.id}_{rp.value}_run_{i}_{uuid.uuid4().hex[:6]}"
                trial_dir = eval_dir / run_id
                if trial_dir.exists():
                    shutil.rmtree(trial_dir)
                
                store = StateStore(directory=str(trial_dir))
                
                try:
                    config = load_config(provider_override=rp)
                    config.temperature = 0.0
                    config.seed = seed
                    config.api_retries = 3
                except Exception as e:
                    log_error(str(e))
                    continue
                
            # Apply task-specific configurations
            if "min_papers" in task.success_criteria:
                config.min_papers = task.success_criteria["min_papers"]
            
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
            
            efficiency_outcome = grade_efficiency(duration_sec=duration, total_tokens=total_tokens)
            
            success = completion_outcome.passed and correctness_outcome.passed and efficiency_outcome.passed
            
            trial = Trial(
                task_id=task.id,
                run_id=run_id,
                topic=task.topic,
                success=success,
                outcomes={
                    "completion": completion_outcome,
                    "correctness": correctness_outcome,
                    "efficiency": efficiency_outcome
                },
                duration_sec=duration,
                total_tokens=total_tokens
            )
            task_trials.append(trial)
            if success:
                log_info(f"    Trial {i} success: {success} (score: {correctness_outcome.score})")
            else:
                log_error(f"    Trial {i} failed. Completion: {completion_outcome.details} | Correctness: {correctness_outcome.details}")
            
        # Cross-trial consistency
        if len(task_trials) > 1:
            scores = [t.outcomes["correctness"].score for t in task_trials]
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency = max(0.0, 1.0 - variance)  # Simple variance-based consistency score
            log_info(f"  Task {task.id} consistency: {consistency:.2f} (avg score: {avg_score:.2f})")
            
            for t in task_trials:
                t.outcomes["consistency"] = Outcome(score=consistency, passed=consistency > 0.8, details=f"Variance: {variance:.3f}")
                
        results.extend([t.model_dump() for t in task_trials])
            
    # Save aggregate results
    results_file = eval_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        
    log_info(f"Saved eval results to {results_file}")
    
    # Return non-zero exit code if any trial failed
    failed_trials = [r for r in results if not r["success"]]
    if failed_trials:
        log_error(f"{len(failed_trials)} eval tasks failed. See {results_file} for details.")
        for r in failed_trials:
            log_error(f"Failed Run {r['run_id']}: Correctness: {r['outcomes']['correctness']['details']}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
