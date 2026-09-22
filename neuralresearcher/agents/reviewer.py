from typing import Any

def run_reviewer(orchestrator: Any) -> None:
    # In v1, review can be a pass-through or basic validation.
    from neuralresearcher.logging import log_info
    log_info("Reviewer agent executed. Plan accepted.")
