from neuralresearcher.evals.core import EvalTask, EvalSuite

CORE_TASKS = [
    EvalTask(
        id="task_mamba_optimization",
        topic="mamba model optimization",
        description="Generic mamba optimization survey",
        success_criteria={"min_papers": 3, "min_gaps": 2}
    ),
    EvalTask(
        id="task_mamba_time_series",
        topic="Mamba models for time-series forecasting",
        description="Mamba applied to time-series",
        success_criteria={"min_papers": 2, "required_keyword": "time-series"}
    ),
    EvalTask(
        id="task_mamba_edge",
        topic="Edge Mamba deployment",
        description="Edge deployment of mamba architectures",
        success_criteria={"min_papers": 2}
    ),
    EvalTask(
        id="task_neg_shakespeare",
        topic="Shakespearean sonnets analysis",
        description="Out of scope topic, should halt early",
        negative_task=True,
        success_criteria={"expect_halt": True}
    ),
    EvalTask(
        id="task_neg_over_trigger",
        topic="The GneebGnab Architecture for Quantum-Linguistic Forecasting",
        description="Extremely fake topic to ensure retrieval finds 0 papers",
        negative_task=True,
        success_criteria={"expect_halt": True}
    )
]

core_suite = EvalSuite(name="core", tasks=CORE_TASKS)
