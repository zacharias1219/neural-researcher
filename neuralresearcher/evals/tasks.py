from neuralresearcher.evals.core import EvalTask, EvalSuite

CORE_TASKS = [
    EvalTask(
        id="task_mamba_optimization",
        topic="mamba model optimization",
        description="Generic mamba optimization survey",
        success_criteria={"min_papers": 3, "min_gaps": 2}
    ),
    EvalTask(
        id="task_mamba_mpc",
        topic="Time-series Mamba MPC",
        description="Time-series Mamba MPC",
        success_criteria={"min_papers": 2, "required_keyword": "MPC"}
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
        topic="Mamba block size parameter exact value optimization for a 3 billion parameter model",
        description="Extremely narrow topic",
        negative_task=True,
        success_criteria={"expect_halt": True}
    )
]

core_suite = EvalSuite(name="core", tasks=CORE_TASKS)
