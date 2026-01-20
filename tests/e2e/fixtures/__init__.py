"""E2E test fixtures for workflows and tasks."""

from tests.e2e.fixtures.tasks import (
    AddTask,
    EchoTask,
    FailingTask,
    ProgressTask,
    SlowTask,
)
from tests.e2e.fixtures.workflows import (
    AwaitPromiseWorkflow,
    ChildWorkflowWorkflow,
    DoublerWorkflow,
    EchoWorkflow,
    FailingWorkflow,
    MultiTaskWorkflow,
    ParallelTasksWorkflow,
    PromiseWorkflow,
    RandomWorkflow,
    RunOperationWorkflow,
    SleepWorkflow,
    StatefulWorkflow,
    TaskSchedulingWorkflow,
)

__all__ = [
    # Workflows
    "EchoWorkflow",
    "DoublerWorkflow",
    "FailingWorkflow",
    "StatefulWorkflow",
    "RunOperationWorkflow",
    "RandomWorkflow",
    "SleepWorkflow",
    "PromiseWorkflow",
    "AwaitPromiseWorkflow",
    "TaskSchedulingWorkflow",
    "MultiTaskWorkflow",
    "ParallelTasksWorkflow",
    "ChildWorkflowWorkflow",
    # Tasks
    "EchoTask",
    "AddTask",
    "SlowTask",
    "FailingTask",
    "ProgressTask",
]
