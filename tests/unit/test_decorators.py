"""Unit tests for workflow and task decorators."""

from datetime import timedelta

from pydantic import BaseModel

from flovyn import RetryPolicy, TaskContext, WorkflowContext, task, workflow
from flovyn.task import get_task_metadata, is_task
from flovyn.workflow import get_workflow_metadata, is_workflow


class SampleInput(BaseModel):
    value: int


class SampleOutput(BaseModel):
    result: int


class TestWorkflowDecorator:
    def test_class_based_workflow(self) -> None:
        @workflow(name="test-workflow")
        class TestWorkflow:
            async def run(self, ctx: WorkflowContext, input: SampleInput) -> SampleOutput:
                return SampleOutput(result=input.value * 2)

        assert is_workflow(TestWorkflow)
        metadata = get_workflow_metadata(TestWorkflow)
        assert metadata is not None
        assert metadata.kind == "test-workflow"
        assert metadata.name == "test-workflow"

    def test_function_based_workflow(self) -> None:
        @workflow(name="func-workflow")
        async def my_workflow(ctx: WorkflowContext, input: SampleInput) -> SampleOutput:
            return SampleOutput(result=input.value)

        assert is_workflow(my_workflow)
        metadata = get_workflow_metadata(my_workflow)
        assert metadata is not None
        assert metadata.kind == "func-workflow"

    def test_workflow_with_options(self) -> None:
        @workflow(
            name="configured-workflow",
            description="A test workflow",
            version="1.0.0",
            tags=["test", "sample"],
            timeout=timedelta(hours=1),
        )
        class ConfiguredWorkflow:
            async def run(self, ctx: WorkflowContext, input: SampleInput) -> SampleOutput:
                return SampleOutput(result=input.value)

        metadata = get_workflow_metadata(ConfiguredWorkflow)
        assert metadata is not None
        assert metadata.description == "A test workflow"
        assert metadata.version == "1.0.0"
        assert metadata.tags == ["test", "sample"]
        assert metadata.timeout == timedelta(hours=1)

    def test_non_workflow_returns_false(self) -> None:
        class NotAWorkflow:
            pass

        assert not is_workflow(NotAWorkflow)


class TestTaskDecorator:
    def test_class_based_task(self) -> None:
        @task(name="test-task")
        class TestTask:
            async def run(self, ctx: TaskContext, input: SampleInput) -> SampleOutput:
                return SampleOutput(result=input.value * 3)

        assert is_task(TestTask)
        metadata = get_task_metadata(TestTask)
        assert metadata is not None
        assert metadata.kind == "test-task"
        assert metadata.name == "test-task"

    def test_function_based_task(self) -> None:
        @task(name="func-task")
        async def my_task(ctx: TaskContext, input: SampleInput) -> SampleOutput:
            return SampleOutput(result=input.value)

        assert is_task(my_task)
        metadata = get_task_metadata(my_task)
        assert metadata is not None
        assert metadata.kind == "func-task"

    def test_task_with_retry_policy(self) -> None:
        @task(
            name="retry-task",
            timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
            ),
        )
        class RetryTask:
            async def run(self, ctx: TaskContext, input: SampleInput) -> SampleOutput:
                return SampleOutput(result=input.value)

        metadata = get_task_metadata(RetryTask)
        assert metadata is not None
        assert metadata.timeout == timedelta(minutes=5)
        assert metadata.retry_policy is not None
        assert metadata.retry_policy.max_attempts == 3

    def test_non_task_returns_false(self) -> None:
        class NotATask:
            pass

        assert not is_task(NotATask)
