"""Workflow fixtures for E2E tests."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pydantic import BaseModel

from flovyn import WorkflowContext, workflow

# Input/Output models


class EchoInput(BaseModel):
    message: str


class EchoOutput(BaseModel):
    message: str
    timestamp: str


class DoublerInput(BaseModel):
    value: int


class DoublerOutput(BaseModel):
    result: int


class FailingInput(BaseModel):
    error_message: str


class StatefulInput(BaseModel):
    key: str
    value: str


class StatefulOutput(BaseModel):
    stored_value: str | None
    all_keys: list[str]


class RunOperationInput(BaseModel):
    operation_name: str


class RunOperationOutput(BaseModel):
    result: str


class RandomInput(BaseModel):
    pass


class RandomOutput(BaseModel):
    uuid: str
    random_int: int
    random_float: float


class SleepInput(BaseModel):
    duration_ms: int


class SleepOutput(BaseModel):
    slept_duration_ms: int
    start_time: str
    end_time: str


class PromiseInput(BaseModel):
    promise_name: str


class PromiseOutput(BaseModel):
    promise_id: str


class AwaitPromiseInput(BaseModel):
    promise_name: str
    timeout_ms: int | None = None


class AwaitPromiseOutput(BaseModel):
    resolved_value: Any


class TaskSchedulingInput(BaseModel):
    count: int


class TaskSchedulingOutput(BaseModel):
    results: list[int]
    total: int


class ChildWorkflowInput(BaseModel):
    child_input: Any


class ChildWorkflowOutput(BaseModel):
    child_result: Any


# Workflows


@workflow(name="echo-workflow")
class EchoWorkflow:
    """Simple workflow that echoes input back with a timestamp."""

    async def run(self, ctx: WorkflowContext, input: EchoInput) -> EchoOutput:
        timestamp = ctx.current_time().isoformat()
        return EchoOutput(
            message=input.message,
            timestamp=timestamp,
        )


@workflow(name="doubler-workflow")
class DoublerWorkflow:
    """Workflow that doubles the input value."""

    async def run(self, ctx: WorkflowContext, input: DoublerInput) -> DoublerOutput:
        return DoublerOutput(result=input.value * 2)


@workflow(name="failing-workflow")
class FailingWorkflow:
    """Workflow that always fails with a configured error message."""

    async def run(self, ctx: WorkflowContext, input: FailingInput) -> None:
        raise Exception(input.error_message)


@workflow(name="stateful-workflow")
class StatefulWorkflow:
    """Workflow that tests state get/set/clear operations."""

    async def run(self, ctx: WorkflowContext, input: StatefulInput) -> StatefulOutput:
        # Set state
        await ctx.set(input.key, input.value)

        # Get state back
        stored = await ctx.get(input.key, type_hint=str)

        # Get all keys (by checking a few known ones)
        keys = []
        if await ctx.get(input.key) is not None:
            keys.append(input.key)

        return StatefulOutput(
            stored_value=stored,
            all_keys=keys,
        )


@workflow(name="run-operation-workflow")
class RunOperationWorkflow:
    """Workflow that tests ctx.run() for durable side effects."""

    async def run(self, ctx: WorkflowContext, input: RunOperationInput) -> RunOperationOutput:
        # Run an operation that would be non-deterministic
        result = await ctx.run(
            input.operation_name,
            lambda: f"executed-{input.operation_name}",
        )

        return RunOperationOutput(result=result)


@workflow(name="random-workflow")
class RandomWorkflow:
    """Workflow that tests deterministic random generation."""

    async def run(self, ctx: WorkflowContext, input: RandomInput) -> RandomOutput:
        uuid = ctx.random_uuid()
        random_value = ctx.random()

        return RandomOutput(
            uuid=str(uuid),
            random_int=int(random_value * 1000),
            random_float=random_value,
        )


@workflow(name="sleep-workflow")
class SleepWorkflow:
    """Workflow that tests durable timers."""

    async def run(self, ctx: WorkflowContext, input: SleepInput) -> SleepOutput:
        start_time = ctx.current_time()

        await ctx.sleep(timedelta(milliseconds=input.duration_ms))

        end_time = ctx.current_time()

        return SleepOutput(
            slept_duration_ms=input.duration_ms,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )


@workflow(name="promise-workflow")
class PromiseWorkflow:
    """Workflow that creates a promise."""

    async def run(self, ctx: WorkflowContext, input: PromiseInput) -> PromiseOutput:
        # Create a promise (this will suspend waiting for resolution)
        # For the test, we just return the promise name as ID
        return PromiseOutput(promise_id=input.promise_name)


@workflow(name="await-promise-workflow")
class AwaitPromiseWorkflow:
    """Workflow that waits for an external promise to be resolved."""

    async def run(self, ctx: WorkflowContext, input: AwaitPromiseInput) -> AwaitPromiseOutput:
        timeout = timedelta(milliseconds=input.timeout_ms) if input.timeout_ms else None

        value = await ctx.promise(
            input.promise_name,
            timeout=timeout,
            type_hint=Any,
        )

        return AwaitPromiseOutput(resolved_value=value)


@workflow(name="task-scheduling-workflow")
class TaskSchedulingWorkflow:
    """Workflow that schedules multiple tasks and aggregates results."""

    async def run(self, ctx: WorkflowContext, input: TaskSchedulingInput) -> TaskSchedulingOutput:
        results = []
        running_total = 0

        for i in range(input.count):
            result = await ctx.schedule(
                "add-task",
                {"a": running_total, "b": i + 1},
            )
            running_total = result["sum"]
            results.append(running_total)

        return TaskSchedulingOutput(
            results=results,
            total=running_total,
        )


@workflow(name="multi-task-workflow")
class MultiTaskWorkflow:
    """Workflow that executes multiple tasks sequentially."""

    async def run(self, ctx: WorkflowContext, input: TaskSchedulingInput) -> TaskSchedulingOutput:
        results = []
        total = 0

        for i in range(input.count):
            result = await ctx.schedule(
                "add-task",
                {"a": i, "b": i},
            )
            total += result["sum"]
            results.append(result["sum"])

        return TaskSchedulingOutput(
            results=results,
            total=total,
        )


@workflow(name="parallel-tasks-workflow")
class ParallelTasksWorkflow:
    """Workflow that executes multiple tasks in parallel."""

    async def run(self, ctx: WorkflowContext, input: TaskSchedulingInput) -> TaskSchedulingOutput:
        # Schedule all tasks
        handles = []
        for i in range(input.count):
            handle = ctx.schedule_async(
                "add-task",
                {"a": i, "b": i},
            )
            handles.append(handle)

        # Await all results
        results = []
        total = 0
        for handle in handles:
            result = await handle.result()
            results.append(result["sum"])
            total += result["sum"]

        return TaskSchedulingOutput(
            results=results,
            total=total,
        )


@workflow(name="child-workflow-workflow")
class ChildWorkflowWorkflow:
    """Workflow that executes a child workflow."""

    async def run(self, ctx: WorkflowContext, input: ChildWorkflowInput) -> ChildWorkflowOutput:
        # Execute the echo workflow as a child
        result = await ctx.schedule_workflow(
            "echo-workflow",
            {"message": str(input.child_input)},
        )

        return ChildWorkflowOutput(child_result=result)


class ChildFailureInput(BaseModel):
    error_message: str


class ChildFailureOutput(BaseModel):
    caught_error: str


@workflow(name="child-failure-workflow")
class ChildFailureWorkflow:
    """Workflow that tests child workflow failure handling."""

    async def run(self, ctx: WorkflowContext, input: ChildFailureInput) -> ChildFailureOutput:
        from flovyn.exceptions import ChildWorkflowFailed

        try:
            await ctx.schedule_workflow(
                "failing-workflow",
                {"error_message": input.error_message},
            )
            return ChildFailureOutput(caught_error="")
        except ChildWorkflowFailed as e:
            return ChildFailureOutput(caught_error=str(e))
        except Exception as e:
            return ChildFailureOutput(caught_error=f"Unexpected: {e}")


class NestedChildInput(BaseModel):
    depth: int
    value: str


class NestedChildOutput(BaseModel):
    result: str
    levels: int


@workflow(name="nested-child-workflow")
class NestedChildWorkflow:
    """Workflow that can be nested to test multi-level child workflows."""

    async def run(self, ctx: WorkflowContext, input: NestedChildInput) -> NestedChildOutput:
        if input.depth <= 1:
            # Base case: just return the value
            return NestedChildOutput(result=f"leaf:{input.value}", levels=1)
        else:
            # Recursive case: call child workflow with reduced depth
            child_result = await ctx.schedule_workflow(
                "nested-child-workflow",
                {"depth": input.depth - 1, "value": input.value},
            )
            return NestedChildOutput(
                result=f"level{input.depth}:{child_result['result']}",
                levels=child_result["levels"] + 1,
            )


# Mixed Commands Workflow (for replay testing)


class MixedCommandsInput(BaseModel):
    value: int


class MixedCommandsOutput(BaseModel):
    operation_result: str
    sleep_completed: bool
    task_result: int
    final_value: int


@workflow(name="mixed-commands-workflow")
class MixedCommandsWorkflow:
    """Workflow that tests mixed command types for replay verification.

    Executes operations, timers, and tasks in sequence to verify
    per-type sequence matching during replay.
    """

    async def run(self, ctx: WorkflowContext, input: MixedCommandsInput) -> MixedCommandsOutput:
        # Step 1: Run a side-effect operation
        op_result = await ctx.run("compute-step", lambda: f"computed-{input.value}")

        # Step 2: Sleep for a short duration
        await ctx.sleep(timedelta(milliseconds=100))

        # Step 3: Execute a task
        task_result = await ctx.schedule(
            "add-task",
            {"a": input.value, "b": 10},
        )

        # Step 4: Run another operation
        final = await ctx.run("finalize-step", lambda: input.value * 2)

        return MixedCommandsOutput(
            operation_result=op_result,
            sleep_completed=True,
            task_result=task_result["sum"],
            final_value=final,
        )


# Fan-out/Fan-in Workflow (for parallel testing)


class FanOutInput(BaseModel):
    items: list[str]


class FanOutOutput(BaseModel):
    input_count: int
    output_count: int
    processed_items: list[str]
    total_length: int


@workflow(name="fan-out-fan-in-workflow")
class FanOutFanInWorkflow:
    """Workflow that demonstrates fan-out/fan-in pattern.

    Schedules multiple tasks in parallel and aggregates results.
    """

    async def run(self, ctx: WorkflowContext, input: FanOutInput) -> FanOutOutput:
        # Fan-out: Schedule all tasks in parallel
        handles = []
        for item in input.items:
            handle = ctx.schedule_async(
                "echo-task",
                {"message": item},
            )
            handles.append(handle)

        # Fan-in: Collect all results
        processed = []
        total_length = 0
        for handle in handles:
            result = await handle.result()
            processed.append(result["message"])
            total_length += len(result["message"])

        return FanOutOutput(
            input_count=len(input.items),
            output_count=len(processed),
            processed_items=processed,
            total_length=total_length,
        )


# Large Batch Workflow (for parallel stress testing)


class LargeBatchInput(BaseModel):
    count: int


class LargeBatchOutput(BaseModel):
    task_count: int
    total: int
    min_value: int
    max_value: int


@workflow(name="large-batch-workflow")
class LargeBatchWorkflow:
    """Workflow that schedules many parallel tasks.

    Tests scalability of parallel task execution.
    """

    async def run(self, ctx: WorkflowContext, input: LargeBatchInput) -> LargeBatchOutput:
        # Schedule many tasks in parallel
        handles = []
        for i in range(input.count):
            handle = ctx.schedule_async(
                "add-task",
                {"a": i, "b": 1},
            )
            handles.append(handle)

        # Collect results
        results = []
        for handle in handles:
            result = await handle.result()
            results.append(result["sum"])

        return LargeBatchOutput(
            task_count=len(results),
            total=sum(results),
            min_value=min(results),
            max_value=max(results),
        )


# Mixed Parallel Operations Workflow


class MixedParallelInput(BaseModel):
    pass


class MixedParallelOutput(BaseModel):
    success: bool
    phase1_results: list[str]
    timer_fired: bool
    phase3_results: list[int]


@workflow(name="mixed-parallel-workflow")
class MixedParallelWorkflow:
    """Workflow that combines parallel tasks with timers.

    Tests mixed parallel operations:
    1. Phase 1: Two parallel tasks
    2. Timer: Wait for 100ms
    3. Phase 3: Three parallel add tasks
    """

    async def run(self, ctx: WorkflowContext, input: MixedParallelInput) -> MixedParallelOutput:
        # Phase 1: Two parallel echo tasks
        handle1 = ctx.schedule_async("echo-task", {"message": "task-1"})
        handle2 = ctx.schedule_async("echo-task", {"message": "task-2"})

        result1 = await handle1.result()
        result2 = await handle2.result()
        phase1_results = [result1["message"], result2["message"]]

        # Phase 2: Timer
        await ctx.sleep(timedelta(milliseconds=100))
        timer_fired = True

        # Phase 3: Three parallel add tasks
        handles = []
        for i in range(3):
            handle = ctx.schedule_async("add-task", {"a": i, "b": i})
            handles.append(handle)

        phase3_results = []
        for handle in handles:
            result = await handle.result()
            phase3_results.append(result["sum"])

        return MixedParallelOutput(
            success=True,
            phase1_results=phase1_results,
            timer_fired=timer_fired,
            phase3_results=phase3_results,
        )


# Child Workflow Loop Workflow


class ChildLoopInput(BaseModel):
    count: int


class ChildLoopOutput(BaseModel):
    results: list[str]
    total_count: int


@workflow(name="child-loop-workflow")
class ChildLoopWorkflow:
    """Workflow that executes child workflows in a loop.

    Tests replay with multiple child workflow invocations.
    """

    async def run(self, ctx: WorkflowContext, input: ChildLoopInput) -> ChildLoopOutput:
        results = []

        for i in range(input.count):
            # Execute child echo workflow for each iteration
            result = await ctx.schedule_workflow(
                "echo-workflow",
                {"message": f"child-{i}"},
            )
            results.append(result["message"])

        return ChildLoopOutput(
            results=results,
            total_count=len(results),
        )


# Comprehensive Workflow (tests multiple features)


class ComprehensiveInput(BaseModel):
    value: int


class ComprehensiveOutput(BaseModel):
    input_value: int
    run_result: int
    state_set: bool
    state_retrieved: dict[str, Any] | None
    state_matches: bool
    triple_result: int
    tests_passed_count: int
    tests_passed: list[str]


@workflow(name="comprehensive-workflow")
class ComprehensiveWorkflow:
    """Workflow that tests multiple SDK features in a single execution.

    Tests (matching Rust SDK):
    - Basic input processing
    - Operation recording (ctx.run)
    - State set/get operations
    - Multiple operations in sequence
    """

    async def run(self, ctx: WorkflowContext, input: ComprehensiveInput) -> ComprehensiveOutput:
        tests_passed: list[str] = []

        # Test 1: Basic input processing
        input_value = input.value
        tests_passed.append("basic_input")

        # Test 2: Operation recording with ctx.run()
        run_result = await ctx.run("double-operation", lambda: input.value * 2)
        tests_passed.append("run_operation")

        # Test 3: State set
        state_key = "test-state-key"
        state_value = {
            "counter": input.value,
            "message": "state test",
            "nested": {"a": 1, "b": 2},
        }
        await ctx.set(state_key, state_value)
        tests_passed.append("state_set")

        # Test 4: State get (should return what we just set)
        retrieved = await ctx.get(state_key, type_hint=dict)

        # Verify state matches
        state_matches = retrieved == state_value
        if state_matches:
            tests_passed.append("state_get")

        # Test 5: Multiple operations to test replay
        triple_result = await ctx.run("triple-operation", lambda: input.value * 3)
        tests_passed.append("multiple_operations")

        return ComprehensiveOutput(
            input_value=input_value,
            run_result=run_result,
            state_set=True,
            state_retrieved=retrieved,
            state_matches=state_matches,
            triple_result=triple_result,
            tests_passed_count=len(tests_passed),
            tests_passed=tests_passed,
        )


# Task Scheduler Workflow - for testing arbitrary tasks


class TaskSchedulerInput(BaseModel):
    task_name: str
    """The name of the task to schedule."""
    task_input: dict[str, Any]
    """The input to pass to the task."""


class TaskSchedulerOutput(BaseModel):
    task_completed: bool
    task_result: dict[str, Any] | None


@workflow(name="task-scheduler-workflow")
class TaskSchedulerWorkflow:
    """Generic workflow that schedules a task by name.

    Used for testing arbitrary tasks without needing a specific workflow.
    """

    async def run(self, ctx: WorkflowContext, input: TaskSchedulerInput) -> TaskSchedulerOutput:
        # Known task names for validation
        known_tasks = {
            "streaming-token-task",
            "streaming-progress-task",
            "streaming-data-task",
            "streaming-error-task",
            "streaming-all-types-task",
        }

        if input.task_name not in known_tasks:
            return TaskSchedulerOutput(
                task_completed=False,
                task_result={"error": f"Unknown task: {input.task_name}"},
            )

        result = await ctx.schedule(input.task_name, input.task_input)

        return TaskSchedulerOutput(
            task_completed=True,
            task_result=result if isinstance(result, dict) else dict(result),
        )


# ============================================================================
# Typed API Workflows (for testing single-server use case)
# ============================================================================


class TypedTaskInput(BaseModel):
    a: int
    b: int


class TypedTaskOutput(BaseModel):
    result: int


@workflow(name="typed-task-workflow")
class TypedTaskWorkflow:
    """Workflow that uses the typed API to execute a task.

    This workflow demonstrates the typed API where you pass the task class
    instead of a string. This is useful for single-server deployments
    where the client and worker are on the same machine.
    """

    async def run(self, ctx: WorkflowContext, input: TypedTaskInput) -> TypedTaskOutput:
        # Import the task class for typed API
        from tests.e2e.fixtures.tasks import AddTask

        # Use the typed API: pass the class instead of string "add-task"
        result = await ctx.schedule(
            AddTask,  # Typed API: pass class instead of "add-task"
            {"a": input.a, "b": input.b},
        )

        return TypedTaskOutput(result=result["sum"])
