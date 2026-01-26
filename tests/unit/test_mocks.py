"""Unit tests for mock contexts."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import BaseModel

from flovyn import TaskContext, task
from flovyn.exceptions import TaskFailed, WorkflowCancelled
from flovyn.testing import MockTaskContext, MockWorkflowContext, TimeController


class AddInput(BaseModel):
    a: int
    b: int


class AddOutput(BaseModel):
    result: int


@task(name="add-mock")
class AddTask:
    async def run(self, ctx: TaskContext, input: AddInput) -> AddOutput:
        return AddOutput(result=input.a + input.b)


class TestMockTaskContext:
    @pytest.mark.asyncio
    async def test_progress_reporting(self) -> None:
        ctx = MockTaskContext()

        await ctx.report_progress(0.0, "Starting")
        await ctx.report_progress(0.5, "Halfway")
        await ctx.report_progress(1.0, "Done")

        assert len(ctx.progress_reports) == 3
        assert ctx.progress_reports[0].progress == 0.0
        assert ctx.progress_reports[1].progress == 0.5
        assert ctx.progress_reports[2].progress == 1.0

    @pytest.mark.asyncio
    async def test_invalid_progress_raises(self) -> None:
        ctx = MockTaskContext()

        with pytest.raises(ValueError):
            await ctx.report_progress(-0.1)

        with pytest.raises(ValueError):
            await ctx.report_progress(1.1)

    @pytest.mark.asyncio
    async def test_heartbeat_counting(self) -> None:
        ctx = MockTaskContext()

        await ctx.heartbeat()
        await ctx.heartbeat()
        await ctx.heartbeat()

        assert ctx.heartbeat_count == 3

    def test_cancellation(self) -> None:
        ctx = MockTaskContext(is_cancelled=False)
        assert not ctx.is_cancelled

        ctx.set_cancelled(True)
        assert ctx.is_cancelled

        error = ctx.cancellation_error()
        assert isinstance(error, Exception)


class TestMockWorkflowContext:
    def test_deterministic_time(self) -> None:
        start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        time_controller = TimeController(start_time=start_time)
        ctx = MockWorkflowContext(time_controller=time_controller)

        assert ctx.current_time() == start_time

    def test_deterministic_uuid(self) -> None:
        ctx = MockWorkflowContext()

        uuid1 = ctx.random_uuid()
        uuid2 = ctx.random_uuid()

        assert isinstance(uuid1, UUID)
        assert isinstance(uuid2, UUID)
        assert uuid1 != uuid2

    def test_deterministic_random(self) -> None:
        ctx = MockWorkflowContext()

        r1 = ctx.random()
        r2 = ctx.random()

        assert 0.0 <= r1 < 1.0
        assert 0.0 <= r2 < 1.0

    @pytest.mark.asyncio
    async def test_mock_task_result(self) -> None:
        ctx = MockWorkflowContext()
        ctx.mock_task_result(AddTask, AddOutput(result=42))

        result = await ctx.schedule(AddTask, AddInput(a=1, b=2))

        assert result.result == 42
        assert len(ctx.executed_tasks) == 1

    @pytest.mark.asyncio
    async def test_mock_task_with_callable(self) -> None:
        ctx = MockWorkflowContext()
        ctx.mock_task_result(
            AddTask,
            lambda input: AddOutput(result=input.a + input.b),
        )

        result = await ctx.schedule(AddTask, AddInput(a=5, b=7))

        assert result.result == 12

    @pytest.mark.asyncio
    async def test_mock_task_failure(self) -> None:
        ctx = MockWorkflowContext()
        ctx.mock_task_failure(AddTask, TaskFailed("Mock failure"))

        with pytest.raises(TaskFailed):
            await ctx.schedule(AddTask, AddInput(a=1, b=2))

    @pytest.mark.asyncio
    async def test_state_operations(self) -> None:
        ctx = MockWorkflowContext()

        await ctx.set("key1", "value1")
        await ctx.set("key2", 42)

        assert await ctx.get("key1") == "value1"
        assert await ctx.get("key2") == 42
        assert await ctx.get("missing") is None
        assert await ctx.get("missing", default="default") == "default"

        await ctx.clear("key1")
        assert await ctx.get("key1") is None

    @pytest.mark.asyncio
    async def test_cancellation_check(self) -> None:
        ctx = MockWorkflowContext()

        # Should not raise when not cancelled
        ctx.check_cancellation()

        # Should raise when cancelled
        ctx.request_cancellation()
        with pytest.raises(WorkflowCancelled):
            ctx.check_cancellation()

    @pytest.mark.asyncio
    async def test_promise_mock(self) -> None:
        ctx = MockWorkflowContext()
        ctx.mock_promise_value("my-promise", {"data": "resolved"})

        result = await ctx.promise("my-promise")
        assert result == {"data": "resolved"}


class TestTimeController:
    def test_initial_time(self) -> None:
        start = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        controller = TimeController(start_time=start)

        assert controller.current_time == start

    @pytest.mark.asyncio
    async def test_advance_time(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        controller = TimeController(start_time=start)

        await controller.advance(timedelta(hours=1))

        expected = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)
        assert controller.current_time == expected

    def test_current_time_millis(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        controller = TimeController(start_time=start)

        millis = controller.current_time_millis()
        assert isinstance(millis, int)
        assert millis > 0
