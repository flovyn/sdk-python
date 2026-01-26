"""E2E tests for replay and determinism functionality."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mixed_commands_workflow(env: FlovynTestEnvironment) -> None:
    """Test workflow with mixed command types (operations, timers, tasks).

    This validates per-type sequence matching during replay:
    1. Operation (ctx.run)
    2. Timer (ctx.sleep)
    3. Task (ctx.schedule)
    4. Another operation

    The workflow will be replayed multiple times as each async operation
    completes, and must produce consistent results.
    """
    handle = await env.start_workflow(
        "mixed-commands-workflow",
        {"value": 42},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify all steps completed correctly
    assert result["operation_result"] == "computed-42"
    assert result["sleep_completed"] is True
    assert result["task_result"] == 52  # 42 + 10
    assert result["final_value"] == 84  # 42 * 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sequential_tasks_in_loop(env: FlovynTestEnvironment) -> None:
    """Test that tasks scheduled in a loop replay correctly.

    This is an implicit replay test - the workflow schedules multiple
    tasks sequentially. Each time a task completes, the workflow is
    replayed and must produce the same task schedule sequence.
    """
    handle = await env.start_workflow(
        "task-scheduling-workflow",
        {"count": 5},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Each task adds 1, 2, 3, 4, 5 to running total
    # Results: [1, 3, 6, 10, 15]
    assert result["results"] == [1, 3, 6, 10, 15]
    assert result["total"] == 15


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parallel_tasks_replay(env: FlovynTestEnvironment) -> None:
    """Test that parallel tasks scheduled together replay correctly.

    When multiple tasks are scheduled in parallel, replay must
    correctly match each task to its result event.
    """
    handle = await env.start_workflow(
        "parallel-tasks-workflow",
        {"count": 5},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Each task computes i + i: [0, 2, 4, 6, 8]
    assert result["results"] == [0, 2, 4, 6, 8]
    assert result["total"] == 20


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sleep_replay(env: FlovynTestEnvironment) -> None:
    """Test that timer events replay correctly.

    A workflow with sleep should replay correctly, using the
    stored timer duration from the event history.
    """
    handle = await env.start_workflow(
        "sleep-workflow",
        {"duration_ms": 100},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["slept_duration_ms"] == 100
    # Verify timestamps are present
    assert result["start_time"] is not None
    assert result["end_time"] is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_child_workflow_loop_replay(env: FlovynTestEnvironment) -> None:
    """Test that child workflows scheduled in a loop replay correctly.

    This tests per-type sequence matching for child workflows during replay.
    Each iteration schedules a child workflow, and on replay, the correct
    child workflow result must be matched to each iteration.
    """
    handle = await env.start_workflow(
        "child-loop-workflow",
        {"count": 3},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=60))

    assert result["total_count"] == 3
    assert result["results"] == ["child-0", "child-1", "child-2"]
