"""E2E tests for task functionality."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_workflow_scheduling_tasks(env: FlovynTestEnvironment) -> None:
    """Test workflow that schedules multiple tasks sequentially."""
    handle = await env.start_workflow(
        "task-scheduling-workflow",
        {"count": 5},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Results should be cumulative sums: 1, 3, 6, 10, 15
    assert result["results"] == [1, 3, 6, 10, 15]
    assert result["total"] == 15


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_workflow_many_tasks(env: FlovynTestEnvironment) -> None:
    """Test workflow with many sequential tasks."""
    handle = await env.start_workflow(
        "multi-task-workflow",
        {"count": 10},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=60))

    # Each task adds i + i, so results are [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    expected_results = [i * 2 for i in range(10)]
    assert result["results"] == expected_results
    assert result["total"] == sum(expected_results)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_workflow_parallel_tasks(env: FlovynTestEnvironment) -> None:
    """Test workflow that executes tasks in parallel."""
    handle = await env.start_workflow(
        "parallel-tasks-workflow",
        {"count": 5},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Each task adds i + i, so results are [0, 2, 4, 6, 8]
    expected_results = [i * 2 for i in range(5)]
    assert sorted(result["results"]) == sorted(expected_results)  # Order may vary
    assert result["total"] == sum(expected_results)
