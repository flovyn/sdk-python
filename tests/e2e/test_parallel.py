"""E2E tests for parallel execution patterns."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fan_out_fan_in(env: FlovynTestEnvironment) -> None:
    """Test fan-out/fan-in pattern with parallel tasks.

    Flow:
    1. Schedule multiple tasks in parallel (fan-out)
    2. Collect all results (fan-in)
    3. Aggregate results
    """
    items = ["apple", "banana", "cherry", "date"]

    handle = await env.start_workflow(
        "fan-out-fan-in-workflow",
        {"items": items},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["input_count"] == 4
    assert result["output_count"] == 4
    # All items should be echoed back
    assert set(result["processed_items"]) == set(items)
    # Total length should be sum of all item lengths
    assert result["total_length"] == sum(len(item) for item in items)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parallel_large_batch(env: FlovynTestEnvironment) -> None:
    """Test parallel execution with many tasks.

    Schedules 20 tasks in parallel and verifies all complete correctly.
    """
    handle = await env.start_workflow(
        "large-batch-workflow",
        {"count": 20},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=60))

    assert result["task_count"] == 20
    # Each task computes i + 1 for i in range(20)
    # So results are [1, 2, 3, ..., 20]
    # Total = sum(1..20) = 20 * 21 / 2 = 210
    assert result["total"] == 210
    assert result["min_value"] == 1
    assert result["max_value"] == 20


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parallel_empty_batch(env: FlovynTestEnvironment) -> None:
    """Test handling of empty parallel batch.

    Verifies workflow handles zero items gracefully.
    """
    handle = await env.start_workflow(
        "fan-out-fan-in-workflow",
        {"items": []},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["input_count"] == 0
    assert result["output_count"] == 0
    assert result["processed_items"] == []
    assert result["total_length"] == 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parallel_single_item(env: FlovynTestEnvironment) -> None:
    """Test parallel pattern with single item.

    Verifies edge case of batch size 1.
    """
    handle = await env.start_workflow(
        "fan-out-fan-in-workflow",
        {"items": ["only-one"]},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["input_count"] == 1
    assert result["output_count"] == 1
    assert result["processed_items"] == ["only-one"]
    assert result["total_length"] == 8  # len("only-one")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parallel_tasks_join_all(env: FlovynTestEnvironment) -> None:
    """Test basic parallel task scheduling with join_all pattern.

    Schedules multiple tasks and awaits all results.
    """
    items = ["a", "b", "c"]

    handle = await env.start_workflow(
        "fan-out-fan-in-workflow",
        {"items": items},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["input_count"] == 3
    assert result["output_count"] == 3
    # Verify all items were processed
    assert set(result["processed_items"]) == set(items)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mixed_parallel_operations(env: FlovynTestEnvironment) -> None:
    """Test mixed parallel operations with tasks and timers.

    This workflow:
    1. Phase 1: Two parallel echo tasks
    2. Timer: Wait for 100ms
    3. Phase 3: Three parallel add tasks
    """
    handle = await env.start_workflow(
        "mixed-parallel-workflow",
        {},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=60))

    assert result["success"] is True
    assert len(result["phase1_results"]) == 2
    assert result["timer_fired"] is True
    assert len(result["phase3_results"]) == 3
    # Phase 3 computes i + i for i in [0, 1, 2] = [0, 2, 4]
    assert result["phase3_results"] == [0, 2, 4]
