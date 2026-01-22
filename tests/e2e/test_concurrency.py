"""E2E tests for concurrent execution patterns."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_workflow_execution(env: FlovynTestEnvironment) -> None:
    """Test multiple workflows executing concurrently.

    This tests that the worker can handle multiple workflows being:
    1. Started concurrently
    2. Executed in parallel
    3. Completed independently with correct results
    """
    num_workflows = 5

    # Start multiple workflows concurrently
    handles = []
    for i in range(num_workflows):
        handle = await env.start_workflow(
            "doubler-workflow",
            {"value": i * 10},
        )
        handles.append((i * 10, handle))

    # Wait for all workflows to complete
    results = []
    for input_value, handle in handles:
        result = await env.await_completion(handle, timeout=timedelta(seconds=30))
        results.append((input_value, result["result"]))

    # Verify all results are correct
    assert len(results) == num_workflows
    for input_value, output_value in results:
        expected = input_value * 2
        assert output_value == expected, f"Expected {expected}, got {output_value}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_task_execution(env: FlovynTestEnvironment) -> None:
    """Test multiple tasks executing concurrently within workflows.

    This tests that tasks are executed in parallel correctly when
    scheduled at the same time.
    """
    # Schedule 3 workflows, each with parallel tasks
    handles = []
    for i in range(3):
        items = [f"item-{i}-{j}" for j in range(4)]
        handle = await env.start_workflow(
            "fan-out-fan-in-workflow",
            {"items": items},
        )
        handles.append((i, items, handle))

    # Wait for all workflows
    for _, items, handle in handles:
        result = await env.await_completion(handle, timeout=timedelta(seconds=30))
        assert result["input_count"] == len(items)
        assert result["output_count"] == len(items)
        assert set(result["processed_items"]) == set(items)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_high_throughput_small_workflows(env: FlovynTestEnvironment) -> None:
    """Test high throughput with many small workflows.

    Starts many simple workflows quickly to test throughput.
    """
    num_workflows = 20

    # Start all workflows as fast as possible
    handles = []
    for i in range(num_workflows):
        handle = await env.start_workflow(
            "echo-workflow",
            {"message": f"msg-{i}"},
        )
        handles.append((i, handle))

    # Wait for all to complete
    completed_count = 0
    for i, handle in handles:
        result = await env.await_completion(handle, timeout=timedelta(seconds=60))
        assert result["message"] == f"msg-{i}"
        completed_count += 1

    assert completed_count == num_workflows


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mixed_workflow_types_concurrent(env: FlovynTestEnvironment) -> None:
    """Test concurrent execution of different workflow types.

    Verifies that different workflow types can run concurrently
    without interfering with each other.
    """
    handles = []

    # Echo workflows
    for i in range(3):
        handle = await env.start_workflow(
            "echo-workflow",
            {"message": f"echo-{i}"},
        )
        handles.append(("echo", i, handle))

    # Doubler workflows
    for i in range(3):
        handle = await env.start_workflow(
            "doubler-workflow",
            {"value": i * 5},
        )
        handles.append(("doubler", i, handle))

    # Sleep workflows (short sleeps)
    for i in range(2):
        handle = await env.start_workflow(
            "sleep-workflow",
            {"duration_ms": 50},
        )
        handles.append(("sleep", i, handle))

    # Wait for all and verify
    for workflow_type, i, handle in handles:
        if workflow_type == "echo":
            result = await env.await_completion(handle, timeout=timedelta(seconds=30))
            assert result["message"] == f"echo-{i}"
        elif workflow_type == "doubler":
            result = await env.await_completion(handle, timeout=timedelta(seconds=30))
            assert result["result"] == i * 5 * 2
        elif workflow_type == "sleep":
            result = await env.await_completion(handle, timeout=timedelta(seconds=30))
            assert result["slept_duration_ms"] == 50
