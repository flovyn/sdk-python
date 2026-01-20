"""E2E tests for workflow functionality."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_echo_workflow(env: FlovynTestEnvironment) -> None:
    """Test basic workflow execution with echo workflow."""
    handle = await env.start_workflow(
        "echo-workflow",
        {"message": "Hello, World!"},
    )

    result = await env.await_completion(handle)

    assert result["message"] == "Hello, World!"
    assert result["timestamp"] is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_doubler_workflow(env: FlovynTestEnvironment) -> None:
    """Test workflow with computation."""
    handle = await env.start_workflow(
        "doubler-workflow",
        {"value": 21},
    )

    result = await env.await_completion(handle)

    assert result["result"] == 42


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_failing_workflow(env: FlovynTestEnvironment) -> None:
    """Test workflow error handling."""
    handle = await env.start_workflow(
        "failing-workflow",
        {"error_message": "Intentional test failure"},
    )

    with pytest.raises(Exception) as exc_info:
        await env.await_completion(handle)

    assert "Intentional test failure" in str(exc_info.value)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateful_workflow(env: FlovynTestEnvironment) -> None:
    """Test workflow state operations."""
    handle = await env.start_workflow(
        "stateful-workflow",
        {"key": "test-key", "value": "test-value"},
    )

    result = await env.await_completion(handle)

    assert result["stored_value"] == "test-value"
    assert "test-key" in result["all_keys"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_operation_workflow(env: FlovynTestEnvironment) -> None:
    """Test ctx.run() for durable side effects."""
    handle = await env.start_workflow(
        "run-operation-workflow",
        {"operation_name": "my-operation"},
    )

    result = await env.await_completion(handle)

    assert result["result"] == "executed-my-operation"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_random_workflow(env: FlovynTestEnvironment) -> None:
    """Test deterministic random generation."""
    handle = await env.start_workflow(
        "random-workflow",
        {},
    )

    result = await env.await_completion(handle)

    # Verify we got values (determinism is verified by successful replay)
    assert result["uuid"] is not None
    assert len(result["uuid"]) == 36  # UUID format
    assert 0 <= result["random_float"] < 1.0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sleep_workflow(env: FlovynTestEnvironment) -> None:
    """Test durable timers."""
    handle = await env.start_workflow(
        "sleep-workflow",
        {"duration_ms": 100},  # 100ms sleep
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["slept_duration_ms"] == 100
    assert result["start_time"] is not None
    assert result["end_time"] is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multiple_workflows_parallel(env: FlovynTestEnvironment) -> None:
    """Test multiple workflows executing in parallel."""
    # Start 5 workflows concurrently
    handles = []
    for i in range(5):
        handle = await env.start_workflow(
            "doubler-workflow",
            {"value": i},
        )
        handles.append((i, handle))

    # Await all results
    for expected_input, handle in handles:
        result = await env.await_completion(handle)
        assert result["result"] == expected_input * 2
