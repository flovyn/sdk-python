"""E2E tests for worker lifecycle functionality."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_registration(env: FlovynTestEnvironment) -> None:
    """Test that worker registers successfully with the server."""
    # The environment fixture starts the worker, so if we get here
    # without errors, registration succeeded
    assert env._started is True


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_processes_multiple_workflows(env: FlovynTestEnvironment) -> None:
    """Test that worker can process multiple workflows."""
    # Start multiple workflows
    handles = []
    for i in range(3):
        handle = await env.start_workflow(
            "doubler-workflow",
            {"value": i + 1},
        )
        handles.append(handle)

    # All should complete successfully
    for i, handle in enumerate(handles):
        result = await env.await_completion(handle)
        assert result["result"] == (i + 1) * 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_status_running(env: FlovynTestEnvironment) -> None:
    """Test that worker status is 'running' after start.

    Verifies:
    - Worker status API is accessible
    - Status shows 'running' after successful start
    """
    # Worker should be running after env.start()
    status = env.worker_status
    assert status == "running", f"Expected 'running', got '{status}'"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_continues_after_workflow(env: FlovynTestEnvironment) -> None:
    """Test that worker stays running after processing a workflow.

    Verifies the worker doesn't exit after completing work.
    """
    # Verify running before
    assert env.worker_status == "running"

    # Process a workflow
    handle = await env.start_workflow(
        "echo-workflow",
        {"message": "test"},
    )
    result = await env.await_completion(handle, timeout=timedelta(seconds=30))
    assert result["message"] == "test"

    # Verify still running after
    assert env.worker_status == "running"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_handles_workflow_errors(env: FlovynTestEnvironment) -> None:
    """Test that worker continues running after a workflow failure.

    Verifies the worker is resilient to individual workflow failures.
    """
    # Start a failing workflow
    handle = await env.start_workflow(
        "failing-workflow",
        {"error_message": "Expected failure"},
    )

    with pytest.raises(Exception):  # noqa: B017 - intentionally catching any exception
        await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Worker should still be running
    assert env.worker_status == "running"

    # Should be able to process more workflows
    handle2 = await env.start_workflow(
        "echo-workflow",
        {"message": "after-failure"},
    )
    result = await env.await_completion(handle2, timeout=timedelta(seconds=30))
    assert result["message"] == "after-failure"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_uptime(env: FlovynTestEnvironment) -> None:
    """Test that worker uptime API works correctly.

    Verifies:
    - Uptime is available after worker starts
    - Uptime increases over time
    """
    import asyncio

    # Get initial uptime
    uptime1 = env.worker_uptime_ms
    assert uptime1 is not None, "Uptime should be available"
    assert uptime1 >= 0, f"Uptime should be non-negative, got {uptime1}"

    # Wait a bit
    await asyncio.sleep(0.1)

    # Get uptime again
    uptime2 = env.worker_uptime_ms
    assert uptime2 is not None, "Uptime should still be available"
    assert uptime2 > uptime1, f"Uptime should increase: {uptime2} > {uptime1}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_metrics(env: FlovynTestEnvironment) -> None:
    """Test that worker metrics API works correctly.

    Verifies:
    - Metrics are available after worker starts
    - Metrics contain expected fields
    """
    metrics = env.get_worker_metrics()
    assert metrics is not None, "Metrics should be available"

    # Check metrics fields
    assert hasattr(metrics, "uptime_ms"), "Metrics should have uptime_ms"
    assert hasattr(metrics, "status"), "Metrics should have status"

    assert metrics.uptime_ms >= 0, f"Uptime should be non-negative: {metrics.uptime_ms}"
    assert metrics.status == "running", f"Status should be 'running', got '{metrics.status}'"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_started_at(env: FlovynTestEnvironment) -> None:
    """Test that worker start time is recorded correctly.

    Verifies:
    - Start time is available
    - Start time is a reasonable timestamp
    """
    import time

    started_at = env.worker_started_at_ms
    assert started_at is not None, "Started at should be available"

    # Check that start time is in the past (but not too far)
    now_ms = int(time.time() * 1000)
    assert started_at > 0, f"Started at should be positive: {started_at}"
    assert started_at <= now_ms, f"Started at should be in the past: {started_at} <= {now_ms}"

    # Should have started within the last hour (reasonable for tests)
    one_hour_ago = now_ms - (60 * 60 * 1000)
    assert started_at > one_hour_ago, f"Started at should be recent: {started_at} > {one_hour_ago}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_registration_info(env: FlovynTestEnvironment) -> None:
    """Test that registration info API works correctly.

    Verifies:
    - Registration info is available after worker starts
    - Contains expected fields
    """
    reg_info = env.get_registration_info()
    assert reg_info is not None, "Registration info should be available"

    # Check fields
    assert hasattr(reg_info, "worker_id"), "Should have worker_id"
    assert hasattr(reg_info, "success"), "Should have success"
    assert hasattr(reg_info, "workflow_kinds"), "Should have workflow_kinds"
    assert hasattr(reg_info, "task_kinds"), "Should have task_kinds"

    assert reg_info.success is True, "Registration should be successful"
    assert reg_info.worker_id, "Worker ID should be set"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_connection_info(env: FlovynTestEnvironment) -> None:
    """Test that connection info API works correctly.

    Verifies:
    - Connection info is available after worker starts
    - Shows connected state
    """
    conn_info = env.get_connection_info()
    assert conn_info is not None, "Connection info should be available"

    # Check fields
    assert hasattr(conn_info, "connected"), "Should have connected"
    assert hasattr(conn_info, "heartbeat_failures"), "Should have heartbeat_failures"
    assert hasattr(conn_info, "poll_failures"), "Should have poll_failures"

    assert conn_info.connected is True, "Should be connected"
    assert conn_info.heartbeat_failures == 0, "Should have no heartbeat failures"
    assert conn_info.poll_failures == 0, "Should have no poll failures"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_lifecycle_events(env: FlovynTestEnvironment) -> None:
    """Test that lifecycle events are emitted and can be polled.

    Verifies:
    - Lifecycle events are available after worker starts
    - Events include expected event types (starting, registered, ready)
    """
    # Poll for events - there should be at least starting/registered/ready events
    events = env.poll_lifecycle_events()

    # Convert to list of event names
    event_names = [e.event_name for e in events]

    # Should have at least some events
    assert len(event_names) > 0, "Should have received at least one event"

    # Check that we received expected event types
    has_expected = any(name in ["starting", "registered", "ready"] for name in event_names)
    assert has_expected, f"Should have starting, registered, or ready event. Got: {event_names}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pause_resume(env: FlovynTestEnvironment) -> None:
    """Test pause and resume functionality.

    Verifies:
    - Worker can be paused
    - Status reflects paused state
    - Worker can be resumed
    - Status reflects running state after resume
    """
    # Verify worker is running
    assert env.is_running, "Worker should be running initially"
    assert not env.is_paused, "Worker should not be paused initially"

    # Pause the worker
    env.pause("test pause reason")

    # Verify paused state
    assert env.is_paused, "Worker should be paused after pause()"
    assert not env.is_running, "Worker should not be running while paused"
    assert env.worker_status == "paused", f"Expected 'paused', got '{env.worker_status}'"
    assert env.get_pause_reason() == "test pause reason", "Pause reason should be preserved"

    # Resume the worker
    env.resume()

    # Verify running state
    assert env.is_running, "Worker should be running after resume()"
    assert not env.is_paused, "Worker should not be paused after resume()"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pause_invalid_state(env: FlovynTestEnvironment) -> None:
    """Test that pause fails when worker is not in Running state.

    Verifies:
    - Pause fails when already paused
    """
    # Pause first time - should succeed
    env.pause("first pause")

    # Try to pause again while already paused - should fail
    try:
        env.pause("second pause")
        pytest.fail("Pause while already paused should fail")
    except Exception as e:
        assert "already paused" in str(e).lower(), f"Expected 'already paused' error, got: {e}"

    # Resume to clean up
    env.resume()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_resume_invalid_state(env: FlovynTestEnvironment) -> None:
    """Test that resume fails when worker is not in Paused state.

    Verifies:
    - Resume fails when not paused
    """
    # Try to resume when not paused - should fail
    try:
        env.resume()
        pytest.fail("Resume while not paused should fail")
    except Exception as e:
        assert "not paused" in str(e).lower(), f"Expected 'not paused' error, got: {e}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_client_config_accessors(env: FlovynTestEnvironment) -> None:
    """Test configuration accessor methods.

    Verifies:
    - Max concurrent settings are accessible
    - Values are positive integers
    """
    # Verify configuration accessors
    max_workflows = env.max_concurrent_workflows
    max_tasks = env.max_concurrent_tasks

    assert max_workflows > 0, "Max concurrent workflows should be positive"
    assert max_tasks > 0, "Max concurrent tasks should be positive"

    # Check default values (100)
    assert max_workflows == 100, f"Default max workflows should be 100, got {max_workflows}"
    assert max_tasks == 100, f"Default max tasks should be 100, got {max_tasks}"
