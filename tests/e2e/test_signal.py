"""E2E tests for signal functionality."""

import asyncio
from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_signal_workflow(env: FlovynTestEnvironment) -> None:
    """Test external signal delivery.

    Flow:
    1. Start workflow that waits for a signal
    2. Workflow suspends waiting for signal
    3. Send signal externally
    4. Workflow resumes and completes with the signal value
    """
    # Start the workflow (it will suspend waiting for the signal)
    handle = await env.start_workflow(
        "signal-workflow",
        {},
    )

    # Wait for workflow to suspend and start waiting for signals
    await asyncio.sleep(2.0)

    # Send signal externally
    await env.signal_workflow(
        handle,
        "user-action",
        {"action": "approve", "user": "admin@example.com"},
    )

    # Wait for workflow to complete
    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify the signal value was received
    assert result["signal_count"] == 1
    assert len(result["received_signals"]) == 1
    assert result["received_signals"][0] == {"action": "approve", "user": "admin@example.com"}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multiple_signals(env: FlovynTestEnvironment) -> None:
    """Test multiple signals are delivered in order.

    Flow:
    1. Start workflow that waits for 3 signals
    2. Send 3 signals
    3. Workflow should receive them in order
    """
    # Start the workflow
    handle = await env.start_workflow(
        "multi-signal-workflow",
        {"expected_count": 3},
    )

    # Wait for workflow to start
    await asyncio.sleep(2.0)

    # Send 3 signals
    await env.signal_workflow(handle, "signal-1", {"order": 1})
    await env.signal_workflow(handle, "signal-2", {"order": 2})
    await env.signal_workflow(handle, "signal-3", {"order": 3})

    # Wait for workflow to complete
    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify signals were received in order
    assert result["signal_count"] == 3
    assert len(result["received_signals"]) == 3
    assert result["received_signals"][0] == {"order": 1}
    assert result["received_signals"][1] == {"order": 2}
    assert result["received_signals"][2] == {"order": 3}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_signal_with_start_new_workflow(env: FlovynTestEnvironment) -> None:
    """Test signal_with_start creates a new workflow with signal.

    Flow:
    1. Call signal_with_start on a workflow ID that doesn't exist
    2. Workflow should be created and receive the signal
    """
    import uuid

    workflow_id = f"signal-with-start-{uuid.uuid4().hex[:8]}"

    # Use signal_with_start to create workflow with signal
    handle = await env.signal_with_start_workflow(
        "signal-workflow",
        workflow_id,
        {},  # Empty input
        "init-signal",
        {"data": "from signal_with_start"},
    )

    # Wait for workflow to complete
    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify signal was received
    assert result["signal_count"] == 1
    assert result["received_signals"][0] == {"data": "from signal_with_start"}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_signal_with_start_existing_workflow(env: FlovynTestEnvironment) -> None:
    """Test signal_with_start on existing workflow only sends signal.

    Flow:
    1. Start a workflow waiting for 2 signals
    2. Call signal_with_start with same workflow ID
    3. Workflow should receive only the signal (not be recreated)
    4. Send second signal to complete
    """
    import uuid

    workflow_id = f"signal-with-start-existing-{uuid.uuid4().hex[:8]}"

    # First, start workflow waiting for 2 signals
    handle = await env.start_workflow(
        "multi-signal-workflow",
        {"expected_count": 2},
        workflow_id=workflow_id,
    )

    # Wait for workflow to start
    await asyncio.sleep(2.0)

    # Use signal_with_start on the same workflow ID - should only send signal
    await env.signal_with_start_workflow(
        "multi-signal-workflow",
        workflow_id,
        {"expected_count": 2},  # Would be different if workflow was recreated
        "signal-via-sws",
        {"source": "signal_with_start"},
    )

    # Send second signal to complete
    await env.signal_workflow(handle, "second-signal", {"source": "direct"})

    # Wait for workflow to complete
    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify both signals were received
    assert result["signal_count"] == 2
    assert {"source": "signal_with_start"} in result["received_signals"]
    assert {"source": "direct"} in result["received_signals"]
