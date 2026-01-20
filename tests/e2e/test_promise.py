"""E2E tests for promise functionality."""

import asyncio
from datetime import timedelta

import pytest

from flovyn.exceptions import PromiseRejected
from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_promise_resolve(env: FlovynTestEnvironment) -> None:
    """Test external promise resolution.

    Flow:
    1. Start workflow that waits for a promise
    2. Workflow suspends waiting for promise
    3. Resolve the promise externally
    4. Workflow resumes and completes with the promise value
    """
    # Start the workflow (it will suspend waiting for the promise)
    handle = await env.start_workflow(
        "await-promise-workflow",
        {"promise_name": "approval", "timeout_ms": 30000},
    )

    # Wait for workflow to suspend and create the promise
    await asyncio.sleep(2.0)

    # Resolve the promise externally
    await env.resolve_promise(
        handle,
        "approval",
        {"approved": True, "approver": "admin@example.com"},
    )

    # Wait for workflow to complete
    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify the promise value was received
    assert result["resolved_value"] == {"approved": True, "approver": "admin@example.com"}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_promise_reject(env: FlovynTestEnvironment) -> None:
    """Test external promise rejection.

    Flow:
    1. Start workflow that waits for a promise
    2. Workflow suspends waiting for promise
    3. Reject the promise externally
    4. Workflow receives PromiseRejected error
    """
    # Start the workflow (it will suspend waiting for the promise)
    handle = await env.start_workflow(
        "await-promise-workflow",
        {"promise_name": "approval", "timeout_ms": 30000},
    )

    # Wait for workflow to suspend and create the promise
    await asyncio.sleep(2.0)

    # Reject the promise externally
    await env.reject_promise(
        handle,
        "approval",
        "Request denied by admin",
    )

    # Workflow should fail with PromiseRejected
    with pytest.raises((PromiseRejected, Exception)) as exc_info:
        await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify error message contains the rejection reason
    error_msg = str(exc_info.value)
    assert "denied" in error_msg.lower() or "rejected" in error_msg.lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_promise_timeout(env: FlovynTestEnvironment) -> None:
    """Test promise with timeout.

    Flow:
    1. Start workflow that waits for a promise with short timeout
    2. Don't resolve the promise
    3. Workflow should timeout
    """
    # Start the workflow with a short timeout (2 seconds)
    handle = await env.start_workflow(
        "await-promise-workflow",
        {"promise_name": "approval", "timeout_ms": 2000},
    )

    # Don't resolve the promise - let it timeout
    # The workflow should fail due to promise timeout
    with pytest.raises(Exception) as exc_info:
        await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify it's a timeout-related error
    error_msg = str(exc_info.value).lower()
    assert "timeout" in error_msg or "timed out" in error_msg or "failed" in error_msg
