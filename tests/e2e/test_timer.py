"""E2E tests for timer/sleep functionality."""

import time
from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_short_timer(env: FlovynTestEnvironment) -> None:
    """Test timer with very short duration (100ms).

    This tests that even short timers work correctly through suspend/resume.
    """
    start_time = time.time()

    handle = await env.start_workflow(
        "sleep-workflow",
        {"duration_ms": 100},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))
    wall_elapsed_ms = (time.time() - start_time) * 1000

    assert result["slept_duration_ms"] == 100

    # Wall-clock time should be at least 100ms
    assert wall_elapsed_ms >= 100, f"Expected >= 100ms, got {wall_elapsed_ms}ms"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_durable_timer_sleep(env: FlovynTestEnvironment) -> None:
    """Test durable timer with longer duration.

    The workflow sleeps for 1 second and returns timing information.
    """
    start_time = time.time()

    handle = await env.start_workflow(
        "sleep-workflow",
        {"duration_ms": 1000},  # 1 second
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))
    wall_elapsed_ms = (time.time() - start_time) * 1000

    assert result["slept_duration_ms"] == 1000

    # Wall-clock time should be at least 1000ms
    assert wall_elapsed_ms >= 1000, f"Expected >= 1000ms, got {wall_elapsed_ms}ms"
    # And not too much more (allow 5 seconds for overhead)
    assert wall_elapsed_ms < 6000, f"Took too long: {wall_elapsed_ms}ms"
