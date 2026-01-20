"""E2E tests for comprehensive workflow functionality."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_comprehensive_workflow_features(env: FlovynTestEnvironment) -> None:
    """Test workflow that combines multiple features together.

    This comprehensive test verifies (matching Rust SDK):
    - Basic workflow execution
    - Input/output handling
    - Operation recording (ctx.run)
    - State set/get operations
    - Multiple operations in sequence
    """
    handle = await env.start_workflow(
        "comprehensive-workflow",
        {"value": 21},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Validate all features tested by the comprehensive workflow
    assert result["input_value"] == 21, "Basic input should work"
    assert result["run_result"] == 42, "ctx.run() should record operation"
    assert result["state_set"] is True, "State set should succeed"
    assert result["state_matches"] is True, "State get should return what was set"
    assert result["triple_result"] == 63, "Multiple operations should work"
    assert result["tests_passed_count"] == 5, "All 5 feature tests should pass"

    # Verify specific state content
    assert result["state_retrieved"] is not None
    assert result["state_retrieved"]["counter"] == 21
    assert result["state_retrieved"]["message"] == "state test"
    assert result["state_retrieved"]["nested"]["a"] == 1
    assert result["state_retrieved"]["nested"]["b"] == 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_comprehensive_with_different_input(env: FlovynTestEnvironment) -> None:
    """Test comprehensive workflow with different input value.

    Validates the same features with a different input to ensure determinism.
    """
    handle = await env.start_workflow(
        "comprehensive-workflow",
        {"value": 50},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Validate all features
    assert result["input_value"] == 50, "Basic input should work"
    assert result["run_result"] == 100, "ctx.run() should double value (50*2=100)"
    assert result["state_set"] is True, "State set should succeed"
    assert result["state_matches"] is True, "State get should return what was set"
    assert result["triple_result"] == 150, "Triple operation should work (50*3=150)"
    assert result["tests_passed_count"] == 5, "All 5 feature tests should pass"

    # Verify state content
    assert result["state_retrieved"] is not None
    assert result["state_retrieved"]["counter"] == 50


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_all_basic_workflows(env: FlovynTestEnvironment) -> None:
    """Test all basic workflows execute correctly.

    Runs multiple different workflow types to ensure basic functionality.
    """
    # Echo workflow
    echo_handle = await env.start_workflow(
        "echo-workflow",
        {"message": "hello"},
    )
    echo_result = await env.await_completion(echo_handle, timeout=timedelta(seconds=30))
    assert echo_result["message"] == "hello"

    # Doubler workflow
    doubler_handle = await env.start_workflow(
        "doubler-workflow",
        {"value": 25},
    )
    doubler_result = await env.await_completion(
        doubler_handle, timeout=timedelta(seconds=30)
    )
    assert doubler_result["result"] == 50

    # Random workflow
    random_handle = await env.start_workflow(
        "random-workflow",
        {},
    )
    random_result = await env.await_completion(
        random_handle, timeout=timedelta(seconds=30)
    )
    assert random_result["uuid"] is not None
    assert 0 <= random_result["random_float"] < 1.0

    # Sleep workflow
    sleep_handle = await env.start_workflow(
        "sleep-workflow",
        {"duration_ms": 50},
    )
    sleep_result = await env.await_completion(
        sleep_handle, timeout=timedelta(seconds=30)
    )
    assert sleep_result["slept_duration_ms"] == 50

    # Stateful workflow
    stateful_handle = await env.start_workflow(
        "stateful-workflow",
        {"key": "my-key", "value": "my-value"},
    )
    stateful_result = await env.await_completion(
        stateful_handle, timeout=timedelta(seconds=30)
    )
    assert stateful_result["stored_value"] == "my-value"
