"""E2E tests for child workflow functionality."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_child_workflow_success(env: FlovynTestEnvironment) -> None:
    """Test successful child workflow execution.

    Flow:
    1. Parent workflow calls child workflow
    2. Child workflow completes successfully
    3. Parent workflow receives child result
    """
    handle = await env.start_workflow(
        "child-workflow-workflow",
        {"child_input": "hello from parent"},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Child result should contain the echo message
    assert "hello from parent" in str(result["child_result"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_child_workflow_failure(env: FlovynTestEnvironment) -> None:
    """Test child workflow failure handling.

    Flow:
    1. Parent workflow calls child workflow that fails
    2. Parent workflow catches the ChildWorkflowFailed exception
    3. Parent workflow handles the error gracefully
    """
    handle = await env.start_workflow(
        "child-failure-workflow",
        {"error_message": "intentional child failure"},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # The parent should have caught the child failure
    assert result["caught_error"] != ""
    # Error message should contain the original error
    assert (
        "intentional child failure" in result["caught_error"].lower()
        or "child" in result["caught_error"].lower()
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_nested_child_workflows(env: FlovynTestEnvironment) -> None:
    """Test multi-level nested child workflows.

    Flow:
    1. Parent workflow calls child workflow
    2. Child workflow calls grandchild workflow
    3. All levels complete and return results
    """
    handle = await env.start_workflow(
        "nested-child-workflow",
        {"depth": 3, "value": "nested"},
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=60))

    # Result should show all nesting levels
    assert "leaf:nested" in result["result"]
    assert result["levels"] == 3
