"""E2E tests for error handling."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_error_message_preserved(env: FlovynTestEnvironment) -> None:
    """Test that specific error messages are preserved in workflow failures.

    This verifies that:
    1. Custom error messages are preserved through the failure
    2. Unique identifiers in error messages can be retrieved
    """
    specific_error = "Custom error message with specific details XYZ-123"

    handle = await env.start_workflow(
        "failing-workflow",
        {"error_message": specific_error},
    )

    with pytest.raises(Exception) as exc_info:
        await env.await_completion(handle, timeout=timedelta(seconds=30))

    # Verify the specific error identifier is preserved
    error_str = str(exc_info.value)
    assert "XYZ-123" in error_str, f"Expected 'XYZ-123' in error, got: {error_str}"
