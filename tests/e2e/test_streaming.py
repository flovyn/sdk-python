"""E2E tests for task streaming functionality."""

from datetime import timedelta

import pytest

from flovyn.testing import FlovynTestEnvironment
from tests.e2e.fixtures.tasks import (
    StreamingAllTypesInput,
    StreamingAllTypesOutput,
    StreamingDataInput,
    StreamingDataOutput,
    StreamingErrorInput,
    StreamingErrorOutput,
    StreamingProgressInput,
    StreamingProgressOutput,
    StreamingTokenInput,
    StreamingTokenOutput,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_task_streams_tokens(env: FlovynTestEnvironment) -> None:
    """Test that a task can stream tokens to connected clients.

    Verifies:
    - Task can call stream_token()
    - Task completes successfully after streaming
    - Correct number of tokens were processed
    """
    tokens = ["Hello", " ", "world", "!"]

    # Use a workflow that schedules the streaming task
    handle = await env.start_workflow(
        "task-scheduler-workflow",
        {
            "task_name": "streaming-token-task",
            "task_input": StreamingTokenInput(tokens=tokens).model_dump(),
        },
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    # The task should have completed successfully
    assert result["task_completed"] is True
    # Verify token count from the task result
    task_result = StreamingTokenOutput.model_validate(result["task_result"])
    assert task_result.token_count == len(tokens)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_task_streams_progress(env: FlovynTestEnvironment) -> None:
    """Test that a task can stream progress updates.

    Verifies:
    - Task can call stream_progress()
    - Progress values are valid (0.0 to 1.0)
    - Task completes after streaming all progress
    """
    steps = 5

    handle = await env.start_workflow(
        "task-scheduler-workflow",
        {
            "task_name": "streaming-progress-task",
            "task_input": StreamingProgressInput(steps=steps).model_dump(),
        },
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["task_completed"] is True
    task_result = StreamingProgressOutput.model_validate(result["task_result"])
    assert task_result.final_progress == 1.0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_task_streams_data(env: FlovynTestEnvironment) -> None:
    """Test that a task can stream arbitrary data.

    Verifies:
    - Task can call stream_data()
    - Data is serialized correctly
    - Task completes after streaming all data
    """
    items = [
        {"id": 1, "name": "item1"},
        {"id": 2, "name": "item2"},
        {"id": 3, "name": "item3"},
    ]

    handle = await env.start_workflow(
        "task-scheduler-workflow",
        {
            "task_name": "streaming-data-task",
            "task_input": StreamingDataInput(items=items).model_dump(),
        },
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["task_completed"] is True
    task_result = StreamingDataOutput.model_validate(result["task_result"])
    assert task_result.items_streamed == len(items)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_task_streams_errors(env: FlovynTestEnvironment) -> None:
    """Test that a task can stream error notifications.

    Verifies:
    - Task can call stream_error()
    - Task continues after streaming error (non-fatal)
    - Task completes successfully
    """
    handle = await env.start_workflow(
        "task-scheduler-workflow",
        {
            "task_name": "streaming-error-task",
            "task_input": StreamingErrorInput(
                error_message="Recoverable warning",
                error_code="WARN_001",
            ).model_dump(),
        },
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["task_completed"] is True
    task_result = StreamingErrorOutput.model_validate(result["task_result"])
    assert task_result.error_sent is True


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_task_streams_all_types(env: FlovynTestEnvironment) -> None:
    """Test that a task can stream all event types in sequence.

    Verifies:
    - Task can mix token, progress, data, and error streaming
    - All stream calls succeed
    - Task completes successfully
    """
    handle = await env.start_workflow(
        "task-scheduler-workflow",
        {
            "task_name": "streaming-all-types-task",
            "task_input": StreamingAllTypesInput(
                token="Generated token",
                progress=0.75,
                data={"key": "value", "count": 42},
                error_message="Warning: operation slow",
            ).model_dump(),
        },
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["task_completed"] is True
    task_result = StreamingAllTypesOutput.model_validate(result["task_result"])
    assert task_result.all_types_sent is True


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_task_streams_custom_tokens(env: FlovynTestEnvironment) -> None:
    """Test streaming custom/complex tokens.

    Verifies:
    - Tokens with special characters work
    - Unicode tokens work
    - Empty tokens work
    """
    # Include various token types
    tokens = [
        "normal",
        "",  # empty token
        "with spaces and\ttabs",
        "unicode: \u4e2d\u6587",  # Chinese characters
        '{"json": true}',  # JSON-like
        "emoji: \U0001f680",  # rocket emoji
    ]

    handle = await env.start_workflow(
        "task-scheduler-workflow",
        {
            "task_name": "streaming-token-task",
            "task_input": StreamingTokenInput(tokens=tokens).model_dump(),
        },
    )

    result = await env.await_completion(handle, timeout=timedelta(seconds=30))

    assert result["task_completed"] is True
    task_result = StreamingTokenOutput.model_validate(result["task_result"])
    assert task_result.token_count == len(tokens)
