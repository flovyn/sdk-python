"""E2E tests for the typed API (passing classes instead of strings).

These tests verify that the typed API works correctly for use cases
where the client and worker are on the same machine (single-server).
"""

import pytest

from flovyn.testing import FlovynTestEnvironment

# Import workflow classes and typed input/output models for typed API testing
from tests.e2e.fixtures.workflows import (
    DoublerInput,
    DoublerOutput,
    DoublerWorkflow,
    EchoInput,
    EchoOutput,
    EchoWorkflow,
    TypedTaskInput,
    TypedTaskOutput,
    TypedTaskWorkflow,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_start_workflow_with_typed_input_output(env: FlovynTestEnvironment) -> None:
    """Test starting a workflow using the typed API with typed input and output models."""
    # Use the typed API: pass class and Pydantic model
    handle = await env.start_workflow(
        EchoWorkflow,  # Workflow class
        EchoInput(message="Hello from typed API!"),  # Typed input model
    )

    result = await env.await_completion(handle)

    # Parse result into typed output model for type-safe access
    output = EchoOutput.model_validate(result)
    assert output.message == "Hello from typed API!"
    assert output.timestamp is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_start_workflow_with_typed_input_output_doubler(env: FlovynTestEnvironment) -> None:
    """Test starting a doubler workflow using the typed API with typed input and output."""
    # Use the typed API: pass class and Pydantic model
    handle = await env.start_workflow(
        DoublerWorkflow,  # Workflow class
        DoublerInput(value=21),  # Typed input model
    )

    result = await env.await_completion(handle)

    # Parse result into typed output model for type-safe access
    output = DoublerOutput.model_validate(result)
    assert output.result == 42


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_start_and_await_with_typed_input_output(env: FlovynTestEnvironment) -> None:
    """Test start_and_await helper with typed API and typed input/output."""
    # Use the typed API with the combined start_and_await method
    result = await env.start_and_await(
        EchoWorkflow,  # Workflow class
        EchoInput(message="Combined start and await!"),  # Typed input model
    )

    # Parse result into typed output model for type-safe access
    output = EchoOutput.model_validate(result)
    assert output.message == "Combined start and await!"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_typed_task_execution_in_workflow(env: FlovynTestEnvironment) -> None:
    """Test workflow that uses typed API internally to execute a task.

    This verifies that ctx.execute_task(TaskClass, input) works within a workflow.
    """
    # The TypedTaskWorkflow internally uses AddTask class instead of string
    handle = await env.start_workflow(
        TypedTaskWorkflow,  # This workflow uses ctx.execute_task(AddTask, ...)
        TypedTaskInput(a=10, b=32),  # Typed input model
    )

    result = await env.await_completion(handle)

    # Parse result into typed output model for type-safe access
    output = TypedTaskOutput.model_validate(result)
    assert output.result == 42  # 10 + 32
