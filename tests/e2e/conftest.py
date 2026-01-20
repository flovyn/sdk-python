"""E2E test configuration and fixtures."""

import logging
import uuid
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from flovyn.testing import FlovynTestEnvironment
from flovyn.testing.environment import cleanup_test_harness, get_test_harness

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy for session-scoped fixtures."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def test_harness():
    """Session-scoped fixture to manage the test harness.

    This starts the containers once for all tests in the session.
    """
    harness = await get_test_harness()
    yield harness
    await cleanup_test_harness()


@pytest_asyncio.fixture
async def env(test_harness) -> AsyncGenerator[FlovynTestEnvironment, None]:
    """Per-test fixture for the test environment.

    Each test gets a unique queue to prevent interference.
    """
    # Import fixtures to register them
    from tests.e2e.fixtures.tasks import (
        AddTask,
        EchoTask,
        ProgressTask,
        SlowTask,
        StreamingAllTypesTask,
        StreamingDataTask,
        StreamingErrorTask,
        StreamingProgressTask,
        StreamingTokenTask,
    )
    from tests.e2e.fixtures.workflows import (
        AwaitPromiseWorkflow,
        ChildFailureWorkflow,
        ChildLoopWorkflow,
        ChildWorkflowWorkflow,
        ComprehensiveWorkflow,
        DoublerWorkflow,
        EchoWorkflow,
        FailingWorkflow,
        FanOutFanInWorkflow,
        LargeBatchWorkflow,
        MixedCommandsWorkflow,
        MixedParallelWorkflow,
        MultiTaskWorkflow,
        NestedChildWorkflow,
        ParallelTasksWorkflow,
        RandomWorkflow,
        RunOperationWorkflow,
        SleepWorkflow,
        StatefulWorkflow,
        TaskSchedulerWorkflow,
        TaskSchedulingWorkflow,
    )

    # Create environment with unique queue for this module
    unique_queue = f"test-{uuid.uuid4().hex[:8]}"
    environment = FlovynTestEnvironment(queue=unique_queue)

    # Register all workflows
    environment.register_workflow(EchoWorkflow)
    environment.register_workflow(DoublerWorkflow)
    environment.register_workflow(FailingWorkflow)
    environment.register_workflow(StatefulWorkflow)
    environment.register_workflow(RunOperationWorkflow)
    environment.register_workflow(RandomWorkflow)
    environment.register_workflow(SleepWorkflow)
    environment.register_workflow(TaskSchedulingWorkflow)
    environment.register_workflow(MultiTaskWorkflow)
    environment.register_workflow(ParallelTasksWorkflow)
    environment.register_workflow(AwaitPromiseWorkflow)
    environment.register_workflow(ChildWorkflowWorkflow)
    environment.register_workflow(ChildFailureWorkflow)
    environment.register_workflow(NestedChildWorkflow)
    environment.register_workflow(MixedCommandsWorkflow)
    environment.register_workflow(FanOutFanInWorkflow)
    environment.register_workflow(LargeBatchWorkflow)
    environment.register_workflow(MixedParallelWorkflow)
    environment.register_workflow(ChildLoopWorkflow)
    environment.register_workflow(ComprehensiveWorkflow)
    environment.register_workflow(TaskSchedulerWorkflow)

    # Register all tasks
    environment.register_task(EchoTask)
    environment.register_task(AddTask)
    environment.register_task(SlowTask)
    environment.register_task(ProgressTask)
    environment.register_task(StreamingTokenTask)
    environment.register_task(StreamingProgressTask)
    environment.register_task(StreamingDataTask)
    environment.register_task(StreamingErrorTask)
    environment.register_task(StreamingAllTypesTask)

    async with environment:
        await environment.start()
        yield environment
