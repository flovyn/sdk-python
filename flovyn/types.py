"""Core types and protocols for Flovyn SDK."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from flovyn.context import TaskContext, WorkflowContext


# Streaming types
StreamEventType = Literal["token", "progress", "data", "error"]


@dataclass
class StreamEvent:
    """A streaming event from a task.

    Attributes:
        type: The type of stream event.
        data: The event payload (varies by type).
        timestamp: Unix timestamp in milliseconds when the event was created.
    """

    type: StreamEventType
    data: Any
    timestamp: int


# Type variables for generics
InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for task retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts (including the first one).
        initial_interval: Initial delay between retries.
        max_interval: Maximum delay between retries (caps exponential backoff).
        backoff_coefficient: Multiplier for exponential backoff (default 2.0).
    """

    max_attempts: int = 3
    initial_interval: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    max_interval: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    backoff_coefficient: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff_coefficient < 1.0:
            raise ValueError("backoff_coefficient must be at least 1.0")


@runtime_checkable
class Workflow(Protocol[InputT, OutputT]):
    """Protocol for typed workflow definitions.

    Workflows must implement a `run` method that takes a WorkflowContext
    and input, returning the output.
    """

    async def run(self, ctx: WorkflowContext, input: InputT) -> OutputT:
        """Execute the workflow logic.

        Args:
            ctx: The workflow context for deterministic operations.
            input: The workflow input.

        Returns:
            The workflow output.
        """
        ...


@runtime_checkable
class Task(Protocol[InputT, OutputT]):
    """Protocol for typed task definitions.

    Tasks must implement a `run` method that takes a TaskContext
    and input, returning the output.
    """

    async def run(self, ctx: TaskContext, input: InputT) -> OutputT:
        """Execute the task logic.

        Args:
            ctx: The task context for progress reporting and cancellation.
            input: The task input.

        Returns:
            The task output.
        """
        ...


class TaskHandle(Generic[T]):
    """Handle to a scheduled task with typed result.

    Allows waiting for task completion and accessing task metadata.
    """

    def __init__(
        self,
        task_execution_id: str,
        result_getter: Callable[[], Awaitable[T]],
    ) -> None:
        self._task_execution_id = task_execution_id
        self._result_getter = result_getter
        self._result: T | None = None
        self._completed = False

    @property
    def task_execution_id(self) -> str:
        """Get the unique task execution ID."""
        return self._task_execution_id

    async def result(self) -> T:
        """Wait for and return the task result.

        Returns:
            The task output.

        Raises:
            TaskFailed: If the task failed.
            TaskCancelled: If the task was cancelled.
            TaskTimeout: If the task timed out.
        """
        if not self._completed:
            self._result = await self._result_getter()
            self._completed = True
        return self._result  # type: ignore[return-value]


class WorkflowHandle(Generic[T]):
    """Handle to a running workflow with typed result.

    Allows waiting for workflow completion, sending signals, and querying state.
    """

    def __init__(
        self,
        workflow_id: str,
        workflow_execution_id: str,
        result_getter: Callable[[timedelta | None], Awaitable[T]],
        signal_sender: Callable[[str, Any], Awaitable[None]],
        query_executor: Callable[[str, Any], Awaitable[Any]],
        canceller: Callable[[], Awaitable[None]],
    ) -> None:
        self._workflow_id = workflow_id
        self._workflow_execution_id = workflow_execution_id
        self._result_getter = result_getter
        self._signal_sender = signal_sender
        self._query_executor = query_executor
        self._canceller = canceller

    @property
    def workflow_id(self) -> str:
        """Get the workflow ID."""
        return self._workflow_id

    @property
    def workflow_execution_id(self) -> str:
        """Get the workflow execution ID."""
        return self._workflow_execution_id

    async def result(self, *, timeout: timedelta | None = None) -> T:
        """Wait for and return the workflow result.

        Args:
            timeout: Maximum time to wait for the result.

        Returns:
            The workflow output.

        Raises:
            WorkflowCancelled: If the workflow was cancelled.
            TimeoutError: If the timeout is exceeded.
        """
        return await self._result_getter(timeout)

    async def signal(self, signal_name: str, payload: Any = None) -> None:
        """Send a signal to the workflow.

        Args:
            signal_name: The name of the signal.
            payload: Optional payload data.
        """
        await self._signal_sender(signal_name, payload)

    async def query(self, query_name: str, args: Any = None) -> Any:
        """Query the workflow state.

        Args:
            query_name: The name of the query.
            args: Optional query arguments.

        Returns:
            The query result.
        """
        return await self._query_executor(query_name, args)

    async def cancel(self) -> None:
        """Request cancellation of the workflow."""
        await self._canceller()


# Type aliases for workflow/task functions
WorkflowFunction = Callable[["WorkflowContext", Any], Awaitable[Any]]
TaskFunction = Callable[["TaskContext", Any], Awaitable[Any]]


@dataclass
class WorkflowMetadata:
    """Metadata for a registered workflow.

    Attributes:
        kind: The workflow kind identifier.
        name: Human-readable workflow name.
        description: Optional description.
        version: Optional version string.
        tags: Optional tags for categorization.
        timeout: Optional workflow timeout.
        input_type: The input type (for type checking).
        output_type: The output type (for type checking).
    """

    kind: str
    name: str
    description: str | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)
    timeout: timedelta | None = None
    input_type: type[Any] | None = None
    output_type: type[Any] | None = None
    cancellable: bool = True


@dataclass
class TaskMetadata:
    """Metadata for a registered task.

    Attributes:
        kind: The task kind identifier.
        name: Human-readable task name.
        description: Optional description.
        version: Optional version string.
        timeout: Optional task timeout.
        retry_policy: Optional retry policy.
        cancellable: Whether the task supports cancellation.
        input_type: The input type (for type checking).
        output_type: The output type (for type checking).
    """

    kind: str
    name: str
    description: str | None = None
    version: str | None = None
    timeout: timedelta | None = None
    retry_policy: RetryPolicy | None = None
    cancellable: bool = True
    input_type: type[Any] | None = None
    output_type: type[Any] | None = None
