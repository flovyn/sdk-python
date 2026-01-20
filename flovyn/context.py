"""Context implementations for workflows and tasks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import UUID

from flovyn.exceptions import (
    ChildWorkflowFailed,
    PromiseRejected,
    PromiseTimeout,
    TaskCancelled,
    TaskFailed,
    WorkflowSuspended,
)
from flovyn.serde import Serializer, get_default_serde
from flovyn.types import RetryPolicy, TaskHandle, WorkflowHandle

T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class TaskContext(ABC):
    """Execution context for tasks.

    Provides progress reporting, cancellation checking, logging, and streaming
    for task execution.
    """

    @property
    @abstractmethod
    def task_execution_id(self) -> str:
        """Get the unique ID for this task execution."""
        ...

    @property
    @abstractmethod
    def attempt(self) -> int:
        """Get the current retry attempt number (1-based)."""
        ...

    @property
    @abstractmethod
    def is_cancelled(self) -> bool:
        """Check if task cancellation has been requested."""
        ...

    @abstractmethod
    def cancellation_error(self) -> TaskCancelled:
        """Create a cancellation error to raise."""
        ...

    @abstractmethod
    async def report_progress(self, progress: float, message: str | None = None) -> None:
        """Report task progress.

        Args:
            progress: Progress value from 0.0 to 1.0.
            message: Optional progress message.
        """
        ...

    @abstractmethod
    async def heartbeat(self) -> None:
        """Send a heartbeat to indicate the task is still running."""
        ...

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """Get a task-aware logger."""
        ...

    # Streaming methods

    @abstractmethod
    async def stream_token(self, text: str) -> bool:
        """Stream a token to connected clients.

        Use for streaming text generation from language models.

        Args:
            text: The token or text chunk to stream.

        Returns:
            True if the stream was acknowledged by the server.
        """
        ...

    @abstractmethod
    async def stream_progress(self, progress: float, details: str | None = None) -> bool:
        """Stream progress to connected clients.

        Unlike report_progress, this is ephemeral and not persisted.

        Args:
            progress: Progress value from 0.0 to 1.0.
            details: Optional progress details.

        Returns:
            True if the stream was acknowledged by the server.
        """
        ...

    @abstractmethod
    async def stream_data(self, data: Any) -> bool:
        """Stream arbitrary data to connected clients.

        Args:
            data: The data to stream (will be serialized to JSON).

        Returns:
            True if the stream was acknowledged by the server.
        """
        ...

    @abstractmethod
    async def stream_error(self, message: str, code: str | None = None) -> bool:
        """Stream an error notification to connected clients.

        Use to notify clients of recoverable errors during execution.
        For fatal errors, let the task fail normally.

        Args:
            message: Error message.
            code: Optional error code.

        Returns:
            True if the stream was acknowledged by the server.
        """
        ...


class WorkflowContext(ABC):
    """Deterministic execution context for workflows.

    All non-deterministic operations must go through this context to ensure
    proper replay behavior. Using standard library functions for time, random,
    etc. will cause determinism violations during replay.
    """

    # Deterministic time and randomness

    @abstractmethod
    def current_time(self) -> datetime:
        """Get current time (deterministic on replay).

        Returns:
            The current datetime in UTC.
        """
        ...

    @abstractmethod
    def current_time_millis(self) -> int:
        """Get current time as milliseconds since epoch.

        Returns:
            Milliseconds since Unix epoch.
        """
        ...

    @abstractmethod
    def random_uuid(self) -> UUID:
        """Generate a deterministic UUID.

        Returns:
            A UUID that is consistent across replays.
        """
        ...

    @abstractmethod
    def random(self) -> float:
        """Get a deterministic random number.

        Returns:
            A random float in [0, 1).
        """
        ...

    # Task execution

    @abstractmethod
    async def execute_task(
        self,
        task: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        queue: str | None = None,
    ) -> Any:
        """Execute a task and await its result.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: execute_task("add-task", {"a": 1, "b": 2})
        - Typed: execute_task(AddTask, AddInput(a=1, b=2))

        Args:
            task: The task kind (string) or task class/function to execute.
            input: The task input.
            timeout: Optional execution timeout.
            retry_policy: Optional retry policy.
            queue: Optional queue override.

        Returns:
            The task output.

        Raises:
            TaskFailed: If the task fails.
            TaskCancelled: If the task is cancelled.
            TaskTimeout: If the task times out.
        """
        ...

    @abstractmethod
    def schedule_task(
        self,
        task: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        timeout: timedelta | None = None,
        queue: str | None = None,
    ) -> TaskHandle[Any]:
        """Schedule a task for execution, returns immediately.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: schedule_task("add-task", {"a": 1, "b": 2})
        - Typed: schedule_task(AddTask, AddInput(a=1, b=2))

        Args:
            task: The task kind (string) or task class/function to execute.
            input: The task input.
            timeout: Optional execution timeout.
            queue: Optional queue override.

        Returns:
            A handle to await the task result.
        """
        ...

    # Child workflows

    @abstractmethod
    async def execute_workflow(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        workflow_id: str | None = None,
        timeout: timedelta | None = None,
        queue: str | None = None,
    ) -> Any:
        """Execute a child workflow and await its result.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: execute_workflow("order-workflow", {"order_id": "123"})
        - Typed: execute_workflow(OrderWorkflow, OrderInput(order_id="123"))

        Args:
            workflow: The workflow kind (string) or workflow class/function to execute.
            input: The workflow input.
            workflow_id: Optional custom workflow ID.
            timeout: Optional execution timeout.
            queue: Optional queue override.

        Returns:
            The workflow output.

        Raises:
            ChildWorkflowFailed: If the child workflow fails.
        """
        ...

    @abstractmethod
    def schedule_workflow(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        workflow_id: str | None = None,
        queue: str | None = None,
    ) -> WorkflowHandle[Any]:
        """Schedule a child workflow, returns immediately.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: schedule_workflow("order-workflow", {"order_id": "123"})
        - Typed: schedule_workflow(OrderWorkflow, OrderInput(order_id="123"))

        Args:
            workflow: The workflow kind (string) or workflow class/function to execute.
            input: The workflow input.
            workflow_id: Optional custom workflow ID.
            queue: Optional queue override.

        Returns:
            A handle to await the workflow result.
        """
        ...

    # Timers

    @abstractmethod
    async def sleep(self, duration: timedelta) -> None:
        """Sleep for a duration (durable across replays).

        Args:
            duration: The duration to sleep.
        """
        ...

    @abstractmethod
    async def sleep_until(self, until: datetime) -> None:
        """Sleep until a specific time.

        Args:
            until: The datetime to sleep until.
        """
        ...

    # Promises (external completion)

    @abstractmethod
    async def wait_for_promise(
        self,
        name: str,
        *,
        timeout: timedelta | None = None,
        type_hint: type[T] = Any,  # type: ignore[assignment]
    ) -> T:
        """Wait for an external promise to be resolved.

        Args:
            name: The promise name.
            timeout: Optional timeout for waiting.
            type_hint: Type hint for the promise value.

        Returns:
            The resolved promise value.

        Raises:
            PromiseTimeout: If the promise times out.
            PromiseRejected: If the promise is rejected.
        """
        ...

    # Signals

    @abstractmethod
    async def wait_for_signal(
        self,
        name: str,
        *,
        timeout: timedelta | None = None,
        type_hint: type[T] = Any,  # type: ignore[assignment]
    ) -> T:
        """Wait for an external signal.

        Args:
            name: The signal name.
            timeout: Optional timeout for waiting.
            type_hint: Type hint for the signal payload.

        Returns:
            The signal payload.
        """
        ...

    # State management

    @abstractmethod
    async def get_state(
        self,
        key: str,
        *,
        type_hint: type[T] = Any,  # type: ignore[assignment]
        default: T | None = None,
    ) -> T | None:
        """Get workflow state by key.

        Args:
            key: The state key.
            type_hint: Type hint for the value.
            default: Default value if key not found.

        Returns:
            The state value or default.
        """
        ...

    @abstractmethod
    async def set_state(self, key: str, value: Any) -> None:
        """Set workflow state.

        Args:
            key: The state key.
            value: The value to store.
        """
        ...

    @abstractmethod
    async def clear_state(self, key: str) -> None:
        """Clear a state key.

        Args:
            key: The state key to clear.
        """
        ...

    # Side effects (cached on replay)

    @abstractmethod
    async def run(
        self,
        name: str,
        fn: Callable[[], T | Awaitable[T]],
        *,
        max_attempts: int = 1,
        retry_interval: timedelta | None = None,
    ) -> T:
        """Execute a side effect (result cached on replay).

        This is used for non-deterministic operations that should only
        execute once and have their result cached for replay.

        Args:
            name: Unique name for this operation.
            fn: The function to execute (sync or async).
            max_attempts: Maximum retry attempts.
            retry_interval: Interval between retries.

        Returns:
            The function result.
        """
        ...

    # Cancellation

    @property
    @abstractmethod
    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested."""
        ...

    @abstractmethod
    def check_cancellation(self) -> None:
        """Raise WorkflowCancelled if cancellation requested.

        Raises:
            WorkflowCancelled: If cancellation has been requested.
        """
        ...

    # Logging

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """Get a workflow-aware logger."""
        ...

    # Workflow metadata

    @property
    @abstractmethod
    def workflow_execution_id(self) -> str:
        """Get the workflow execution ID."""
        ...


class TaskContextImpl(TaskContext):
    """Implementation of TaskContext wrapping FFI task context.

    This implementation uses the FFI task context for streaming and cancellation,
    with fallback callbacks for legacy functionality.
    """

    def __init__(
        self,
        task_execution_id: str,
        attempt: int,
        workflow_execution_id: str | None,
        cancellation_checker: Callable[[], bool],
        progress_reporter: Callable[[float, str | None], Awaitable[None]],
        heartbeat_sender: Callable[[], Awaitable[None]],
        ffi_context: Any | None = None,  # FfiTaskContext from FFI
    ) -> None:
        self._task_execution_id = task_execution_id
        self._attempt = attempt
        self._workflow_execution_id = workflow_execution_id
        self._cancellation_checker = cancellation_checker
        self._progress_reporter = progress_reporter
        self._heartbeat_sender = heartbeat_sender
        self._ffi_context = ffi_context
        self._logger = logging.getLogger(f"flovyn.task.{task_execution_id}")

    @property
    def task_execution_id(self) -> str:
        return self._task_execution_id

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def is_cancelled(self) -> bool:
        if self._ffi_context is not None:
            return bool(self._ffi_context.is_cancelled())
        return self._cancellation_checker()

    def cancellation_error(self) -> TaskCancelled:
        return TaskCancelled(f"Task {self._task_execution_id} was cancelled")

    async def report_progress(self, progress: float, message: str | None = None) -> None:
        if progress < 0.0 or progress > 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")
        await self._progress_reporter(progress, message)

    async def heartbeat(self) -> None:
        await self._heartbeat_sender()

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    # Streaming methods

    async def stream_token(self, text: str) -> bool:
        """Stream a token to connected clients."""
        if self._ffi_context is None:
            self._logger.warning("Streaming not available: no FFI context")
            return False
        try:
            return bool(self._ffi_context.stream_token(text))
        except Exception as e:
            self._logger.warning(f"Failed to stream token: {e}")
            return False

    async def stream_progress(self, progress: float, details: str | None = None) -> bool:
        """Stream progress to connected clients."""
        if self._ffi_context is None:
            self._logger.warning("Streaming not available: no FFI context")
            return False
        try:
            return bool(self._ffi_context.stream_progress(progress, details))
        except Exception as e:
            self._logger.warning(f"Failed to stream progress: {e}")
            return False

    async def stream_data(self, data: Any) -> bool:
        """Stream arbitrary data to connected clients."""
        if self._ffi_context is None:
            self._logger.warning("Streaming not available: no FFI context")
            return False
        try:
            from flovyn.serde import get_default_serde

            serializer = get_default_serde()
            data_bytes = serializer.serialize(data)
            return bool(self._ffi_context.stream_data(data_bytes))
        except Exception as e:
            self._logger.warning(f"Failed to stream data: {e}")
            return False

    async def stream_error(self, message: str, code: str | None = None) -> bool:
        """Stream an error notification to connected clients."""
        if self._ffi_context is None:
            self._logger.warning("Streaming not available: no FFI context")
            return False
        try:
            return bool(self._ffi_context.stream_error(message, code))
        except Exception as e:
            self._logger.warning(f"Failed to stream error: {e}")
            return False


class WorkflowContextImpl(WorkflowContext):
    """Implementation of WorkflowContext wrapping FFI workflow context."""

    def __init__(
        self,
        ffi_context: Any,  # FfiWorkflowContext from UniFFI
        serializer: Serializer[Any] | None = None,
        task_registry: dict[str, Any] | None = None,
        workflow_registry: dict[str, Any] | None = None,
    ) -> None:
        self._ffi = ffi_context
        self._serializer = serializer or get_default_serde()
        self._task_registry = task_registry or {}
        self._workflow_registry = workflow_registry or {}
        self._logger = logging.getLogger(f"flovyn.workflow.{ffi_context.workflow_execution_id()}")
        self._pending_tasks: dict[str, tuple[type[Any], Callable[..., Any]]] = {}
        self._pending_workflows: dict[str, tuple[type[Any], Callable[..., Any]]] = {}

    @property
    def workflow_execution_id(self) -> str:
        result: str = self._ffi.workflow_execution_id()
        return result

    def current_time(self) -> datetime:
        millis: int = self._ffi.current_time_millis()
        return datetime.fromtimestamp(millis / 1000.0, tz=UTC)

    def current_time_millis(self) -> int:
        result: int = self._ffi.current_time_millis()
        return result

    def random_uuid(self) -> UUID:
        result: str = self._ffi.random_uuid()
        return UUID(result)

    def random(self) -> float:
        result: float = self._ffi.random()
        return result

    async def execute_task(
        self,
        task: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        queue: str | None = None,
    ) -> Any:
        """Execute a task and await its result.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: execute_task("add-task", {"a": 1, "b": 2})
        - Typed: execute_task(AddTask, AddInput(a=1, b=2))

        Args:
            task: The task kind (string) or task class/function to execute.
            input: The task input (dict or serializable object).
            timeout: Optional timeout for task execution.
            retry_policy: Optional retry policy.
            queue: Optional queue override.

        Returns:
            The task result (as dict/primitive - caller deserializes as needed).
        """
        from flovyn.task import get_task_kind, is_task

        # Determine task kind from string or class
        if isinstance(task, str):
            task_kind = task
        elif is_task(task):
            task_kind = get_task_kind(task)
        else:
            raise ValueError(
                f"task must be a string kind or a @task decorated class/function, got {type(task)}"
            )

        input_bytes = self._serializer.serialize(input)
        timeout_ms = int(timeout.total_seconds() * 1000) if timeout else None

        result = self._ffi.schedule_task(task_kind, input_bytes, queue, timeout_ms)

        # Import the result types from FFI
        ffi = _get_ffi_module()

        if isinstance(result, ffi.FfiTaskResult.COMPLETED):
            # Return raw deserialized output - caller interprets as needed
            return self._serializer.deserialize(result.output, dict)
        elif isinstance(result, ffi.FfiTaskResult.FAILED):
            raise TaskFailed(result.error, retryable=result.retryable)
        elif isinstance(result, ffi.FfiTaskResult.PENDING):
            raise WorkflowSuspended(f"Waiting for task {result.task_execution_id}")
        else:
            raise RuntimeError(f"Unknown task result type: {type(result)}")

    def schedule_task(
        self,
        task: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        timeout: timedelta | None = None,
        queue: str | None = None,
    ) -> TaskHandle[Any]:
        """Schedule a task for execution, returns immediately.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: schedule_task("add-task", {"a": 1, "b": 2})
        - Typed: schedule_task(AddTask, AddInput(a=1, b=2))

        Args:
            task: The task kind (string) or task class/function to execute.
            input: The task input (dict or serializable object).
            timeout: Optional timeout for task execution.
            queue: Optional queue override.

        Returns:
            A handle to await the task result.
        """
        from flovyn.task import get_task_kind, is_task

        # Determine task kind from string or class
        if isinstance(task, str):
            task_kind = task
        elif is_task(task):
            task_kind = get_task_kind(task)
        else:
            raise ValueError(
                f"task must be a string kind or a @task decorated class/function, got {type(task)}"
            )

        input_bytes = self._serializer.serialize(input)
        timeout_ms = int(timeout.total_seconds() * 1000) if timeout else None

        result = self._ffi.schedule_task(task_kind, input_bytes, queue, timeout_ms)

        ffi = _get_ffi_module()

        if isinstance(result, ffi.FfiTaskResult.COMPLETED):
            # Return raw deserialized output - caller interprets as needed
            output = self._serializer.deserialize(result.output, dict)
            captured_output = output  # Capture for closure

            async def get_completed_result() -> Any:
                return captured_output

            return TaskHandle(
                task_execution_id=result.task_execution_id
                if hasattr(result, "task_execution_id")
                else str(self.random_uuid()),
                result_getter=get_completed_result,
            )
        elif isinstance(result, ffi.FfiTaskResult.PENDING):
            task_execution_id = result.task_execution_id

            async def get_result() -> Any:
                # This will be called on resume - the result should be available
                raise WorkflowSuspended(f"Waiting for task {task_execution_id}")

            return TaskHandle(
                task_execution_id=task_execution_id,
                result_getter=get_result,
            )
        elif isinstance(result, ffi.FfiTaskResult.FAILED):
            raise TaskFailed(result.error, retryable=result.retryable)
        else:
            raise RuntimeError(f"Unknown task result type: {type(result)}")

    async def execute_workflow(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        workflow_id: str | None = None,
        timeout: timedelta | None = None,
        queue: str | None = None,
    ) -> Any:
        """Execute a child workflow and await its result.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: execute_workflow("order-workflow", {"order_id": "123"})
        - Typed: execute_workflow(OrderWorkflow, OrderInput(order_id="123"))

        Args:
            workflow: The workflow kind (string) or workflow class/function to execute.
            input: The workflow input (dict or serializable object).
            workflow_id: Optional custom workflow ID.
            timeout: Optional timeout for workflow execution.
            queue: Optional queue override.

        Returns:
            The workflow result (as dict/primitive - caller deserializes as needed).
        """
        from flovyn.workflow import get_workflow_kind, is_workflow

        # Determine workflow kind from string or class
        if isinstance(workflow, str):
            workflow_kind = workflow
        elif is_workflow(workflow):
            workflow_kind = get_workflow_kind(workflow)
        else:
            raise ValueError(
                f"workflow must be a string kind or a @workflow decorated class/function, got {type(workflow)}"
            )

        input_bytes = self._serializer.serialize(input)
        name = workflow_id or str(self.random_uuid())

        result = self._ffi.schedule_child_workflow(name, workflow_kind, input_bytes, queue, None)

        ffi = _get_ffi_module()

        if isinstance(result, ffi.FfiChildWorkflowResult.COMPLETED):
            # Return raw deserialized output - caller interprets as needed
            return self._serializer.deserialize(result.output, dict)
        elif isinstance(result, ffi.FfiChildWorkflowResult.FAILED):
            raise ChildWorkflowFailed(result.error)
        elif isinstance(result, ffi.FfiChildWorkflowResult.PENDING):
            raise WorkflowSuspended(f"Waiting for child workflow {result.child_execution_id}")
        else:
            raise RuntimeError(f"Unknown child workflow result type: {type(result)}")

    def schedule_workflow(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        workflow_id: str | None = None,
        queue: str | None = None,
    ) -> WorkflowHandle[Any]:
        """Schedule a child workflow, returns immediately.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: schedule_workflow("order-workflow", {"order_id": "123"})
        - Typed: schedule_workflow(OrderWorkflow, OrderInput(order_id="123"))

        Args:
            workflow: The workflow kind (string) or workflow class/function to execute.
            input: The workflow input (dict or serializable object).
            workflow_id: Optional custom workflow ID.
            queue: Optional queue override.

        Returns:
            A handle to await the workflow result.
        """
        from flovyn.workflow import get_workflow_kind, is_workflow

        # Determine workflow kind from string or class
        if isinstance(workflow, str):
            workflow_kind = workflow
        elif is_workflow(workflow):
            workflow_kind = get_workflow_kind(workflow)
        else:
            raise ValueError(
                f"workflow must be a string kind or a @workflow decorated class/function, got {type(workflow)}"
            )

        input_bytes = self._serializer.serialize(input)
        name = workflow_id or str(self.random_uuid())

        result = self._ffi.schedule_child_workflow(name, workflow_kind, input_bytes, queue, None)

        ffi = _get_ffi_module()

        if isinstance(result, ffi.FfiChildWorkflowResult.COMPLETED):
            # Return raw deserialized output - caller interprets as needed
            output = self._serializer.deserialize(result.output, dict)
            return _create_completed_workflow_handle(name, name, output)
        elif isinstance(result, ffi.FfiChildWorkflowResult.PENDING):
            child_execution_id = result.child_execution_id
            return _create_pending_workflow_handle(name, child_execution_id)
        elif isinstance(result, ffi.FfiChildWorkflowResult.FAILED):
            raise ChildWorkflowFailed(result.error)
        else:
            raise RuntimeError(f"Unknown child workflow result type: {type(result)}")

    async def sleep(self, duration: timedelta) -> None:
        duration_ms = int(duration.total_seconds() * 1000)
        result = self._ffi.start_timer(duration_ms)

        ffi = _get_ffi_module()

        if isinstance(result, ffi.FfiTimerResult.FIRED):
            return
        elif isinstance(result, ffi.FfiTimerResult.PENDING):
            raise WorkflowSuspended(f"Waiting for timer {result.timer_id}")
        else:
            raise RuntimeError(f"Unknown timer result type: {type(result)}")

    async def sleep_until(self, until: datetime) -> None:
        now = self.current_time()
        if until <= now:
            return
        duration = until - now
        await self.sleep(duration)

    async def wait_for_promise(
        self,
        name: str,
        *,
        timeout: timedelta | None = None,
        type_hint: type[T] = Any,  # type: ignore[assignment]
    ) -> T:
        timeout_ms = int(timeout.total_seconds() * 1000) if timeout else None
        result = self._ffi.create_promise(name, timeout_ms)

        ffi = _get_ffi_module()

        if isinstance(result, ffi.FfiPromiseResult.RESOLVED):
            return self._serializer.deserialize(result.value, type_hint)  # type: ignore[no-any-return]
        elif isinstance(result, ffi.FfiPromiseResult.REJECTED):
            raise PromiseRejected(result.error, promise_name=name)
        elif isinstance(result, ffi.FfiPromiseResult.TIMED_OUT):
            raise PromiseTimeout(f"Promise '{name}' timed out", promise_name=name)
        elif isinstance(result, ffi.FfiPromiseResult.PENDING):
            raise WorkflowSuspended(f"Waiting for promise {result.promise_id}")
        else:
            raise RuntimeError(f"Unknown promise result type: {type(result)}")

    async def wait_for_signal(
        self,
        name: str,
        *,
        timeout: timedelta | None = None,
        type_hint: type[T] = Any,  # type: ignore[assignment]
    ) -> T:
        # Signals are handled through the activation job mechanism
        # For now, this is a placeholder that will be implemented via promise
        return await self.wait_for_promise(name, timeout=timeout, type_hint=type_hint)

    async def get_state(
        self,
        key: str,
        *,
        type_hint: type[T] = Any,  # type: ignore[assignment]
        default: T | None = None,
    ) -> T | None:
        value_bytes = self._ffi.get_state(key)
        if value_bytes is None:
            return default
        return self._serializer.deserialize(bytes(value_bytes), type_hint)  # type: ignore[no-any-return]

    async def set_state(self, key: str, value: Any) -> None:
        value_bytes = self._serializer.serialize(value)
        self._ffi.set_state(key, value_bytes)

    async def clear_state(self, key: str) -> None:
        self._ffi.clear_state(key)

    async def run(
        self,
        name: str,
        fn: Callable[[], T | Awaitable[T]],
        *,
        max_attempts: int = 1,
        retry_interval: timedelta | None = None,
    ) -> T:
        result = self._ffi.run_operation(name)

        ffi = _get_ffi_module()

        if isinstance(result, ffi.FfiOperationResult.CACHED):
            # Return cached result from replay
            return self._serializer.deserialize(result.value, Any)  # type: ignore[no-any-return]
        elif isinstance(result, ffi.FfiOperationResult.EXECUTE):
            # Execute the operation and record the result
            import asyncio
            import inspect

            try:
                if inspect.iscoroutinefunction(fn):
                    value = await fn()
                else:
                    value = fn()
                    if asyncio.iscoroutine(value):
                        value = await value

                # Record the result
                value_bytes = self._serializer.serialize(value)
                self._ffi.record_operation_result(name, value_bytes)
                return value  # type: ignore[no-any-return]
            except Exception:
                # For now, re-raise - retry logic can be added later
                raise
        else:
            raise RuntimeError(f"Unknown operation result type: {type(result)}")

    @property
    def is_cancellation_requested(self) -> bool:
        result: bool = self._ffi.is_cancellation_requested()
        return result

    def check_cancellation(self) -> None:
        from flovyn.exceptions import WorkflowCancelled

        if self.is_cancellation_requested:
            raise WorkflowCancelled()

    @property
    def logger(self) -> logging.Logger:
        return self._logger


def _get_ffi_module() -> Any:
    """Get the FFI module lazily."""
    from flovyn._native.loader import get_native_module

    return get_native_module()


async def _immediate_result(value: T) -> T:
    """Return a value immediately (for completed operations)."""
    return value


def _create_completed_workflow_handle(
    workflow_id: str,
    execution_id: str,
    output: Any,
) -> WorkflowHandle[Any]:
    """Create a workflow handle for an already-completed workflow."""
    return WorkflowHandle(
        workflow_id=workflow_id,
        workflow_execution_id=execution_id,
        result_getter=lambda timeout: _immediate_result(output),
        signal_sender=lambda name, payload: _immediate_result(None),
        query_executor=lambda name, args: _immediate_result(None),
        canceller=lambda: _immediate_result(None),
    )


def _create_pending_workflow_handle(
    workflow_id: str,
    execution_id: str,
) -> WorkflowHandle[Any]:
    """Create a workflow handle for a pending workflow."""

    async def get_result(timeout: timedelta | None) -> Any:
        raise WorkflowSuspended(f"Waiting for child workflow {execution_id}")

    async def noop(*args: Any, **kwargs: Any) -> None:
        pass

    async def query(*args: Any, **kwargs: Any) -> Any:
        return None

    return WorkflowHandle(
        workflow_id=workflow_id,
        workflow_execution_id=execution_id,
        result_getter=get_result,
        signal_sender=noop,
        query_executor=query,
        canceller=noop,
    )
