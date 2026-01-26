"""Mock contexts for unit testing workflows and tasks."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import UUID, uuid4

from flovyn.context import TaskContext, WorkflowContext
from flovyn.exceptions import (
    ChildWorkflowFailed,
    PromiseRejected,
    PromiseTimeout,
    TaskCancelled,
    TaskFailed,
    WorkflowCancelled,
)
from flovyn.types import RetryPolicy, TaskHandle, WorkflowHandle

T = TypeVar("T")


@dataclass
class ProgressReport:
    """A recorded progress report from a task."""

    progress: float
    message: str | None
    timestamp: datetime


class TimeController:
    """Controller for manipulating time in tests.

    Allows tests to advance time without waiting, useful for testing
    timer-based workflows.
    """

    def __init__(self, start_time: datetime | None = None) -> None:
        """Initialize the time controller.

        Args:
            start_time: The initial time (defaults to now in UTC).
        """
        self._current_time = start_time or datetime.now(UTC)
        self._pending_timers: list[tuple[datetime, Callable[[], Any]]] = []

    @property
    def current_time(self) -> datetime:
        """Get the current simulated time."""
        return self._current_time

    def current_time_millis(self) -> int:
        """Get the current time as milliseconds since epoch."""
        return int(self._current_time.timestamp() * 1000)

    async def advance(self, duration: timedelta) -> None:
        """Advance time by the given duration.

        This will fire any timers that would have expired during
        the advancement.

        Args:
            duration: The duration to advance time by.
        """
        target_time = self._current_time + duration
        self._current_time = target_time

        # Fire expired timers
        fired = []
        remaining = []
        for fire_time, callback in self._pending_timers:
            if fire_time <= target_time:
                fired.append(callback)
            else:
                remaining.append((fire_time, callback))

        self._pending_timers = remaining

        for callback in fired:
            callback()

    def schedule_timer(self, duration: timedelta, callback: Callable[[], Any]) -> str:
        """Schedule a timer to fire after the given duration.

        Args:
            duration: The duration until the timer fires.
            callback: The function to call when the timer fires.

        Returns:
            A timer ID.
        """
        fire_time = self._current_time + duration
        timer_id = str(uuid4())
        self._pending_timers.append((fire_time, callback))
        return timer_id


@dataclass
class StreamEvent:
    """A recorded stream event from a task."""

    event_type: str  # "token", "progress", "data", "error"
    payload: Any
    timestamp: datetime


class MockTaskContext(TaskContext):
    """Mock TaskContext for unit testing tasks."""

    def __init__(
        self,
        task_execution_id: str | None = None,
        attempt: int = 1,
        is_cancelled: bool = False,
    ) -> None:
        """Initialize the mock task context.

        Args:
            task_execution_id: Optional task execution ID (generates one if not provided).
            attempt: The retry attempt number (1-based).
            is_cancelled: Whether cancellation is requested.
        """
        self._task_execution_id = task_execution_id or str(uuid4())
        self._attempt = attempt
        self._is_cancelled = is_cancelled
        self._progress_reports: list[ProgressReport] = []
        self._heartbeat_count = 0
        self._stream_events: list[StreamEvent] = []
        self._logger = logging.getLogger(f"flovyn.test.task.{self._task_execution_id}")

    @property
    def task_execution_id(self) -> str:
        return self._task_execution_id

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def cancellation_error(self) -> TaskCancelled:
        return TaskCancelled(f"Task {self._task_execution_id} was cancelled")

    async def report_progress(self, progress: float, message: str | None = None) -> None:
        if progress < 0.0 or progress > 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")
        self._progress_reports.append(
            ProgressReport(
                progress=progress,
                message=message,
                timestamp=datetime.now(UTC),
            )
        )

    async def heartbeat(self) -> None:
        self._heartbeat_count += 1

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    # Test utilities

    @property
    def progress_reports(self) -> list[ProgressReport]:
        """Get all progress reports made by the task."""
        return self._progress_reports

    @property
    def heartbeat_count(self) -> int:
        """Get the number of heartbeats sent."""
        return self._heartbeat_count

    def set_cancelled(self, cancelled: bool = True) -> None:
        """Set the cancellation state."""
        self._is_cancelled = cancelled

    # Streaming methods

    async def stream(self, event_type: str, data: Any) -> bool:
        """Stream an event to connected clients (mock implementation)."""
        self._stream_events.append(
            StreamEvent(
                event_type=event_type,
                payload=data,
                timestamp=datetime.now(UTC),
            )
        )
        return True

    async def stream_token(self, text: str) -> bool:
        """Stream a token to connected clients (mock implementation)."""
        self._stream_events.append(
            StreamEvent(
                event_type="token",
                payload={"text": text},
                timestamp=datetime.now(UTC),
            )
        )
        return True

    async def stream_progress(self, progress: float, details: str | None = None) -> bool:
        """Stream progress to connected clients (mock implementation)."""
        self._stream_events.append(
            StreamEvent(
                event_type="progress",
                payload={"progress": progress, "details": details},
                timestamp=datetime.now(UTC),
            )
        )
        return True

    async def stream_data(self, data: Any) -> bool:
        """Stream arbitrary data to connected clients (mock implementation)."""
        self._stream_events.append(
            StreamEvent(
                event_type="data",
                payload={"data": data},
                timestamp=datetime.now(UTC),
            )
        )
        return True

    async def stream_error(self, message: str, code: str | None = None) -> bool:
        """Stream an error notification to connected clients (mock implementation)."""
        self._stream_events.append(
            StreamEvent(
                event_type="error",
                payload={"message": message, "code": code},
                timestamp=datetime.now(UTC),
            )
        )
        return True

    # Test utilities for streaming

    @property
    def stream_events(self) -> list[StreamEvent]:
        """Get all stream events recorded by the task."""
        return self._stream_events

    @property
    def streamed_tokens(self) -> list[str]:
        """Get all streamed tokens."""
        return [e.payload["text"] for e in self._stream_events if e.event_type == "token"]


class MockWorkflowContext(WorkflowContext):
    """Mock WorkflowContext for unit testing workflows."""

    def __init__(
        self,
        workflow_execution_id: str | None = None,
        time_controller: TimeController | None = None,
    ) -> None:
        """Initialize the mock workflow context.

        Args:
            workflow_execution_id: Optional workflow execution ID.
            time_controller: Optional time controller for manipulating time.
        """
        self._workflow_execution_id = workflow_execution_id or str(uuid4())
        self._time_controller = time_controller or TimeController()
        self._random_counter = 0
        self._state: dict[str, Any] = {}
        self._executed_tasks: list[tuple[str, Any]] = []
        self._executed_workflows: list[tuple[str, Any]] = []
        self._task_results: dict[str, Any] = {}
        self._workflow_results: dict[str, Any] = {}
        self._task_failures: dict[str, Exception] = {}
        self._workflow_failures: dict[str, Exception] = {}
        self._promise_values: dict[str, Any] = {}
        self._promise_rejections: dict[str, str] = {}
        self._signal_values: dict[str, Any] = {}
        self._run_results: dict[str, Any] = {}
        self._cancellation_requested = False
        self._logger = logging.getLogger(f"flovyn.test.workflow.{self._workflow_execution_id}")

    @property
    def workflow_execution_id(self) -> str:
        return self._workflow_execution_id

    def current_time(self) -> datetime:
        return self._time_controller.current_time

    def current_time_millis(self) -> int:
        return self._time_controller.current_time_millis()

    def random_uuid(self) -> UUID:
        # Generate deterministic UUIDs for testing
        self._random_counter += 1
        return UUID(int=self._random_counter)

    def random(self) -> float:
        # Generate deterministic random numbers for testing
        self._random_counter += 1
        return (self._random_counter % 1000) / 1000.0

    def _resolve_task_kind(self, task: str | type[Any] | Callable[..., Any]) -> str:
        """Resolve task kind from string or class."""
        if isinstance(task, str):
            return task
        from flovyn.task import get_task_kind, is_task

        if is_task(task):
            return get_task_kind(task)
        raise ValueError(
            f"task must be a string kind or a @task decorated class/function, got {type(task)}"
        )

    def _resolve_workflow_kind(self, workflow: str | type[Any] | Callable[..., Any]) -> str:
        """Resolve workflow kind from string or class."""
        if isinstance(workflow, str):
            return workflow
        from flovyn.workflow import get_workflow_kind, is_workflow

        if is_workflow(workflow):
            return get_workflow_kind(workflow)
        raise ValueError(
            f"workflow must be a string kind or a @workflow decorated class/function, got {type(workflow)}"
        )

    async def schedule(
        self,
        task: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        queue: str | None = None,
    ) -> Any:
        # Resolve task kind from string or class
        task_kind = self._resolve_task_kind(task)
        self._executed_tasks.append((task_kind, input))

        if task_kind in self._task_failures:
            raise self._task_failures[task_kind]

        if task_kind in self._task_results:
            result = self._task_results[task_kind]
            if callable(result):
                return result(input)
            return result

        raise TaskFailed(f"No mock result configured for task {task_kind}")

    def schedule_async(
        self,
        task: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        timeout: timedelta | None = None,
        queue: str | None = None,
    ) -> TaskHandle[Any]:
        # Resolve task kind from string or class
        task_kind = self._resolve_task_kind(task)
        self._executed_tasks.append((task_kind, input))

        async def get_result() -> Any:
            if task_kind in self._task_failures:
                raise self._task_failures[task_kind]
            if task_kind in self._task_results:
                result = self._task_results[task_kind]
                if callable(result):
                    return result(input)
                return result
            raise TaskFailed(f"No mock result configured for task {task_kind}")

        return TaskHandle(
            task_execution_id=str(self.random_uuid()),
            result_getter=get_result,
        )

    async def schedule_workflow(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        workflow_id: str | None = None,
        timeout: timedelta | None = None,
        queue: str | None = None,
    ) -> Any:
        # Resolve workflow kind from string or class
        workflow_kind = self._resolve_workflow_kind(workflow)
        self._executed_workflows.append((workflow_kind, input))

        if workflow_kind in self._workflow_failures:
            raise self._workflow_failures[workflow_kind]

        if workflow_kind in self._workflow_results:
            result = self._workflow_results[workflow_kind]
            if callable(result):
                return result(input)
            return result

        raise ChildWorkflowFailed(f"No mock result configured for workflow {workflow_kind}")

    def schedule_workflow_async(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        workflow_id: str | None = None,
        queue: str | None = None,
    ) -> WorkflowHandle[Any]:
        # Resolve workflow kind from string or class
        workflow_kind = self._resolve_workflow_kind(workflow)
        self._executed_workflows.append((workflow_kind, input))
        wf_id = workflow_id or str(self.random_uuid())

        async def get_result(timeout: timedelta | None) -> Any:
            if workflow_kind in self._workflow_failures:
                raise self._workflow_failures[workflow_kind]
            if workflow_kind in self._workflow_results:
                result = self._workflow_results[workflow_kind]
                if callable(result):
                    return result(input)
                return result
            raise ChildWorkflowFailed(f"No mock result configured for workflow {workflow_kind}")

        async def noop(*args: Any, **kwargs: Any) -> None:
            pass

        async def query(*args: Any, **kwargs: Any) -> Any:
            return None

        return WorkflowHandle(
            workflow_id=wf_id,
            workflow_execution_id=wf_id,
            result_getter=get_result,
            signal_sender=noop,
            query_executor=query,
            canceller=noop,
        )

    async def sleep(self, duration: timedelta) -> None:
        # In mock context, sleep is instant
        pass

    async def sleep_until(self, until: datetime) -> None:
        # In mock context, sleep is instant
        pass

    async def promise(
        self,
        name: str,
        *,
        timeout: timedelta | None = None,
        type_hint: type[T] = Any,  # type: ignore[assignment]
    ) -> T:
        if name in self._promise_rejections:
            raise PromiseRejected(
                self._promise_rejections[name],
                promise_name=name,
                rejection_reason=self._promise_rejections[name],
            )

        if name in self._promise_values:
            return self._promise_values[name]  # type: ignore[no-any-return]

        raise PromiseTimeout(f"Promise '{name}' not resolved in mock", promise_name=name)

    async def wait_for_signal(
        self,
        name: str,
        *,
        timeout: timedelta | None = None,
        type_hint: type[T] = Any,  # type: ignore[assignment]
    ) -> T:
        if name in self._signal_values:
            return self._signal_values[name]  # type: ignore[no-any-return]

        raise TimeoutError(f"Signal '{name}' not received in mock")

    async def get(
        self,
        key: str,
        *,
        type_hint: type[T] = Any,  # type: ignore[assignment]
        default: T | None = None,
    ) -> T | None:
        return self._state.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    async def clear(self, key: str) -> None:
        self._state.pop(key, None)

    async def clear_all(self) -> None:
        self._state.clear()

    def state_keys(self) -> list[str]:
        return list(self._state.keys())

    async def run(
        self,
        name: str,
        fn: Callable[[], T | Awaitable[T]],
        *,
        max_attempts: int = 1,
        retry_interval: timedelta | None = None,
    ) -> T:
        if name in self._run_results:
            return self._run_results[name]  # type: ignore[no-any-return]

        # Execute the function
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(fn):
            result = await fn()
        else:
            result = fn()
            if asyncio.iscoroutine(result):
                result = await result

        return result  # type: ignore[no-any-return]

    @property
    def is_cancellation_requested(self) -> bool:
        return self._cancellation_requested

    def check_cancellation(self) -> None:
        if self._cancellation_requested:
            raise WorkflowCancelled()

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    # Test utilities

    def mock_task_result(
        self,
        task: str | type[Any] | Callable[..., Any],
        result: Any | Callable[[Any], Any],
    ) -> None:
        """Configure a mock result for a task.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: mock_task_result("add-task", {"sum": 10})
        - Typed: mock_task_result(AddTask, {"sum": 10})

        Args:
            task: The task kind (string) or task class/function.
            result: The result to return, or a callable that takes input and returns result.
        """
        task_kind = self._resolve_task_kind(task)
        self._task_results[task_kind] = result

    def mock_task_failure(
        self,
        task: str | type[Any] | Callable[..., Any],
        error: Exception,
    ) -> None:
        """Configure a task to fail with the given error.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: mock_task_failure("add-task", TaskFailed("error"))
        - Typed: mock_task_failure(AddTask, TaskFailed("error"))

        Args:
            task: The task kind (string) or task class/function.
            error: The exception to raise.
        """
        task_kind = self._resolve_task_kind(task)
        self._task_failures[task_kind] = error

    def mock_workflow_result(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        result: Any | Callable[[Any], Any],
    ) -> None:
        """Configure a mock result for a child workflow.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: mock_workflow_result("order-workflow", {"status": "done"})
        - Typed: mock_workflow_result(OrderWorkflow, {"status": "done"})

        Args:
            workflow: The workflow kind (string) or workflow class/function.
            result: The result to return, or a callable that takes input and returns result.
        """
        workflow_kind = self._resolve_workflow_kind(workflow)
        self._workflow_results[workflow_kind] = result

    def mock_workflow_failure(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        error: Exception,
    ) -> None:
        """Configure a child workflow to fail with the given error.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: mock_workflow_failure("order-workflow", ChildWorkflowFailed("error"))
        - Typed: mock_workflow_failure(OrderWorkflow, ChildWorkflowFailed("error"))

        Args:
            workflow: The workflow kind (string) or workflow class/function.
            error: The exception to raise.
        """
        workflow_kind = self._resolve_workflow_kind(workflow)
        self._workflow_failures[workflow_kind] = error

    def mock_promise_value(self, name: str, value: Any) -> None:
        """Configure a promise to resolve with the given value.

        Args:
            name: The promise name.
            value: The value to resolve with.
        """
        self._promise_values[name] = value

    def mock_promise_rejection(self, name: str, error: str) -> None:
        """Configure a promise to be rejected with the given error.

        Args:
            name: The promise name.
            error: The rejection error message.
        """
        self._promise_rejections[name] = error

    def mock_signal_value(self, name: str, value: Any) -> None:
        """Configure a signal to deliver the given value.

        Args:
            name: The signal name.
            value: The signal payload.
        """
        self._signal_values[name] = value

    def mock_run_result(self, name: str, result: Any) -> None:
        """Configure a ctx.run() operation to return the given result.

        Args:
            name: The operation name.
            result: The result to return.
        """
        self._run_results[name] = result

    def request_cancellation(self) -> None:
        """Request workflow cancellation."""
        self._cancellation_requested = True

    @property
    def executed_tasks(self) -> list[tuple[str, Any]]:
        """Get list of (task_kind, input) tuples for all executed tasks."""
        return self._executed_tasks

    @property
    def executed_workflows(self) -> list[tuple[str, Any]]:
        """Get list of (workflow_kind, input) tuples for all executed child workflows."""
        return self._executed_workflows

    @property
    def state(self) -> dict[str, Any]:
        """Get the current state dictionary."""
        return self._state
