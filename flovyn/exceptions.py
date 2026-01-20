"""Flovyn SDK exceptions."""

from __future__ import annotations


class FlovynError(Exception):
    """Base exception for all Flovyn errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


# Workflow exceptions


class WorkflowSuspended(FlovynError):
    """Raised internally when workflow needs to suspend for pending work.

    This exception is used internally to signal that the workflow has scheduled
    async operations (tasks, timers, promises) and needs to wait for their completion.
    User code should not catch this exception.
    """

    def __init__(self, reason: str = "Workflow suspended waiting for pending operations") -> None:
        super().__init__(reason)


class WorkflowCancelled(FlovynError):
    """Raised when workflow cancellation is requested.

    This can be raised by:
    - Calling `ctx.check_cancellation()` when cancellation is requested
    - The workflow engine when an external cancellation request is received
    """

    def __init__(self, reason: str = "Workflow cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


class DeterminismViolation(FlovynError):
    """Raised when workflow code violates determinism during replay.

    This occurs when the workflow code produces different commands during
    replay than it did during the original execution. Common causes:
    - Using non-deterministic operations (random, time, etc.) directly
    - Changing workflow logic between executions
    - External state affecting workflow decisions
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"Determinism violation: {message}")
        self.details = message


# Task exceptions


class TaskFailed(FlovynError):
    """Raised when a task fails.

    Attributes:
        retryable: Whether the task failure is retryable.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class TaskCancelled(FlovynError):
    """Raised when a task is cancelled.

    This can be raised by:
    - The task checking `ctx.is_cancelled` and raising `ctx.cancellation_error()`
    - The workflow engine when the parent workflow is cancelled
    """

    def __init__(self, reason: str = "Task cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


class TaskTimeout(FlovynError):
    """Raised when a task exceeds its timeout.

    Attributes:
        timeout_ms: The timeout in milliseconds that was exceeded.
    """

    def __init__(self, message: str = "Task timed out", *, timeout_ms: int | None = None) -> None:
        super().__init__(message)
        self.timeout_ms = timeout_ms


# Child workflow exceptions


class ChildWorkflowFailed(FlovynError):
    """Raised when a child workflow fails.

    Attributes:
        child_workflow_id: The ID of the failed child workflow.
    """

    def __init__(self, message: str, *, child_workflow_id: str | None = None) -> None:
        super().__init__(message)
        self.child_workflow_id = child_workflow_id


# Promise exceptions


class PromiseTimeout(FlovynError):
    """Raised when a promise times out.

    Attributes:
        promise_name: The name of the promise that timed out.
    """

    def __init__(
        self, message: str = "Promise timed out", *, promise_name: str | None = None
    ) -> None:
        super().__init__(message)
        self.promise_name = promise_name


class PromiseRejected(FlovynError):
    """Raised when a promise is rejected.

    Attributes:
        promise_name: The name of the promise that was rejected.
        rejection_reason: The reason for rejection.
    """

    def __init__(
        self,
        message: str = "Promise rejected",
        *,
        promise_name: str | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.promise_name = promise_name
        self.rejection_reason = rejection_reason


# Connection/infrastructure exceptions


class ConnectionError(FlovynError):
    """Raised when connection to the Flovyn server fails."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Connection error: {message}")


class ConfigurationError(FlovynError):
    """Raised when there is a configuration error."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Configuration error: {message}")
