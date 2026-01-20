"""Task definition decorator and utilities."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import timedelta
from typing import Any, TypeVar, get_type_hints

from flovyn.types import RetryPolicy, TaskMetadata

T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

# Global task registry
_task_registry: dict[str, tuple[Any, TaskMetadata]] = {}


def get_task_registry() -> dict[str, tuple[Any, TaskMetadata]]:
    """Get the global task registry."""
    return _task_registry


def register_task(task: Any, metadata: TaskMetadata) -> None:
    """Register a task in the global registry."""
    _task_registry[metadata.kind] = (task, metadata)


def get_task_kind(task: type[Any] | Callable[..., Any]) -> str:
    """Get the task kind from a task class or function."""
    if hasattr(task, "__flovyn_task_metadata__"):
        metadata: TaskMetadata = task.__flovyn_task_metadata__
        return metadata.kind
    raise ValueError(f"Object {task} is not a registered task. Use @task decorator.")


def get_task_input_type(task: type[Any] | Callable[..., Any]) -> type[Any]:
    """Get the input type from a task class or function."""
    if hasattr(task, "__flovyn_task_metadata__"):
        metadata: TaskMetadata = task.__flovyn_task_metadata__
        return metadata.input_type or Any
    return Any


def get_task_output_type(task: type[Any] | Callable[..., Any]) -> type[Any]:
    """Get the output type from a task class or function."""
    if hasattr(task, "__flovyn_task_metadata__"):
        metadata: TaskMetadata = task.__flovyn_task_metadata__
        return metadata.output_type or Any
    return Any


def task(
    *,
    name: str,
    description: str | None = None,
    version: str | None = None,
    timeout: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
    cancellable: bool = True,
) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
    """Decorator to define a task.

    Can be applied to either a class with a `run` method or an async function.

    Args:
        name: Unique identifier for the task (required).
        description: Optional human-readable description.
        version: Optional version string.
        timeout: Optional execution timeout.
        retry_policy: Optional retry configuration.
        cancellable: Whether the task supports cancellation (default True).

    Examples:
        Class-based task::

            @task(name="send-email")
            class SendEmail:
                async def run(self, ctx: TaskContext, input: EmailRequest) -> EmailResult:
                    ...

        Function-based task::

            @task(name="greet")
            async def greet_task(ctx: TaskContext, name: str) -> str:
                return f"Hello, {name}!"
    """

    def decorator(target: type[T] | Callable[..., T]) -> type[T] | Callable[..., T]:
        # Extract type hints for input and output
        input_type: type[Any] | None = None
        output_type: type[Any] | None = None

        if isinstance(target, type):
            # Class-based task
            if hasattr(target, "run"):
                run_method = target.run
                try:
                    hints = get_type_hints(run_method)
                    # Skip 'self' and 'ctx' parameters
                    params = list(inspect.signature(run_method).parameters.keys())
                    if len(params) >= 3:  # self, ctx, input
                        input_param = params[2]
                        input_type = hints.get(input_param)
                    output_type = hints.get("return")
                except Exception:
                    pass
        else:
            # Function-based task
            try:
                hints = get_type_hints(target)
                params = list(inspect.signature(target).parameters.keys())
                if len(params) >= 2:  # ctx, input
                    input_param = params[1]
                    input_type = hints.get(input_param)
                output_type = hints.get("return")
            except Exception:
                pass

        metadata = TaskMetadata(
            kind=name,
            name=name,
            description=description,
            version=version,
            timeout=timeout,
            retry_policy=retry_policy,
            cancellable=cancellable,
            input_type=input_type,
            output_type=output_type,
        )

        # Attach metadata to the target
        target.__flovyn_task_metadata__ = metadata  # type: ignore[union-attr]

        # Register the task
        register_task(target, metadata)

        return target

    return decorator


def dynamic_task(
    *,
    name: str,
    description: str | None = None,
    version: str | None = None,
    timeout: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
    cancellable: bool = True,
) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
    """Decorator for dynamic (untyped) task definitions.

    Use this when you need runtime flexibility with dict-based inputs/outputs.

    Args:
        name: Unique identifier for the task (required).
        description: Optional human-readable description.
        version: Optional version string.
        timeout: Optional execution timeout.
        retry_policy: Optional retry configuration.
        cancellable: Whether the task supports cancellation (default True).

    Example::

        @dynamic_task(name="dynamic-processor")
        async def process(ctx: TaskContext, input: dict[str, Any]) -> dict[str, Any]:
            return {"processed": True, "data": input}
    """
    return task(
        name=name,
        description=description,
        version=version,
        timeout=timeout,
        retry_policy=retry_policy,
        cancellable=cancellable,
    )


def is_task(obj: Any) -> bool:
    """Check if an object is a registered task."""
    return hasattr(obj, "__flovyn_task_metadata__")


def get_task_metadata(task: Any) -> TaskMetadata | None:
    """Get the metadata for a registered task."""
    if hasattr(task, "__flovyn_task_metadata__"):
        metadata: TaskMetadata = task.__flovyn_task_metadata__
        return metadata
    return None
