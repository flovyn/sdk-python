"""Workflow definition decorator and utilities."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import timedelta
from typing import Any, TypeVar, get_type_hints

from flovyn.types import WorkflowMetadata

T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

# Global workflow registry
_workflow_registry: dict[str, tuple[Any, WorkflowMetadata]] = {}


def get_workflow_registry() -> dict[str, tuple[Any, WorkflowMetadata]]:
    """Get the global workflow registry."""
    return _workflow_registry


def register_workflow(workflow: Any, metadata: WorkflowMetadata) -> None:
    """Register a workflow in the global registry."""
    _workflow_registry[metadata.kind] = (workflow, metadata)


def get_workflow_kind(workflow: type[Any] | Callable[..., Any]) -> str:
    """Get the workflow kind from a workflow class or function."""
    if hasattr(workflow, "__flovyn_workflow_metadata__"):
        metadata: WorkflowMetadata = workflow.__flovyn_workflow_metadata__
        return metadata.kind
    raise ValueError(f"Object {workflow} is not a registered workflow. Use @workflow decorator.")


def get_workflow_input_type(workflow: type[Any] | Callable[..., Any]) -> type[Any]:
    """Get the input type from a workflow class or function."""
    if hasattr(workflow, "__flovyn_workflow_metadata__"):
        metadata: WorkflowMetadata = workflow.__flovyn_workflow_metadata__
        return metadata.input_type or Any
    return Any


def get_workflow_output_type(workflow: type[Any] | Callable[..., Any]) -> type[Any]:
    """Get the output type from a workflow class or function."""
    if hasattr(workflow, "__flovyn_workflow_metadata__"):
        metadata: WorkflowMetadata = workflow.__flovyn_workflow_metadata__
        return metadata.output_type or Any
    return Any


def workflow(
    *,
    name: str,
    description: str | None = None,
    version: str | None = None,
    tags: list[str] | None = None,
    timeout: timedelta | None = None,
    cancellable: bool = True,
) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
    """Decorator to define a workflow.

    Can be applied to either a class with a `run` method or an async function.

    Args:
        name: Unique identifier for the workflow (required).
        description: Optional human-readable description.
        version: Optional version string.
        tags: Optional tags for categorization.
        timeout: Optional workflow timeout.
        cancellable: Whether the workflow supports cancellation (default True).

    Examples:
        Class-based workflow::

            @workflow(name="order-processing")
            class OrderWorkflow:
                async def run(self, ctx: WorkflowContext, input: OrderInput) -> OrderResult:
                    validation = await ctx.execute_task(ValidateOrder, input)
                    ...

        Function-based workflow::

            @workflow(name="simple-workflow")
            async def simple_workflow(ctx: WorkflowContext, name: str) -> str:
                greeting = await ctx.execute_task(greet_task, name)
                return f"Workflow completed: {greeting}"
    """

    def decorator(target: type[T] | Callable[..., T]) -> type[T] | Callable[..., T]:
        # Extract type hints for input and output
        input_type: type[Any] | None = None
        output_type: type[Any] | None = None

        if isinstance(target, type):
            # Class-based workflow
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
            # Function-based workflow
            try:
                hints = get_type_hints(target)
                params = list(inspect.signature(target).parameters.keys())
                if len(params) >= 2:  # ctx, input
                    input_param = params[1]
                    input_type = hints.get(input_param)
                output_type = hints.get("return")
            except Exception:
                pass

        metadata = WorkflowMetadata(
            kind=name,
            name=name,
            description=description,
            version=version,
            tags=tags or [],
            timeout=timeout,
            cancellable=cancellable,
            input_type=input_type,
            output_type=output_type,
        )

        # Attach metadata to the target
        target.__flovyn_workflow_metadata__ = metadata  # type: ignore[union-attr]

        # Register the workflow
        register_workflow(target, metadata)

        return target

    return decorator


def dynamic_workflow(
    *,
    name: str,
    description: str | None = None,
    version: str | None = None,
    tags: list[str] | None = None,
    timeout: timedelta | None = None,
    cancellable: bool = True,
) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
    """Decorator for dynamic (untyped) workflow definitions.

    Use this when you need runtime flexibility with dict-based inputs/outputs.

    Args:
        name: Unique identifier for the workflow (required).
        description: Optional human-readable description.
        version: Optional version string.
        tags: Optional tags for categorization.
        timeout: Optional workflow timeout.
        cancellable: Whether the workflow supports cancellation (default True).

    Example::

        @dynamic_workflow(name="dynamic-processor")
        async def process(ctx: WorkflowContext, input: dict[str, Any]) -> dict[str, Any]:
            task_kind = input.get("task_kind", "default-task")
            result = await ctx.execute_task_by_name(task_kind, input.get("payload", {}))
            return {"result": result}
    """
    return workflow(
        name=name,
        description=description,
        version=version,
        tags=tags,
        timeout=timeout,
        cancellable=cancellable,
    )


def is_workflow(obj: Any) -> bool:
    """Check if an object is a registered workflow."""
    return hasattr(obj, "__flovyn_workflow_metadata__")


def get_workflow_metadata(workflow: Any) -> WorkflowMetadata | None:
    """Get the metadata for a registered workflow."""
    if hasattr(workflow, "__flovyn_workflow_metadata__"):
        metadata: WorkflowMetadata = workflow.__flovyn_workflow_metadata__
        return metadata
    return None
