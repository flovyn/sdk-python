"""Internal worker implementations for processing activations."""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Callable
from typing import Any

from flovyn.context import TaskContextImpl, WorkflowContextImpl
from flovyn.exceptions import TaskCancelled, TaskFailed, WorkflowCancelled, WorkflowSuspended
from flovyn.serde import Serializer, get_default_serde
from flovyn.types import TaskMetadata, WorkflowMetadata

logger = logging.getLogger(__name__)


class WorkflowWorker:
    """Internal worker that processes workflow activations from the Rust core."""

    def __init__(
        self,
        core_worker: Any,  # CoreWorker from FFI
        workflows: dict[str, tuple[Any, WorkflowMetadata]],
        tasks: dict[str, tuple[Any, TaskMetadata]],
        serializer: Serializer[Any] | None = None,
    ) -> None:
        self._core_worker = core_worker
        self._workflows = workflows
        self._tasks = tasks
        self._serializer = serializer or get_default_serde()
        self._shutdown = False
        self._workflow_instances: dict[str, Any] = {}

    async def run(self) -> None:
        """Run the workflow worker loop."""
        logger.info("Starting workflow worker")

        while not self._shutdown and not self._core_worker.is_shutdown_requested():
            try:
                # Poll for activation from Rust core
                activation = self._core_worker.poll_workflow_activation()

                if activation is None:
                    # No work available, sleep briefly
                    await asyncio.sleep(0.01)
                    continue

                # Process the activation
                await self._process_activation(activation)

            except Exception as e:
                if "ShuttingDown" in str(type(e).__name__) or "shutting down" in str(e).lower():
                    logger.info("Worker shutting down")
                    break
                logger.exception(f"Error in workflow worker loop: {e}")
                await asyncio.sleep(0.1)

        logger.info("Workflow worker stopped")

    async def _process_activation(self, activation: Any) -> None:
        """Process a workflow activation."""
        workflow_kind = activation.workflow_kind
        context = activation.context

        # Get the workflow definition
        if workflow_kind not in self._workflows:
            logger.error(f"Unknown workflow kind: {workflow_kind}")
            self._complete_workflow_failed(
                context,
                f"Unknown workflow kind: {workflow_kind}",
            )
            return

        workflow_def, metadata = self._workflows[workflow_kind]

        # Create the Python context wrapping the FFI context
        ctx = WorkflowContextImpl(
            ffi_context=context,
            serializer=self._serializer,
            task_registry={k: v[1] for k, v in self._tasks.items()},
            workflow_registry={k: v[1] for k, v in self._workflows.items()},
        )

        try:
            # Process all jobs in the activation
            status = await self._execute_workflow(
                workflow_def,
                metadata,
                ctx,
                activation.input,
                activation.jobs,
            )

            # Send completion back to Rust core
            self._complete_workflow(context, status)

        except Exception as e:
            logger.exception(f"Error processing workflow activation: {e}")
            self._complete_workflow_failed(
                context,
                str(e),
                stack_trace=traceback.format_exc(),
            )

    async def _execute_workflow(
        self,
        workflow_def: Any,
        metadata: WorkflowMetadata,
        ctx: WorkflowContextImpl,
        input_bytes: bytes,
        jobs: list[Any],
    ) -> Any:
        """Execute the workflow logic."""
        from flovyn._native.loader import get_native_module

        ffi = get_native_module()

        # Process jobs first (timers, task completions, etc.)
        for job in jobs:
            self._process_job(ctx, job)

        # Get or create workflow instance
        execution_id = ctx.workflow_execution_id
        if execution_id not in self._workflow_instances:
            if isinstance(workflow_def, type):
                self._workflow_instances[execution_id] = workflow_def()
            else:
                self._workflow_instances[execution_id] = workflow_def

        workflow_instance = self._workflow_instances[execution_id]

        # Deserialize input
        input_type = metadata.input_type or Any
        if input_bytes:
            input_value = self._serializer.deserialize(bytes(input_bytes), input_type)
        else:
            input_value = None

        try:
            # Execute the workflow
            if isinstance(workflow_def, type):
                result = await workflow_instance.run(ctx, input_value)
            else:
                result = await workflow_instance(ctx, input_value)

            # Workflow completed successfully
            output_bytes = self._serializer.serialize(result)
            return ffi.WorkflowCompletionStatus.COMPLETED(output=output_bytes)

        except WorkflowSuspended:
            # Workflow needs to wait for pending operations
            return ffi.WorkflowCompletionStatus.SUSPENDED()

        except WorkflowCancelled as e:
            return ffi.WorkflowCompletionStatus.CANCELLED(reason=str(e))

        except Exception as e:
            return ffi.WorkflowCompletionStatus.FAILED(
                error=str(e),
            )

    def _process_job(self, ctx: WorkflowContextImpl, job: Any) -> None:
        """Process a single activation job."""
        # Jobs are handled internally by the FFI context during replay
        # The context tracks pending operations and resolves them when jobs arrive
        pass

    def _complete_workflow(self, context: Any, status: Any) -> None:
        """Complete the workflow activation."""
        self._core_worker.complete_workflow_activation(context, status)

    def _complete_workflow_failed(
        self,
        context: Any,
        error: str,
        stack_trace: str | None = None,
    ) -> None:
        """Complete the workflow activation with failure."""
        from flovyn._native.loader import get_native_module

        ffi = get_native_module()
        status = ffi.WorkflowCompletionStatus.FAILED(error=error)
        self._core_worker.complete_workflow_activation(context, status)

    def shutdown(self) -> None:
        """Signal the worker to shut down."""
        self._shutdown = True

    @property
    def status(self) -> str:
        """Get the current worker status.

        Returns:
            Status string: "initializing", "running", or "shutting_down"
        """
        return self._core_worker.get_status()


class TaskWorker:
    """Internal worker that processes task activations."""

    def __init__(
        self,
        core_worker: Any,  # CoreWorker from FFI
        tasks: dict[str, tuple[Any, TaskMetadata]],
        serializer: Serializer[Any] | None = None,
    ) -> None:
        self._core_worker = core_worker
        self._tasks = tasks
        self._serializer = serializer or get_default_serde()
        self._shutdown = False
        self._cancelled_tasks: set[str] = set()

    async def run(self) -> None:
        """Run the task worker loop."""
        logger.info("Starting task worker")

        while not self._shutdown and not self._core_worker.is_shutdown_requested():
            try:
                # Poll for task activation from Rust core
                activation = self._core_worker.poll_task_activation()

                if activation is None:
                    # No work available, sleep briefly
                    await asyncio.sleep(0.01)
                    continue

                # Process the task activation
                await self._process_activation(activation)

            except Exception as e:
                if "ShuttingDown" in str(type(e).__name__) or "shutting down" in str(e).lower():
                    logger.info("Task worker shutting down")
                    break
                logger.exception(f"Error in task worker loop: {e}")
                await asyncio.sleep(0.1)

        logger.info("Task worker stopped")

    async def _process_activation(self, activation: Any) -> None:
        """Process a task activation."""
        task_kind = activation.task_kind
        task_execution_id = activation.task_execution_id

        # Get the task definition
        if task_kind not in self._tasks:
            logger.error(f"Unknown task kind: {task_kind}")
            self._complete_task_failed(
                task_execution_id,
                f"Unknown task kind: {task_kind}",
                retryable=False,
            )
            return

        task_def, metadata = self._tasks[task_kind]

        # Get the FFI task context (if available)
        ffi_context = getattr(activation, "context", None)

        # Create the task context
        ctx = TaskContextImpl(
            task_execution_id=task_execution_id,
            attempt=activation.attempt,
            workflow_execution_id=activation.workflow_execution_id,
            cancellation_checker=lambda: task_execution_id in self._cancelled_tasks,
            progress_reporter=self._create_progress_reporter(task_execution_id),
            heartbeat_sender=self._create_heartbeat_sender(task_execution_id),
            ffi_context=ffi_context,
        )

        try:
            # Deserialize input
            input_type = metadata.input_type or Any
            if activation.input:
                input_value = self._serializer.deserialize(bytes(activation.input), input_type)
            else:
                input_value = None

            # Execute the task
            if isinstance(task_def, type):
                task_instance = task_def()
                result = await task_instance.run(ctx, input_value)
            else:
                result = await task_def(ctx, input_value)

            # Task completed successfully
            output_bytes = self._serializer.serialize(result)
            self._complete_task_success(task_execution_id, output_bytes)

        except TaskCancelled:
            self._complete_task_cancelled(task_execution_id)

        except TaskFailed as e:
            self._complete_task_failed(task_execution_id, str(e), retryable=e.retryable)

        except Exception as e:
            logger.exception(f"Task {task_execution_id} failed with error: {e}")
            self._complete_task_failed(task_execution_id, str(e), retryable=True)

    def _create_progress_reporter(
        self, task_execution_id: str
    ) -> Callable[[float, str | None], Any]:
        """Create a progress reporter for a task."""

        async def report_progress(progress: float, message: str | None = None) -> None:
            # Progress reporting can be implemented via heartbeat with metadata
            pass

        return report_progress

    def _create_heartbeat_sender(self, task_execution_id: str) -> Callable[[], Any]:
        """Create a heartbeat sender for a task."""

        async def send_heartbeat() -> None:
            # Heartbeat is implicit in the task polling mechanism
            pass

        return send_heartbeat

    def _complete_task_success(self, task_execution_id: str, output: bytes) -> None:
        """Complete a task with success."""
        from flovyn._native.loader import get_native_module

        ffi = get_native_module()
        completion = ffi.TaskCompletion.COMPLETED(
            task_execution_id=task_execution_id,
            output=output,
        )
        self._core_worker.complete_task(completion)

    def _complete_task_failed(self, task_execution_id: str, error: str, retryable: bool) -> None:
        """Complete a task with failure."""
        from flovyn._native.loader import get_native_module

        ffi = get_native_module()
        completion = ffi.TaskCompletion.FAILED(
            task_execution_id=task_execution_id,
            error=error,
            retryable=retryable,
        )
        self._core_worker.complete_task(completion)

    def _complete_task_cancelled(self, task_execution_id: str) -> None:
        """Complete a task with cancellation."""
        from flovyn._native.loader import get_native_module

        ffi = get_native_module()
        completion = ffi.TaskCompletion.CANCELLED(
            task_execution_id=task_execution_id,
        )
        self._core_worker.complete_task(completion)

    def cancel_task(self, task_execution_id: str) -> None:
        """Request cancellation of a task."""
        self._cancelled_tasks.add(task_execution_id)

    def shutdown(self) -> None:
        """Signal the worker to shut down."""
        self._shutdown = True

    @property
    def status(self) -> str:
        """Get the current worker status.

        Returns:
            Status string: "initializing", "running", or "shutting_down"
        """
        return self._core_worker.get_status()
