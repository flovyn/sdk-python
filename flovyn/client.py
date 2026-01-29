"""FlovynClient and builder for connecting to Flovyn server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any, TypeVar, cast

from flovyn.exceptions import ConfigurationError
from flovyn.hooks import WorkflowHook
from flovyn.serde import Serializer, get_default_serde
from flovyn.task import get_task_metadata, is_task
from flovyn.types import TaskMetadata, WorkflowHandle, WorkflowMetadata
from flovyn.worker import TaskWorker, WorkflowWorker
from flovyn.workflow import get_workflow_metadata, is_workflow

T = TypeVar("T")
logger = logging.getLogger(__name__)


class FlovynClientBuilder:
    """Builder for configuring and creating a FlovynClient."""

    def __init__(self) -> None:
        self._server_url: str | None = None
        self._org_id: str | None = None
        self._queue: str = "default"
        self._worker_token: str | None = None
        self._worker_identity: str | None = None
        self._workflows: dict[str, tuple[Any, WorkflowMetadata]] = {}
        self._tasks: dict[str, tuple[Any, TaskMetadata]] = {}
        self._hooks: list[WorkflowHook] = []
        self._serializer: Serializer[Any] | None = None
        self._max_concurrent_workflows: int | None = None
        self._max_concurrent_tasks: int | None = None

    def server_url(self, url: str) -> FlovynClientBuilder:
        """Set the Flovyn server URL.

        Args:
            url: Full server URL including scheme (e.g., "https://worker.flovyn.ai" or "http://localhost:9090").

        Returns:
            The builder for chaining.
        """
        self._server_url = url
        return self

    def org_id(self, org_id: str) -> FlovynClientBuilder:
        """Set the organization ID.

        Args:
            org_id: The organization identifier.

        Returns:
            The builder for chaining.
        """
        self._org_id = org_id
        return self

    def queue(self, queue: str) -> FlovynClientBuilder:
        """Set the task queue name.

        Args:
            queue: The queue name (default "default").

        Returns:
            The builder for chaining.
        """
        self._queue = queue
        return self

    def worker_token(self, token: str) -> FlovynClientBuilder:
        """Set the worker authentication token.

        Args:
            token: The worker token for authentication.

        Returns:
            The builder for chaining.
        """
        self._worker_token = token
        return self

    def worker_identity(self, identity: str) -> FlovynClientBuilder:
        """Set the worker identity.

        Args:
            identity: A human-readable worker identifier.

        Returns:
            The builder for chaining.
        """
        self._worker_identity = identity
        return self

    def register_workflow(self, workflow: Any) -> FlovynClientBuilder:
        """Register a workflow definition.

        Args:
            workflow: A workflow class or function decorated with @workflow.

        Returns:
            The builder for chaining.

        Raises:
            ValueError: If the workflow is not properly decorated.
        """
        if not is_workflow(workflow):
            raise ValueError(
                f"Object {workflow} is not a workflow. "
                "Use the @workflow decorator to define workflows."
            )

        metadata = get_workflow_metadata(workflow)
        if metadata is None:
            raise ValueError(f"Could not get metadata for workflow {workflow}")

        self._workflows[metadata.kind] = (workflow, metadata)
        return self

    def register_task(self, task: Any) -> FlovynClientBuilder:
        """Register a task definition.

        Args:
            task: A task class or function decorated with @task.

        Returns:
            The builder for chaining.

        Raises:
            ValueError: If the task is not properly decorated.
        """
        if not is_task(task):
            raise ValueError(
                f"Object {task} is not a task. Use the @task decorator to define tasks."
            )

        metadata = get_task_metadata(task)
        if metadata is None:
            raise ValueError(f"Could not get metadata for task {task}")

        self._tasks[metadata.kind] = (task, metadata)
        return self

    def add_hook(self, hook: WorkflowHook) -> FlovynClientBuilder:
        """Add a workflow lifecycle hook.

        Args:
            hook: A WorkflowHook implementation.

        Returns:
            The builder for chaining.
        """
        self._hooks.append(hook)
        return self

    def default_serde(self, serde: Serializer[Any]) -> FlovynClientBuilder:
        """Set the default serializer.

        Args:
            serde: The serializer to use.

        Returns:
            The builder for chaining.
        """
        self._serializer = serde
        return self

    def max_concurrent_workflows(self, count: int) -> FlovynClientBuilder:
        """Set maximum concurrent workflow tasks.

        Args:
            count: Maximum number of concurrent workflow executions.

        Returns:
            The builder for chaining.
        """
        self._max_concurrent_workflows = count
        return self

    def max_concurrent_tasks(self, count: int) -> FlovynClientBuilder:
        """Set maximum concurrent task executions.

        Args:
            count: Maximum number of concurrent task executions.

        Returns:
            The builder for chaining.
        """
        self._max_concurrent_tasks = count
        return self

    def build(self) -> FlovynClient:
        """Build the FlovynClient.

        Returns:
            A configured FlovynClient instance.

        Raises:
            ConfigurationError: If required configuration is missing.
        """
        if not self._server_url:
            raise ConfigurationError("Server URL is required. Use .server_url(url)")
        if not self._org_id:
            raise ConfigurationError("Organization ID is required. Use .org_id(org_id)")

        return FlovynClient(
            server_url=self._server_url,
            org_id=self._org_id,
            queue=self._queue,
            worker_token=self._worker_token,
            worker_identity=self._worker_identity,
            workflows=self._workflows,
            tasks=self._tasks,
            hooks=self._hooks,
            serializer=self._serializer or get_default_serde(),
            max_concurrent_workflows=self._max_concurrent_workflows,
            max_concurrent_tasks=self._max_concurrent_tasks,
        )


class FlovynClient:
    """Client for connecting to Flovyn server and running workers."""

    def __init__(
        self,
        server_url: str,
        org_id: str,
        queue: str,
        worker_token: str | None,
        worker_identity: str | None,
        workflows: dict[str, tuple[Any, WorkflowMetadata]],
        tasks: dict[str, tuple[Any, TaskMetadata]],
        hooks: list[WorkflowHook],
        serializer: Serializer[Any],
        max_concurrent_workflows: int | None,
        max_concurrent_tasks: int | None,
    ) -> None:
        self._server_url = server_url
        self._org_id = org_id
        self._queue = queue
        self._worker_token = worker_token
        self._worker_identity = worker_identity
        self._workflows = workflows
        self._tasks = tasks
        self._hooks = hooks
        self._serializer = serializer
        self._max_concurrent_workflows = max_concurrent_workflows
        self._max_concurrent_tasks = max_concurrent_tasks

        self._core_worker: Any = None
        self._core_client: Any = None
        self._workflow_worker: WorkflowWorker | None = None
        self._task_worker: TaskWorker | None = None
        self._worker_tasks: list[asyncio.Task[Any]] = []
        self._started = False

    @staticmethod
    def builder() -> FlovynClientBuilder:
        """Create a new client builder.

        Returns:
            A FlovynClientBuilder for configuring the client.
        """
        return FlovynClientBuilder()

    @property
    def server_url(self) -> str:
        """Get the server URL."""
        return self._server_url

    async def start(self) -> None:
        """Start the worker and begin processing.

        This method blocks until shutdown is requested.
        """
        if self._started:
            raise RuntimeError("Client already started")

        self._started = True
        logger.info(f"Starting Flovyn client connecting to {self.server_url}")

        try:
            # Initialize the FFI core worker
            await self._initialize_core()

            # Register with the server
            worker_id = self._core_worker.register()
            logger.info(f"Registered worker with ID: {worker_id}")

            # Start the internal workers
            self._workflow_worker = WorkflowWorker(
                core_worker=self._core_worker,
                workflows=self._workflows,
                tasks=self._tasks,
                serializer=self._serializer,
            )

            self._task_worker = TaskWorker(
                core_worker=self._core_worker,
                tasks=self._tasks,
                serializer=self._serializer,
            )

            # Run workers concurrently
            self._worker_tasks = [
                asyncio.create_task(self._workflow_worker.run()),
                asyncio.create_task(self._task_worker.run()),
            ]

            # Wait for all workers to complete
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        except Exception as e:
            logger.exception(f"Error running client: {e}")
            raise
        finally:
            self._started = False

    async def _initialize_core(self) -> None:
        """Initialize the FFI core worker and client."""
        from flovyn._native.loader import get_native_module

        ffi = get_native_module()

        # Build workflow metadata for FFI
        workflow_metadata = []
        for kind, (_, wf_metadata) in self._workflows.items():
            wf_meta = ffi.WorkflowMetadataFfi(
                kind=kind,
                name=wf_metadata.name,
                description=wf_metadata.description,
                version=wf_metadata.version,
                tags=wf_metadata.tags,
                cancellable=wf_metadata.cancellable,
                timeout_seconds=int(wf_metadata.timeout.total_seconds())
                if wf_metadata.timeout
                else None,
                input_schema=None,
                output_schema=None,
            )
            workflow_metadata.append(wf_meta)

        # Build task metadata for FFI
        task_metadata = []
        for kind, (_, task_meta_info) in self._tasks.items():
            task_meta = ffi.TaskMetadataFfi(
                kind=kind,
                name=task_meta_info.name,
                description=task_meta_info.description,
                version=task_meta_info.version,
                tags=[],
                cancellable=task_meta_info.cancellable,
                timeout_seconds=int(task_meta_info.timeout.total_seconds())
                if task_meta_info.timeout
                else None,
                input_schema=None,
                output_schema=None,
            )
            task_metadata.append(task_meta)

        # Create worker config
        worker_config = ffi.WorkerConfig(
            server_url=self.server_url,
            worker_token=self._worker_token,
            oauth2_credentials=None,
            org_id=self._org_id,
            queue=self._queue,
            worker_identity=self._worker_identity,
            max_concurrent_workflow_tasks=self._max_concurrent_workflows,
            max_concurrent_tasks=self._max_concurrent_tasks,
            workflow_metadata=workflow_metadata,
            task_metadata=task_metadata,
        )

        # Create the core worker
        self._core_worker = ffi.CoreWorker(worker_config)

        # Create the core client (for starting workflows, etc.)
        client_config = ffi.ClientConfig(
            server_url=self.server_url,
            client_token=self._worker_token,
            oauth2_credentials=None,
            org_id=self._org_id,
        )
        self._core_client = ffi.CoreClient(client_config)

    async def shutdown(self) -> None:
        """Request graceful shutdown of the workers."""
        logger.info("Shutting down Flovyn client")

        if self._core_worker:
            self._core_worker.initiate_shutdown()

        if self._workflow_worker:
            self._workflow_worker.shutdown()

        if self._task_worker:
            self._task_worker.shutdown()

        # Wait for worker tasks to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

    async def __aenter__(self) -> FlovynClient:
        """Enter async context manager."""
        # Don't start automatically - let user control when to start
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self.shutdown()

    async def start_workflow(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        input: Any,
        *,
        workflow_id: str | None = None,
        queue: str | None = None,
    ) -> WorkflowHandle[Any]:
        """Start a new workflow execution.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: start_workflow("order-workflow", {"field": "value"})
        - Typed: start_workflow(OrderWorkflow, OrderInput(field="value"))

        Args:
            workflow: The workflow kind (string) or workflow class/function.
            input: The workflow input (dict or Pydantic model).
            workflow_id: Optional custom workflow ID.
            queue: Optional queue override.

        Returns:
            A handle to the running workflow.
        """
        from flovyn.workflow import get_workflow_kind, is_workflow

        if self._core_client is None:
            await self._initialize_core()

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

        response = self._core_client.start_workflow(
            workflow_kind=workflow_kind,
            input=input_bytes,
            queue=queue or self._queue,
            workflow_version=None,
            idempotency_key=workflow_id,
        )

        return self._create_workflow_handle(
            workflow_id=workflow_id or response.workflow_execution_id,
            workflow_execution_id=response.workflow_execution_id,
        )

    def _create_workflow_handle(
        self,
        workflow_id: str,
        workflow_execution_id: str,
    ) -> WorkflowHandle[Any]:
        """Create a workflow handle for a running workflow."""

        async def get_result(timeout: timedelta | None) -> Any:
            # Poll for workflow completion
            deadline = None
            if timeout:
                import time

                deadline = time.time() + timeout.total_seconds()

            while True:
                events = self._core_client.get_workflow_events(workflow_execution_id)

                for event in events:
                    if event.event_type == "WORKFLOW_COMPLETED":
                        # Payload is wrapped in {output: ...} structure
                        import json

                        wrapper = json.loads(event.payload)
                        # Return raw output - caller deserializes as needed
                        return wrapper.get("output")
                    elif event.event_type == "WORKFLOW_EXECUTION_FAILED":
                        error_data = self._serializer.deserialize(event.payload, dict)
                        raise Exception(error_data.get("error", "Workflow failed"))

                if deadline and time.time() > deadline:
                    raise TimeoutError("Workflow did not complete within timeout")

                await asyncio.sleep(0.5)

        async def send_signal(signal_name: str, payload: Any) -> None:
            # Signal sending through the core client
            if self._core_client is None:
                raise RuntimeError("Client not initialized")
            value_bytes = self._serializer.serialize(payload)
            self._core_client.signal_workflow(
                workflow_execution_id=workflow_execution_id,
                signal_name=signal_name,
                signal_value=value_bytes,
            )

        async def execute_query(query_name: str, args: Any) -> Any:
            result_bytes = self._core_client.query_workflow(
                workflow_execution_id,
                query_name,
                self._serializer.serialize(args) if args else b"{}",
            )
            return self._serializer.deserialize(result_bytes, Any)

        async def cancel() -> None:
            # Cancellation would go through the core client
            pass

        return WorkflowHandle(
            workflow_id=workflow_id,
            workflow_execution_id=workflow_execution_id,
            result_getter=get_result,
            signal_sender=send_signal,
            query_executor=execute_query,
            canceller=cancel,
        )

    def _get_promise_id(self, workflow_id: str, promise_name: str) -> str:
        """Look up the promise UUID from workflow events.

        Args:
            workflow_id: The workflow execution ID.
            promise_name: The name of the promise.

        Returns:
            The promise UUID.

        Raises:
            ValueError: If the promise is not found.
        """
        if self._core_client is None:
            raise RuntimeError("Client not initialized")

        events = self._core_client.get_workflow_events(workflow_id)

        for event in events:
            if event.event_type == "PROMISE_CREATED":
                # Parse payload to find promiseName and promiseId
                import json

                payload = json.loads(event.payload.decode("utf-8"))
                if payload.get("promiseName") == promise_name:
                    promise_id = payload.get("promiseId")
                    if not promise_id:
                        raise ValueError(
                            f"Promise '{promise_name}' has no promiseId in event payload"
                        )
                    return cast(str, promise_id)

        raise ValueError(f"Promise '{promise_name}' not found in workflow {workflow_id}")

    async def resolve_promise(
        self,
        workflow_id: str,
        promise_name: str,
        value: Any,
    ) -> None:
        """Resolve an external promise for a workflow.

        Args:
            workflow_id: The workflow execution ID.
            promise_name: The name of the promise to resolve.
            value: The value to resolve the promise with.
        """
        if self._core_client is None:
            await self._initialize_core()

        # Look up the promise UUID from workflow events
        promise_id = self._get_promise_id(workflow_id, promise_name)

        value_bytes = self._serializer.serialize(value)
        self._core_client.resolve_promise(promise_id=promise_id, value=value_bytes)

    async def reject_promise(
        self,
        workflow_id: str,
        promise_name: str,
        error: str,
    ) -> None:
        """Reject an external promise for a workflow.

        Args:
            workflow_id: The workflow execution ID.
            promise_name: The name of the promise to reject.
            error: The error message.
        """
        if self._core_client is None:
            await self._initialize_core()

        # Look up the promise UUID from workflow events
        promise_id = self._get_promise_id(workflow_id, promise_name)

        self._core_client.reject_promise(promise_id=promise_id, error=error)

    async def signal_workflow(
        self,
        workflow_execution_id: str,
        signal_name: str,
        value: Any,
    ) -> int:
        """Send a signal to an existing workflow.

        Args:
            workflow_execution_id: The workflow execution ID.
            signal_name: The name of the signal.
            value: The signal payload.

        Returns:
            The sequence number of the signal event.
        """
        if self._core_client is None:
            await self._initialize_core()

        value_bytes = self._serializer.serialize(value)
        response = self._core_client.signal_workflow(
            workflow_execution_id=workflow_execution_id,
            signal_name=signal_name,
            signal_value=value_bytes,
        )
        return response.signal_event_sequence

    async def signal_with_start_workflow(
        self,
        workflow: str | type[Any] | Callable[..., Any],
        workflow_id: str,
        input: Any,
        signal_name: str,
        signal_value: Any,
        *,
        queue: str | None = None,
    ) -> WorkflowHandle[Any]:
        """Send a signal to an existing workflow, or create a new workflow and send the signal.

        This is an atomic operation - either the workflow exists and receives the signal,
        or a new workflow is created with the signal. This prevents race conditions
        where a workflow might be created between checking for existence and signaling.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: signal_with_start_workflow("order-workflow", "order-123", {"field": "value"}, "signal", payload)
        - Typed: signal_with_start_workflow(OrderWorkflow, "order-123", OrderInput(field="value"), "signal", payload)

        Args:
            workflow: The workflow kind (string) or workflow class/function.
            workflow_id: The workflow ID (used as idempotency key).
            input: The workflow input (dict or Pydantic model).
            signal_name: The name of the signal.
            signal_value: The signal payload.
            queue: Optional queue override.

        Returns:
            A handle to the workflow.
        """
        from flovyn.workflow import get_workflow_kind, is_workflow

        if self._core_client is None:
            await self._initialize_core()

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
        signal_bytes = self._serializer.serialize(signal_value)

        response = self._core_client.signal_with_start_workflow(
            workflow_id=workflow_id,
            workflow_kind=workflow_kind,
            workflow_input=input_bytes,
            queue=queue or self._queue,
            signal_name=signal_name,
            signal_value=signal_bytes,
        )

        return self._create_workflow_handle(
            workflow_id=workflow_id,
            workflow_execution_id=response.workflow_execution_id,
        )

    @property
    def worker_status(self) -> str | None:
        """Get the current worker status.

        Returns:
            Status string ("initializing", "running", "shutting_down") or None if not initialized.
        """
        if self._core_worker is None:
            return None
        return cast(str, self._core_worker.get_status())

    @property
    def worker_uptime_ms(self) -> int | None:
        """Get the worker uptime in milliseconds.

        Returns:
            Uptime in milliseconds or None if not initialized.
        """
        if self._core_worker is None:
            return None
        return cast(int, self._core_worker.get_uptime_ms())

    @property
    def worker_started_at_ms(self) -> int | None:
        """Get the worker start time in milliseconds since Unix epoch.

        Returns:
            Start time in milliseconds or None if not initialized.
        """
        if self._core_worker is None:
            return None
        return cast(int, self._core_worker.get_started_at_ms())

    @property
    def worker_id(self) -> str | None:
        """Get the server-assigned worker ID.

        Returns:
            Worker ID string or None if not registered.
        """
        if self._core_worker is None:
            return None
        return cast(str, self._core_worker.get_worker_id())

    def get_worker_metrics(self) -> Any:
        """Get worker metrics.

        Returns:
            WorkerMetrics record or None if not initialized.
        """
        if self._core_worker is None:
            return None
        return self._core_worker.get_metrics()

    def get_registration_info(self) -> Any:
        """Get worker registration information.

        Returns:
            RegistrationInfo record or None if not registered.
        """
        if self._core_worker is None:
            return None
        return self._core_worker.get_registration_info()

    def get_connection_info(self) -> Any:
        """Get worker connection information.

        Returns:
            ConnectionInfo record.
        """
        if self._core_worker is None:
            return None
        return self._core_worker.get_connection_info()

    # =========================================================================
    # Pause/Resume APIs
    # =========================================================================

    def pause(self, reason: str) -> None:
        """Pause the worker.

        When paused, the worker will not poll for new work but will continue
        processing any in-flight work.

        Args:
            reason: A description of why the worker is being paused.

        Raises:
            RuntimeError: If worker is not initialized.
            Exception: If worker is not in Running state.
        """
        if self._core_worker is None:
            raise RuntimeError("Worker not initialized")
        self._core_worker.pause(reason)

    def resume(self) -> None:
        """Resume the worker.

        Raises:
            RuntimeError: If worker is not initialized.
            Exception: If worker is not in Paused state.
        """
        if self._core_worker is None:
            raise RuntimeError("Worker not initialized")
        self._core_worker.resume()

    @property
    def is_paused(self) -> bool:
        """Check if the worker is paused.

        Returns:
            True if paused, False otherwise.
        """
        if self._core_worker is None:
            return False
        return cast(bool, self._core_worker.is_paused())

    @property
    def is_running(self) -> bool:
        """Check if the worker is running (not paused and not shutting down).

        Returns:
            True if running, False otherwise.
        """
        if self._core_worker is None:
            return False
        return cast(bool, self._core_worker.is_running())

    def get_pause_reason(self) -> str | None:
        """Get the pause reason (if paused).

        Returns:
            The pause reason or None if not paused.
        """
        if self._core_worker is None:
            return None
        return cast(str, self._core_worker.get_pause_reason())

    # =========================================================================
    # Config Accessor APIs
    # =========================================================================

    @property
    def max_concurrent_workflows(self) -> int:
        """Get the maximum concurrent workflows setting.

        Returns:
            The maximum concurrent workflows.
        """
        if self._core_worker is None:
            return self._max_concurrent_workflows or 100
        return cast(int, self._core_worker.get_max_concurrent_workflows())

    @property
    def max_concurrent_tasks(self) -> int:
        """Get the maximum concurrent tasks setting.

        Returns:
            The maximum concurrent tasks.
        """
        if self._core_worker is None:
            return self._max_concurrent_tasks or 100
        return cast(int, self._core_worker.get_max_concurrent_tasks())

    @property
    def queue(self) -> str:
        """Get the queue name.

        Returns:
            The queue name.
        """
        if self._core_worker is None:
            return self._queue
        return cast(str, self._core_worker.get_queue())

    @property
    def org_id(self) -> str:
        """Get the org ID.

        Returns:
            The org ID.
        """
        if self._core_worker is None:
            return self._org_id
        return cast(str, self._core_worker.get_org_id())

    # =========================================================================
    # Lifecycle Events APIs
    # =========================================================================

    def poll_lifecycle_events(self) -> list[Any]:
        """Poll for lifecycle events.

        Returns all events that have occurred since the last poll.
        Events are cleared after being returned.

        Returns:
            List of LifecycleEvent records.
        """
        if self._core_worker is None:
            return []
        return cast(list[Any], self._core_worker.poll_lifecycle_events())

    @property
    def pending_lifecycle_event_count(self) -> int:
        """Get the count of pending lifecycle events.

        Returns:
            Number of pending lifecycle events.
        """
        if self._core_worker is None:
            return 0
        return cast(int, self._core_worker.pending_lifecycle_event_count())
