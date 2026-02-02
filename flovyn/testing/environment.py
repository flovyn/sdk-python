"""E2E test environment with Testcontainers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from datetime import timedelta
from typing import Any

from flovyn.client import FlovynClient
from flovyn.task import get_task_metadata, is_task
from flovyn.types import TaskMetadata, WorkflowHandle, WorkflowMetadata
from flovyn.workflow import get_workflow_metadata, is_workflow

logger = logging.getLogger(__name__)


class FlovynTestEnvironment:
    """Test environment that manages Flovyn server via Testcontainers.

    This provides a complete test environment with PostgreSQL, NATS, and
    Flovyn server containers for E2E testing.

    Example::

        @pytest.fixture
        async def env():
            async with FlovynTestEnvironment() as env:
                env.register_workflow(MyWorkflow)
                env.register_task(MyTask)
                await env.start()
                yield env

        @pytest.mark.asyncio
        async def test_my_workflow(env: FlovynTestEnvironment):
            handle = await env.start_workflow(MyWorkflow, {"input": "value"})
            result = await handle.result(timeout=timedelta(seconds=30))
            assert result["status"] == "completed"
    """

    # Test timeout constants
    TEST_TIMEOUT = timedelta(seconds=60)
    DEFAULT_AWAIT_TIMEOUT = timedelta(seconds=30)
    WORKER_REGISTRATION_DELAY = 0.5  # seconds (reduced for faster tests)

    def __init__(
        self,
        org_id: str | None = None,
        queue: str | None = None,
        worker_token: str | None = None,
    ) -> None:
        """Initialize the test environment.

        Args:
            org_id: The organization ID to use (defaults to harness org_id).
            queue: The queue name (auto-generated if not provided).
            worker_token: The worker authentication token (defaults to harness worker_token).
        """
        self._org_id = org_id
        self._queue = queue or f"test-queue-{uuid.uuid4().hex[:8]}"
        self._worker_token = worker_token
        self._workflows: dict[str, tuple[Any, WorkflowMetadata]] = {}
        self._tasks: dict[str, tuple[Any, TaskMetadata]] = {}
        self._client: FlovynClient | None = None
        self._worker_task: asyncio.Task[Any] | None = None
        self._harness: TestHarness | None = None
        self._started = False

    def register_workflow(self, workflow: Any) -> FlovynTestEnvironment:
        """Register a workflow for testing.

        Args:
            workflow: A workflow class or function decorated with @workflow.

        Returns:
            The environment for chaining.
        """
        if not is_workflow(workflow):
            raise ValueError(f"Object {workflow} is not a workflow")

        metadata = get_workflow_metadata(workflow)
        if metadata is None:
            raise ValueError(f"Could not get metadata for workflow {workflow}")

        self._workflows[metadata.kind] = (workflow, metadata)
        return self

    def register_task(self, task: Any) -> FlovynTestEnvironment:
        """Register a task for testing.

        Args:
            task: A task class or function decorated with @task.

        Returns:
            The environment for chaining.
        """
        if not is_task(task):
            raise ValueError(f"Object {task} is not a task")

        metadata = get_task_metadata(task)
        if metadata is None:
            raise ValueError(f"Could not get metadata for task {task}")

        self._tasks[metadata.kind] = (task, metadata)
        return self

    async def start(self) -> None:
        """Start the test environment and worker."""
        if self._started:
            return

        # Get or create the test harness
        self._harness = await get_test_harness()

        # Use harness credentials if not explicitly set
        org_id = self._org_id or self._harness.org_id
        worker_token = self._worker_token or self._harness.worker_token

        # Build the client
        builder = FlovynClient.builder()
        builder.server_url(f"http://{self._harness.grpc_host}:{self._harness.grpc_port}")
        builder.org_id(org_id)
        builder.queue(self._queue)
        builder.worker_token(worker_token)

        for workflow, _ in self._workflows.values():
            builder.register_workflow(workflow)

        for task, _ in self._tasks.values():
            builder.register_task(task)

        self._client = builder.build()

        # Start the worker in the background
        self._worker_task = asyncio.create_task(self._client.start())

        # Wait for worker to be ready using lifecycle events
        await self._await_worker_ready()
        self._started = True
        logger.info(f"Test environment started with queue: {self._queue}")

    async def _await_worker_ready(self, timeout: float = 10.0) -> None:
        """Wait for worker to be registered and ready.

        Uses worker status to detect registration without consuming lifecycle events.

        Args:
            timeout: Maximum time to wait for registration.
        """
        import time

        start = time.time()
        registered = False

        while time.time() - start < timeout:
            # Give the worker a chance to initialize
            await asyncio.sleep(0.1)

            # Check if we're registered via status (don't consume lifecycle events)
            if self._client and self._client.worker_status == "running":
                registered = True
                break

        if not registered:
            logger.warning(f"Worker registration timeout after {timeout}s")

        # Small delay for server-side processing
        await asyncio.sleep(self.WORKER_REGISTRATION_DELAY)

    async def stop(self) -> None:
        """Stop the test environment."""
        if self._client:
            await self._client.shutdown()

        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

        self._started = False
        logger.info("Test environment stopped")

    async def start_workflow(
        self,
        workflow: str | type[Any] | Any,
        input: Any,
        *,
        workflow_id: str | None = None,
    ) -> WorkflowHandle[Any]:
        """Start a workflow for testing.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: start_workflow("order-workflow", {"order_id": "123"})
        - Typed: start_workflow(OrderWorkflow, OrderInput(order_id="123"))

        Args:
            workflow: The workflow kind (string) or workflow class/function to execute.
            input: The workflow input (dict or serializable object).
            workflow_id: Optional custom workflow ID.

        Returns:
            A handle to the running workflow.
        """
        if not self._started or self._client is None:
            raise RuntimeError("Test environment not started. Call start() first.")

        return await self._client.start_workflow(
            workflow,
            input,
            workflow_id=workflow_id,
            queue=self._queue,
        )

    async def await_completion(
        self,
        handle: WorkflowHandle[Any],
        timeout: timedelta | None = None,
    ) -> Any:
        """Wait for a workflow to complete.

        Args:
            handle: The workflow handle.
            timeout: Maximum time to wait (defaults to DEFAULT_AWAIT_TIMEOUT).

        Returns:
            The workflow result.
        """
        timeout = timeout or self.DEFAULT_AWAIT_TIMEOUT
        return await handle.result(timeout=timeout)

    async def start_and_await(
        self,
        workflow: str | type[Any] | Any,
        input: Any,
        *,
        workflow_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> Any:
        """Start a workflow and wait for it to complete.

        Supports both string-based (distributed) and typed (single-server) APIs:
        - String-based: start_and_await("order-workflow", {"order_id": "123"})
        - Typed: start_and_await(OrderWorkflow, OrderInput(order_id="123"))

        Args:
            workflow: The workflow kind (string) or workflow class/function to execute.
            input: The workflow input (dict or serializable object).
            workflow_id: Optional custom workflow ID.
            timeout: Maximum time to wait.

        Returns:
            The workflow result.
        """
        handle = await self.start_workflow(workflow, input, workflow_id=workflow_id)
        return await self.await_completion(handle, timeout=timeout)

    async def resolve_promise(
        self,
        handle: WorkflowHandle[Any],
        promise_name: str,
        value: Any,
    ) -> None:
        """Resolve a promise for a running workflow.

        Args:
            handle: The workflow handle.
            promise_name: The name of the promise to resolve.
            value: The value to resolve the promise with.
        """
        if not self._started or self._client is None:
            raise RuntimeError("Test environment not started. Call start() first.")

        await self._client.resolve_promise(
            workflow_id=handle.workflow_id,
            promise_name=promise_name,
            value=value,
        )

    async def reject_promise(
        self,
        handle: WorkflowHandle[Any],
        promise_name: str,
        error: str,
    ) -> None:
        """Reject a promise for a running workflow.

        Args:
            handle: The workflow handle.
            promise_name: The name of the promise to reject.
            error: The error message.
        """
        if not self._started or self._client is None:
            raise RuntimeError("Test environment not started. Call start() first.")

        await self._client.reject_promise(
            workflow_id=handle.workflow_id,
            promise_name=promise_name,
            error=error,
        )

    async def signal_workflow(
        self,
        handle: WorkflowHandle[Any],
        signal_name: str,
        value: Any,
    ) -> int:
        """Send a signal to a running workflow.

        Args:
            handle: The workflow handle.
            signal_name: The name of the signal.
            value: The signal payload.

        Returns:
            The sequence number of the signal event.
        """
        if not self._started or self._client is None:
            raise RuntimeError("Test environment not started. Call start() first.")

        return await self._client.signal_workflow(
            workflow_execution_id=handle.workflow_execution_id,
            signal_name=signal_name,
            value=value,
        )

    async def signal_with_start_workflow(
        self,
        workflow: str | type[Any],
        workflow_id: str,
        input: Any,
        signal_name: str,
        signal_value: Any,
    ) -> WorkflowHandle[Any]:
        """Send a signal to an existing workflow, or create a new workflow and send the signal.

        This is an atomic operation - either the workflow exists and receives the signal,
        or a new workflow is created with the signal.

        Args:
            workflow: The workflow kind (string) or workflow class.
            workflow_id: The workflow ID (used as idempotency key).
            input: The workflow input.
            signal_name: The name of the signal.
            signal_value: The signal payload.

        Returns:
            A handle to the workflow.
        """
        if not self._started or self._client is None:
            raise RuntimeError("Test environment not started. Call start() first.")

        return await self._client.signal_with_start_workflow(
            workflow=workflow,
            workflow_id=workflow_id,
            input=input,
            signal_name=signal_name,
            signal_value=signal_value,
        )

    @property
    def client(self) -> FlovynClient | None:
        """Get the underlying FlovynClient."""
        return self._client

    async def __aenter__(self) -> FlovynTestEnvironment:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self.stop()

    @property
    def queue(self) -> str:
        """Get the queue name."""
        return self._queue

    @property
    def org_id(self) -> str | None:
        """Get the organization ID."""
        return self._org_id

    @property
    def worker_status(self) -> str | None:
        """Get the current worker status.

        Returns:
            Status string ("initializing", "running", "shutting_down") or None if not started.
        """
        if self._client is None:
            return None
        return self._client.worker_status

    @property
    def worker_uptime_ms(self) -> int | None:
        """Get the worker uptime in milliseconds.

        Returns:
            Uptime in milliseconds or None if not started.
        """
        if self._client is None:
            return None
        return self._client.worker_uptime_ms

    @property
    def worker_started_at_ms(self) -> int | None:
        """Get the worker start time in milliseconds since Unix epoch.

        Returns:
            Start time in milliseconds or None if not started.
        """
        if self._client is None:
            return None
        return self._client.worker_started_at_ms

    @property
    def worker_id(self) -> str | None:
        """Get the server-assigned worker ID.

        Returns:
            Worker ID string or None if not registered.
        """
        if self._client is None:
            return None
        return self._client.worker_id

    def get_worker_metrics(self) -> Any:
        """Get worker metrics.

        Returns:
            WorkerMetrics record or None if not started.
        """
        if self._client is None:
            return None
        return self._client.get_worker_metrics()

    def get_registration_info(self) -> Any:
        """Get worker registration information.

        Returns:
            RegistrationInfo record or None if not registered.
        """
        if self._client is None:
            return None
        return self._client.get_registration_info()

    def get_connection_info(self) -> Any:
        """Get worker connection information.

        Returns:
            ConnectionInfo record or None if not started.
        """
        if self._client is None:
            return None
        return self._client.get_connection_info()

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
            RuntimeError: If environment is not started.
            Exception: If worker is not in Running state.
        """
        if self._client is None:
            raise RuntimeError("Test environment not started. Call start() first.")
        self._client.pause(reason)

    def resume(self) -> None:
        """Resume the worker.

        Raises:
            RuntimeError: If environment is not started.
            Exception: If worker is not in Paused state.
        """
        if self._client is None:
            raise RuntimeError("Test environment not started. Call start() first.")
        self._client.resume()

    @property
    def is_paused(self) -> bool:
        """Check if the worker is paused.

        Returns:
            True if paused, False otherwise.
        """
        if self._client is None:
            return False
        return self._client.is_paused

    @property
    def is_running(self) -> bool:
        """Check if the worker is running (not paused and not shutting down).

        Returns:
            True if running, False otherwise.
        """
        if self._client is None:
            return False
        return self._client.is_running

    def get_pause_reason(self) -> str | None:
        """Get the pause reason (if paused).

        Returns:
            The pause reason or None if not paused.
        """
        if self._client is None:
            return None
        return self._client.get_pause_reason()

    # =========================================================================
    # Config Accessor APIs
    # =========================================================================

    @property
    def max_concurrent_workflows(self) -> int:
        """Get the maximum concurrent workflows setting.

        Returns:
            The maximum concurrent workflows.
        """
        if self._client is None:
            return 100
        return self._client.max_concurrent_workflows

    @property
    def max_concurrent_tasks(self) -> int:
        """Get the maximum concurrent tasks setting.

        Returns:
            The maximum concurrent tasks.
        """
        if self._client is None:
            return 100
        return self._client.max_concurrent_tasks

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
        if self._client is None:
            return []
        return self._client.poll_lifecycle_events()

    @property
    def pending_lifecycle_event_count(self) -> int:
        """Get the count of pending lifecycle events.

        Returns:
            Number of pending lifecycle events.
        """
        if self._client is None:
            return 0
        return self._client.pending_lifecycle_event_count


class TestHarness:
    """Manages Testcontainers for E2E tests.

    This class handles the lifecycle of Docker containers needed for testing:
    - PostgreSQL for database
    - NATS for messaging
    - Flovyn server

    The harness is typically shared across all tests in a session.
    """

    def __init__(self) -> None:
        self._postgres_container: Any = None
        self._nats_container: Any = None
        self._server_container: Any = None
        self._config_file: Any = None
        self._started = False

        # Connection details
        self.grpc_host: str = "localhost"
        self.grpc_port: int = 9090
        self.http_host: str = "localhost"
        self.http_port: int = 8000

        # Generate unique org and credentials for this test session
        self.org_id: str = str(uuid.uuid4())
        self.org_slug: str = f"test-{uuid.uuid4().hex[:8]}"
        self.api_key: str = f"flovyn_sk_test_{uuid.uuid4().hex[:16]}"
        self.worker_token: str = f"flovyn_wk_test_{uuid.uuid4().hex[:16]}"

    async def start(self) -> None:
        """Start all containers."""
        if self._started:
            return

        logger.info("Starting test harness containers...")

        try:
            from testcontainers.core.container import DockerContainer
            from testcontainers.postgres import PostgresContainer
        except ImportError as err:
            raise ImportError(
                "testcontainers is required for E2E tests. Install with: pip install testcontainers"
            ) from err

        # Start PostgreSQL
        self._postgres_container = PostgresContainer(
            image="postgres:18-alpine",
            username="flovyn",
            password="flovyn",
            dbname="flovyn",
        )
        self._postgres_container.start()
        postgres_port = self._postgres_container.get_exposed_port(5432)
        logger.info(f"PostgreSQL started at localhost:{postgres_port}")

        # Start NATS
        self._nats_container = DockerContainer("nats:latest")
        self._nats_container.with_exposed_ports(4222)
        self._nats_container.start()
        nats_port = self._nats_container.get_exposed_port(4222)
        logger.info(f"NATS started at localhost:{nats_port}")

        # Create temp config file for Flovyn server
        import tempfile

        config_content = self._generate_server_config()
        self._config_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".toml",
            prefix="flovyn-test-config-",
            delete=False,
        )
        self._config_file.write(config_content)
        self._config_file.flush()
        config_path = self._config_file.name
        logger.info(f"Created config file at {config_path}")

        # Start Flovyn server
        server_image = "rg.fr-par.scw.cloud/flovyn/flovyn-server:latest"

        self._server_container = DockerContainer(server_image)
        self._server_container.with_exposed_ports(8000, 9090)

        # Use environment variables for database and NATS (Rust server expects these)
        self._server_container.with_env(
            "DATABASE_URL", f"postgres://flovyn:flovyn@host.docker.internal:{postgres_port}/flovyn"
        )
        self._server_container.with_env("NATS__ENABLED", "true")
        self._server_container.with_env("NATS__URL", f"nats://host.docker.internal:{nats_port}")
        self._server_container.with_env("SERVER_PORT", "8000")
        self._server_container.with_env("GRPC_SERVER_PORT", "9090")

        # Mount config file
        self._server_container.with_volume_mapping(config_path, "/app/config.toml", "ro")
        self._server_container.with_env("CONFIG_FILE", "/app/config.toml")

        # Add extra_hosts for Docker networking on Linux (maps host.docker.internal to host)
        import platform

        if platform.system() == "Linux":
            # For podman/docker on Linux, we need to add host gateway
            self._server_container.with_kwargs(extra_hosts={"host.docker.internal": "host-gateway"})

        self._server_container.start()

        self.grpc_host = self._server_container.get_container_host_ip()
        self.grpc_port = int(self._server_container.get_exposed_port(9090))
        self.http_host = self._server_container.get_container_host_ip()
        self.http_port = int(self._server_container.get_exposed_port(8000))

        logger.info(
            f"Flovyn server started at gRPC={self.grpc_host}:{self.grpc_port}, HTTP={self.http_host}:{self.http_port}"
        )

        # Wait for server to be healthy
        await self._wait_for_health()

        self._started = True
        logger.info("Test harness ready")

    async def _wait_for_health(self, timeout: float = 30.0) -> None:
        """Wait for the server to be healthy."""
        import time

        try:
            import httpx
        except ImportError as err:
            raise ImportError(
                "httpx is required for E2E tests. Install with: pip install httpx"
            ) from err

        url = f"http://{self.http_host}:{self.http_port}/_/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=5.0)
                    if response.status_code == 200:
                        logger.info("Server health check passed")
                        return
            except Exception:
                pass

            await asyncio.sleep(0.5)

        raise TimeoutError(f"Server did not become healthy within {timeout} seconds")

    def _generate_server_config(self) -> str:
        """Generate TOML config for the Flovyn server (Rust format)."""
        return f"""
# Pre-configured organizations
[[orgs]]
id = "{self.org_id}"
name = "Test Organization"
slug = "{self.org_slug}"
tier = "FREE"

# Authentication configuration
[auth]
enabled = true

# Static API keys
[auth.static_api_key]
keys = [
    {{ key = "{self.api_key}", org_id = "{self.org_id}", principal_type = "User", principal_id = "api:test", role = "ADMIN" }},
    {{ key = "{self.worker_token}", org_id = "{self.org_id}", principal_type = "Worker", principal_id = "worker:test" }}
]

# Endpoint authentication
[auth.endpoints.http]
authenticators = ["static_api_key"]
authorizer = "cedar"

[auth.endpoints.grpc]
authenticators = ["static_api_key"]
authorizer = "cedar"
"""

    async def stop(self) -> None:
        """Stop all containers."""
        keep_containers = os.environ.get("FLOVYN_TEST_KEEP_CONTAINERS", "").lower() in ("1", "true")

        if keep_containers:
            logger.info("Keeping containers running (FLOVYN_TEST_KEEP_CONTAINERS=1)")
            return

        logger.info("Stopping test harness containers...")

        if self._server_container:
            try:
                self._server_container.stop()
            except Exception as e:
                logger.warning(f"Error stopping server container: {e}")

        if self._nats_container:
            try:
                self._nats_container.stop()
            except Exception as e:
                logger.warning(f"Error stopping NATS container: {e}")

        if self._postgres_container:
            try:
                self._postgres_container.stop()
            except Exception as e:
                logger.warning(f"Error stopping PostgreSQL container: {e}")

        # Clean up config file
        if self._config_file:
            try:
                import os as os_module

                os_module.unlink(self._config_file.name)
            except Exception as e:
                logger.warning(f"Error removing config file: {e}")

        self._started = False
        logger.info("Test harness stopped")


# Global harness instance
_global_harness: TestHarness | None = None
_harness_lock = asyncio.Lock()


async def get_test_harness() -> TestHarness:
    """Get or create the global test harness.

    The harness is shared across all tests to avoid starting/stopping
    containers for each test.

    Returns:
        The global TestHarness instance.
    """
    global _global_harness

    async with _harness_lock:
        if _global_harness is None:
            _global_harness = TestHarness()
            await _global_harness.start()

    return _global_harness


async def cleanup_test_harness() -> None:
    """Clean up the global test harness.

    Call this at the end of the test session.
    """
    global _global_harness

    if _global_harness is not None:
        await _global_harness.stop()
        _global_harness = None
