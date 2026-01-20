"""Workflow lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class WorkflowStartedEvent:
    """Event fired when a workflow starts."""

    workflow_execution_id: str
    workflow_kind: str
    workflow_id: str | None
    started_at: datetime
    input: Any | None = None


@dataclass
class WorkflowCompletedEvent:
    """Event fired when a workflow completes successfully."""

    workflow_execution_id: str
    workflow_kind: str
    workflow_id: str | None
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    output: Any | None = None


@dataclass
class WorkflowFailedEvent:
    """Event fired when a workflow fails."""

    workflow_execution_id: str
    workflow_kind: str
    workflow_id: str | None
    started_at: datetime
    failed_at: datetime
    duration_ms: int
    error: str
    stack_trace: str | None = None


class WorkflowHook:
    """Base class for workflow lifecycle hooks.

    Subclass this to implement custom behavior on workflow lifecycle events.

    Example::

        class MetricsHook(WorkflowHook):
            async def on_workflow_started(self, event: WorkflowStartedEvent) -> None:
                metrics.increment("workflow.started", tags={"kind": event.workflow_kind})

            async def on_workflow_completed(self, event: WorkflowCompletedEvent) -> None:
                metrics.increment("workflow.completed", tags={"kind": event.workflow_kind})
                metrics.timing("workflow.duration", event.duration_ms)

            async def on_workflow_failed(self, event: WorkflowFailedEvent) -> None:
                metrics.increment("workflow.failed", tags={"kind": event.workflow_kind})
    """

    async def on_workflow_started(self, event: WorkflowStartedEvent) -> None:
        """Called when a workflow starts.

        Args:
            event: The workflow started event.
        """
        pass

    async def on_workflow_completed(self, event: WorkflowCompletedEvent) -> None:
        """Called when a workflow completes successfully.

        Args:
            event: The workflow completed event.
        """
        pass

    async def on_workflow_failed(self, event: WorkflowFailedEvent) -> None:
        """Called when a workflow fails.

        Args:
            event: The workflow failed event.
        """
        pass
