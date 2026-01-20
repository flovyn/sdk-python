"""Task fixtures for E2E tests."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from flovyn import TaskContext, task

# Input/Output models


class EchoTaskInput(BaseModel):
    message: str


class EchoTaskOutput(BaseModel):
    message: str


class AddInput(BaseModel):
    a: int
    b: int


class AddOutput(BaseModel):
    sum: int


class SlowTaskInput(BaseModel):
    duration_ms: int


class SlowTaskOutput(BaseModel):
    completed: bool
    duration_ms: int


class FailingTaskInput(BaseModel):
    fail_count: int
    """Number of times to fail before succeeding."""


class FailingTaskOutput(BaseModel):
    attempts: int


class ProgressTaskInput(BaseModel):
    steps: int


class ProgressTaskOutput(BaseModel):
    completed_steps: int


# Tasks


@task(name="echo-task")
class EchoTask:
    """Simple task that echoes input back."""

    async def run(self, ctx: TaskContext, input: EchoTaskInput) -> EchoTaskOutput:
        return EchoTaskOutput(message=input.message)


@task(name="add-task")
class AddTask:
    """Task that adds two numbers."""

    async def run(self, ctx: TaskContext, input: AddInput) -> AddOutput:
        return AddOutput(sum=input.a + input.b)


@task(name="slow-task")
class SlowTask:
    """Task that sleeps for a configurable duration."""

    async def run(self, ctx: TaskContext, input: SlowTaskInput) -> SlowTaskOutput:
        await ctx.report_progress(0.0, "Starting slow task")

        # Sleep in smaller increments to allow cancellation checks
        remaining_ms = input.duration_ms
        step_ms = min(100, remaining_ms)

        while remaining_ms > 0:
            if ctx.is_cancelled:
                raise ctx.cancellation_error()

            await asyncio.sleep(step_ms / 1000.0)
            remaining_ms -= step_ms

            progress = 1.0 - (remaining_ms / input.duration_ms)
            await ctx.report_progress(progress, f"Sleeping... {remaining_ms}ms remaining")

        await ctx.report_progress(1.0, "Slow task completed")

        return SlowTaskOutput(
            completed=True,
            duration_ms=input.duration_ms,
        )


@task(name="failing-task")
class FailingTask:
    """Task that fails a configurable number of times before succeeding.

    Useful for testing retry logic.
    """

    # Class-level state to track attempts across retries
    _attempts: dict[str, int] = {}

    async def run(self, ctx: TaskContext, input: FailingTaskInput) -> FailingTaskOutput:
        # Track attempts by task execution ID
        key = f"{ctx.task_execution_id}"
        if key not in self._attempts:
            self._attempts[key] = 0

        self._attempts[key] += 1
        current_attempt = self._attempts[key]

        if current_attempt <= input.fail_count:
            from flovyn import TaskFailed

            raise TaskFailed(
                f"Intentional failure {current_attempt}/{input.fail_count}",
                retryable=True,
            )

        # Clean up tracking
        del self._attempts[key]

        return FailingTaskOutput(attempts=current_attempt)


@task(name="progress-task")
class ProgressTask:
    """Task that reports progress in steps."""

    async def run(self, ctx: TaskContext, input: ProgressTaskInput) -> ProgressTaskOutput:
        for i in range(input.steps):
            if ctx.is_cancelled:
                raise ctx.cancellation_error()

            progress = (i + 1) / input.steps
            await ctx.report_progress(progress, f"Step {i + 1}/{input.steps}")

            # Small delay between steps
            await asyncio.sleep(0.01)

        return ProgressTaskOutput(completed_steps=input.steps)


# Streaming Tasks


class StreamingTokenInput(BaseModel):
    tokens: list[str]
    """List of tokens to stream."""


class StreamingTokenOutput(BaseModel):
    token_count: int
    """Number of tokens streamed."""


class StreamingProgressInput(BaseModel):
    steps: int
    """Number of progress steps to stream."""


class StreamingProgressOutput(BaseModel):
    final_progress: float


class StreamingDataInput(BaseModel):
    items: list[dict]
    """Data items to stream."""


class StreamingDataOutput(BaseModel):
    items_streamed: int


class StreamingErrorInput(BaseModel):
    error_message: str
    error_code: str | None = None


class StreamingErrorOutput(BaseModel):
    error_sent: bool


class StreamingAllTypesInput(BaseModel):
    token: str
    progress: float
    data: dict
    error_message: str


class StreamingAllTypesOutput(BaseModel):
    all_types_sent: bool


@task(name="streaming-token-task")
class StreamingTokenTask:
    """Task that streams tokens to connected clients."""

    async def run(self, ctx: TaskContext, input: StreamingTokenInput) -> StreamingTokenOutput:
        for token in input.tokens:
            await ctx.stream_token(token)
            await asyncio.sleep(0.01)  # Small delay between tokens
        return StreamingTokenOutput(token_count=len(input.tokens))


@task(name="streaming-progress-task")
class StreamingProgressTask:
    """Task that streams progress updates."""

    async def run(self, ctx: TaskContext, input: StreamingProgressInput) -> StreamingProgressOutput:
        for i in range(input.steps):
            progress = (i + 1) / input.steps
            details = f"Step {i + 1} of {input.steps}"
            await ctx.stream_progress(progress, details)
            await asyncio.sleep(0.01)
        return StreamingProgressOutput(final_progress=1.0)


@task(name="streaming-data-task")
class StreamingDataTask:
    """Task that streams arbitrary data."""

    async def run(self, ctx: TaskContext, input: StreamingDataInput) -> StreamingDataOutput:
        for item in input.items:
            await ctx.stream_data(item)
            await asyncio.sleep(0.01)
        return StreamingDataOutput(items_streamed=len(input.items))


@task(name="streaming-error-task")
class StreamingErrorTask:
    """Task that streams an error notification."""

    async def run(self, ctx: TaskContext, input: StreamingErrorInput) -> StreamingErrorOutput:
        await ctx.stream_error(input.error_message, input.error_code)
        return StreamingErrorOutput(error_sent=True)


@task(name="streaming-all-types-task")
class StreamingAllTypesTask:
    """Task that streams all event types."""

    async def run(self, ctx: TaskContext, input: StreamingAllTypesInput) -> StreamingAllTypesOutput:
        # Stream a token
        await ctx.stream_token(input.token)

        # Stream progress
        await ctx.stream_progress(input.progress, "Progress update")

        # Stream data
        await ctx.stream_data(input.data)

        # Stream error notification (recoverable)
        await ctx.stream_error(input.error_message, "WARN_001")

        return StreamingAllTypesOutput(all_types_sent=True)
