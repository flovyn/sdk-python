# Flovyn Python SDK

Python SDK for [Flovyn](https://flovyn.ai) workflow orchestration with deterministic replay.

## Features

- **Pythonic API** - Decorators, snake_case, context managers, async/await
- **Fully Typed** - Complete type annotations, generics, PEP 561 compliant
- **Type-Safe at Runtime** - Pydantic models for validation and serialization
- **Deterministic Replay** - Workflows can be replayed and resumed after failures
- **Testable** - Mock contexts for unit testing, Testcontainers for E2E testing

## Installation

```bash
pip install flovyn
```

## Quick Start

### Define a Task

```python
from pydantic import BaseModel
from flovyn import task, TaskContext

class GreetInput(BaseModel):
    name: str

class GreetOutput(BaseModel):
    greeting: str

@task(name="greet")
class GreetTask:
    async def run(self, ctx: TaskContext, input: GreetInput) -> GreetOutput:
        return GreetOutput(greeting=f"Hello, {input.name}!")
```

### Define a Workflow

```python
from flovyn import workflow, WorkflowContext

class HelloInput(BaseModel):
    name: str

class HelloOutput(BaseModel):
    message: str

@workflow(name="hello-world")
class HelloWorldWorkflow:
    async def run(self, ctx: WorkflowContext, input: HelloInput) -> HelloOutput:
        # Use deterministic time from context
        timestamp = ctx.current_time().isoformat()

        # Execute a task
        result = await ctx.schedule(
            GreetTask,
            GreetInput(name=input.name),
        )

        return HelloOutput(message=f"{result.greeting} ({timestamp})")
```

### Run the Worker

```python
import asyncio
from flovyn import FlovynClient

async def main():
    client = (
        FlovynClient.builder()
        .server_url("http://localhost:9090")
        .org_id("my-org")
        .queue("default")
        .worker_token("my-token")
        .register_workflow(HelloWorldWorkflow)
        .register_task(GreetTask)
        .build()
    )

    async with client:
        # Start a workflow
        handle = await client.start_workflow(
            HelloWorldWorkflow,
            HelloInput(name="World"),
        )

        # Wait for the result
        result = await handle.result()
        print(result.message)

asyncio.run(main())
```

## Core Concepts

### Determinism

Workflows must be deterministic for replay to work correctly. Always use the context for non-deterministic operations:

```python
# CORRECT - uses deterministic context methods
timestamp = ctx.current_time()
uuid = ctx.random_uuid()
random_value = ctx.random()

# WRONG - will cause determinism violations on replay
timestamp = datetime.now()  # Non-deterministic!
uuid = uuid4()              # Non-deterministic!
```

### WorkflowContext

The workflow context provides deterministic APIs:

- `ctx.current_time()` - Get deterministic current time
- `ctx.random_uuid()` - Generate deterministic UUID
- `ctx.random()` - Get deterministic random number
- `ctx.schedule(Task, input)` - Execute a task and await result
- `ctx.schedule_async(Task, input)` - Execute a task, returns TaskHandle
- `ctx.schedule_workflow(Workflow, input)` - Execute a child workflow and await result
- `ctx.sleep(duration)` - Durable timer
- `ctx.promise(name)` - Wait for external event
- `ctx.get(key)` / `ctx.set(key, value)` - Workflow state
- `ctx.clear(key)` / `ctx.clear_all()` - Clear workflow state
- `ctx.state_keys()` - Get all state keys
- `ctx.run(name, fn)` - Execute side effects (cached on replay)

### TaskContext

The task context provides:

- `ctx.report_progress(0.5, "message")` - Report progress
- `ctx.heartbeat()` - Send heartbeat
- `ctx.is_cancelled` - Check cancellation
- `ctx.attempt` - Current retry attempt

## Testing

### Unit Testing with Mocks

```python
import pytest
from flovyn.testing import MockWorkflowContext

@pytest.mark.asyncio
async def test_hello_workflow():
    ctx = MockWorkflowContext()
    ctx.mock_task_result(GreetTask, GreetOutput(greeting="Hello, Test!"))

    workflow = HelloWorldWorkflow()
    result = await workflow.run(ctx, HelloInput(name="Test"))

    assert "Hello, Test!" in result.message
    assert len(ctx.executed_tasks) == 1
```

### E2E Testing with Testcontainers

```python
import pytest
from flovyn.testing import FlovynTestEnvironment

@pytest.fixture
async def env():
    async with FlovynTestEnvironment() as env:
        env.register_workflow(HelloWorldWorkflow)
        env.register_task(GreetTask)
        await env.start()
        yield env

@pytest.mark.asyncio
async def test_hello_e2e(env):
    handle = await env.start_workflow(HelloWorldWorkflow, HelloInput(name="E2E"))
    result = await handle.result()
    assert "Hello, E2E!" in result.message
```

## Configuration

### Retry Policy

```python
from datetime import timedelta
from flovyn import task, RetryPolicy

@task(
    name="flaky-task",
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=timedelta(seconds=1),
        max_interval=timedelta(minutes=1),
        backoff_coefficient=2.0,
    ),
)
class FlakyTask:
    ...
```

### Timeouts

```python
@workflow(name="my-workflow", timeout=timedelta(hours=1))
class MyWorkflow:
    ...

@task(name="my-task", timeout=timedelta(minutes=5))
class MyTask:
    ...
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,test]"

# Run unit tests
pytest tests/unit

# Run E2E tests (requires Docker)
FLOVYN_E2E_ENABLED=1 pytest tests/e2e -v

# Type checking
mypy flovyn

# Linting
ruff check flovyn tests
```

## License

MIT
