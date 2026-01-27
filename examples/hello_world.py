"""Hello World example - Simple workflow and task demonstration."""

import asyncio

from pydantic import BaseModel

from flovyn import FlovynClient, TaskContext, WorkflowContext, task, workflow


# Define input/output models using Pydantic
class GreetInput(BaseModel):
    name: str


class GreetOutput(BaseModel):
    greeting: str


class HelloInput(BaseModel):
    name: str


class HelloOutput(BaseModel):
    message: str


# Define a simple task
@task(name="greet")
class GreetTask:
    """A task that generates a greeting."""

    async def run(self, ctx: TaskContext, input: GreetInput) -> GreetOutput:
        # Report progress
        await ctx.report_progress(0.5, "Generating greeting")

        greeting = f"Hello, {input.name}!"

        await ctx.report_progress(1.0, "Done")

        return GreetOutput(greeting=greeting)


# Define a simple workflow
@workflow(name="hello-world")
class HelloWorldWorkflow:
    """A workflow that greets someone."""

    async def run(self, ctx: WorkflowContext, input: HelloInput) -> HelloOutput:
        # Use deterministic operations from context
        timestamp = ctx.current_time().isoformat()
        ctx.logger.info(f"Starting workflow at {timestamp}")

        # Execute a task
        result = await ctx.schedule(
            GreetTask,
            GreetInput(name=input.name),
        )

        return HelloOutput(message=f"{result.greeting} (processed at {timestamp})")


async def main() -> None:
    """Run the hello world example."""
    # Build the client
    client = (
        FlovynClient.builder()
        .server_address("http://localhost:9090")
        .org_id("my-org")
        .queue("hello-world-queue")
        .worker_token("my-worker-token")
        .register_workflow(HelloWorldWorkflow)
        .register_task(GreetTask)
        .build()
    )

    print("Starting Flovyn client...")

    # Run the client - this will block and process workflows
    # In a real application, you might want to run this in a background task
    # while starting workflows from another service

    # For demonstration, we'll run in a context manager
    async with client:
        # Start a workflow
        handle = await client.start_workflow(
            HelloWorldWorkflow,
            HelloInput(name="World"),
        )

        print(f"Started workflow: {handle.workflow_execution_id}")

        # Wait for the result
        result = await handle.result()
        print(f"Workflow result: {result.message}")


if __name__ == "__main__":
    asyncio.run(main())
