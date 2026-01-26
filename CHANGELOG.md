# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Breaking Changes

### WorkflowContext API Changes

The WorkflowContext API has been updated for consistency with SDK-Kotlin and SDK-TypeScript.

#### Method Renames

| Old Name | New Name |
|----------|----------|
| `execute_task()` | `schedule()` |
| `schedule_task()` | `schedule_async()` |
| `execute_workflow()` | `schedule_workflow()` |
| `schedule_workflow()` | `schedule_workflow_async()` |
| `wait_for_promise()` | `promise()` |
| `get_state()` | `get()` |
| `set_state()` | `set()` |
| `clear_state()` | `clear()` |

#### New Methods

- `clear_all()` - Clear all workflow state
- `state_keys()` - Get all state keys

### Migration Guide

**Before:**
```python
@workflow(name="my-workflow")
class MyWorkflow:
    async def run(self, ctx: WorkflowContext, input: MyInput) -> MyOutput:
        # Execute task
        result = await ctx.execute_task(MyTask, TaskInput(value=input.value))

        # Schedule task (non-blocking)
        handle = ctx.schedule_task(MyTask, TaskInput(value=input.value))
        result = await handle.result()

        # Execute child workflow
        child_result = await ctx.execute_workflow(ChildWorkflow, ChildInput())

        # Wait for promise
        approval = await ctx.wait_for_promise("approval")

        # State management
        ctx.set_state("key", "value")
        value = ctx.get_state("key")
        ctx.clear_state("key")

        return MyOutput(result=result.value)
```

**After:**
```python
@workflow(name="my-workflow")
class MyWorkflow:
    async def run(self, ctx: WorkflowContext, input: MyInput) -> MyOutput:
        # Execute task
        result = await ctx.schedule(MyTask, TaskInput(value=input.value))

        # Schedule task (non-blocking)
        handle = ctx.schedule_async(MyTask, TaskInput(value=input.value))
        result = await handle.result()

        # Execute child workflow
        child_result = await ctx.schedule_workflow(ChildWorkflow, ChildInput())

        # Wait for promise
        approval = await ctx.promise("approval")

        # State management
        ctx.set("key", "value")
        value = ctx.get("key")
        ctx.clear("key")
        ctx.clear_all()  # New: clear all state
        keys = ctx.state_keys()  # New: get all keys

        return MyOutput(result=result.value)
```

### Rationale

These changes align the Python SDK with SDK-Kotlin (the reference implementation) and SDK-TypeScript:

1. **Shorter method names** - `schedule()` instead of `execute_task()` is more concise
2. **Consistent naming** - All SDKs now use the same method names
3. **Clearer semantics** - `schedule_async()` clearly indicates non-blocking behavior
4. **State API** - `get()`/`set()` follows common key-value store conventions
