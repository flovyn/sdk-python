"""Order Processing example - Multi-step workflow with state and error handling."""

import asyncio
from datetime import timedelta
from enum import Enum

from pydantic import BaseModel

from flovyn import (
    FlovynClient,
    RetryPolicy,
    TaskContext,
    TaskFailed,
    WorkflowContext,
    task,
    workflow,
)

# Models


class OrderStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    PAID = "paid"
    FULFILLED = "fulfilled"
    FAILED = "failed"


class OrderItem(BaseModel):
    product_id: str
    quantity: int
    price: float


class OrderInput(BaseModel):
    order_id: str
    customer_id: str
    items: list[OrderItem]


class OrderResult(BaseModel):
    order_id: str
    status: OrderStatus
    confirmation_id: str | None = None
    error: str | None = None


class ValidationInput(BaseModel):
    order_id: str
    items: list[OrderItem]


class ValidationResult(BaseModel):
    valid: bool
    issues: list[str] = []


class PaymentInput(BaseModel):
    order_id: str
    customer_id: str
    amount: float


class PaymentResult(BaseModel):
    transaction_id: str
    status: str


class FulfillmentInput(BaseModel):
    order_id: str
    items: list[OrderItem]


class FulfillmentResult(BaseModel):
    tracking_number: str
    estimated_delivery: str


# Tasks


@task(
    name="validate-order",
    timeout=timedelta(seconds=30),
)
class ValidateOrderTask:
    """Validate an order before processing."""

    async def run(self, ctx: TaskContext, input: ValidationInput) -> ValidationResult:
        await ctx.report_progress(0.0, "Starting validation")

        issues = []

        # Simulate validation checks
        if not input.items:
            issues.append("Order has no items")

        for item in input.items:
            if item.quantity <= 0:
                issues.append(f"Invalid quantity for product {item.product_id}")
            if item.price <= 0:
                issues.append(f"Invalid price for product {item.product_id}")

        await ctx.report_progress(1.0, "Validation complete")

        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
        )


@task(
    name="process-payment",
    timeout=timedelta(minutes=2),
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
    ),
)
class ProcessPaymentTask:
    """Process payment for an order."""

    async def run(self, ctx: TaskContext, input: PaymentInput) -> PaymentResult:
        await ctx.report_progress(0.0, "Initiating payment")

        # Simulate payment processing
        await asyncio.sleep(0.1)

        if ctx.is_cancelled:
            raise ctx.cancellation_error()

        await ctx.report_progress(0.5, "Payment authorized")

        # Generate a transaction ID
        transaction_id = f"txn-{input.order_id}-{ctx.attempt}"

        await ctx.report_progress(1.0, "Payment completed")

        return PaymentResult(
            transaction_id=transaction_id,
            status="completed",
        )


@task(
    name="fulfill-order",
    timeout=timedelta(minutes=5),
)
class FulfillOrderTask:
    """Fulfill an order by preparing shipment."""

    async def run(self, ctx: TaskContext, input: FulfillmentInput) -> FulfillmentResult:
        await ctx.report_progress(0.0, "Starting fulfillment")

        # Simulate fulfillment
        total_items = sum(item.quantity for item in input.items)
        for i in range(total_items):
            if ctx.is_cancelled:
                raise ctx.cancellation_error()

            progress = (i + 1) / total_items
            await ctx.report_progress(progress, f"Packing item {i + 1}/{total_items}")
            await asyncio.sleep(0.01)

        return FulfillmentResult(
            tracking_number=f"TRACK-{input.order_id}",
            estimated_delivery="2-3 business days",
        )


# Workflow


@workflow(
    name="order-processing",
    description="Process a customer order through validation, payment, and fulfillment",
    timeout=timedelta(hours=1),
)
class OrderProcessingWorkflow:
    """Process an order through the complete lifecycle."""

    async def run(self, ctx: WorkflowContext, input: OrderInput) -> OrderResult:
        order_id = input.order_id
        ctx.logger.info(f"Starting order processing for {order_id}")

        # Track order status in workflow state
        await ctx.set("status", OrderStatus.PENDING.value)

        try:
            # Step 1: Validate order
            ctx.logger.info("Validating order")
            validation = await ctx.schedule(
                ValidateOrderTask,
                ValidationInput(order_id=order_id, items=input.items),
            )

            if not validation.valid:
                await ctx.set("status", OrderStatus.FAILED.value)
                return OrderResult(
                    order_id=order_id,
                    status=OrderStatus.FAILED,
                    error=f"Validation failed: {', '.join(validation.issues)}",
                )

            await ctx.set("status", OrderStatus.VALIDATED.value)

            # Step 2: Process payment
            ctx.logger.info("Processing payment")
            total_amount = sum(item.price * item.quantity for item in input.items)

            payment = await ctx.schedule(
                ProcessPaymentTask,
                PaymentInput(
                    order_id=order_id,
                    customer_id=input.customer_id,
                    amount=total_amount,
                ),
            )

            await ctx.set("status", OrderStatus.PAID.value)
            await ctx.set("transaction_id", payment.transaction_id)

            # Step 3: Fulfill order
            ctx.logger.info("Fulfilling order")
            fulfillment = await ctx.schedule(
                FulfillOrderTask,
                FulfillmentInput(order_id=order_id, items=input.items),
            )

            await ctx.set("status", OrderStatus.FULFILLED.value)
            await ctx.set("tracking_number", fulfillment.tracking_number)

            ctx.logger.info(f"Order {order_id} completed successfully")

            return OrderResult(
                order_id=order_id,
                status=OrderStatus.FULFILLED,
                confirmation_id=payment.transaction_id,
            )

        except TaskFailed as e:
            ctx.logger.error(f"Order {order_id} failed: {e.message}")
            await ctx.set("status", OrderStatus.FAILED.value)

            return OrderResult(
                order_id=order_id,
                status=OrderStatus.FAILED,
                error=e.message,
            )


async def main() -> None:
    """Run the order processing example."""
    client = (
        FlovynClient.builder()
        .server_url("http://localhost:9090")
        .org_id("my-org")
        .queue("order-queue")
        .worker_token("my-worker-token")
        .register_workflow(OrderProcessingWorkflow)
        .register_task(ValidateOrderTask)
        .register_task(ProcessPaymentTask)
        .register_task(FulfillOrderTask)
        .build()
    )

    print("Starting order processing worker...")

    async with client:
        # Start an order workflow
        handle = await client.start_workflow(
            OrderProcessingWorkflow,
            OrderInput(
                order_id="ORD-12345",
                customer_id="CUST-001",
                items=[
                    OrderItem(product_id="PROD-A", quantity=2, price=29.99),
                    OrderItem(product_id="PROD-B", quantity=1, price=49.99),
                ],
            ),
        )

        print(f"Started order workflow: {handle.workflow_execution_id}")

        result = await handle.result()
        print(f"Order result: {result.model_dump_json(indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
