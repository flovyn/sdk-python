"""Pytest configuration and fixtures."""

import asyncio
import logging
import os

import pytest

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests (require Docker)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip E2E tests unless explicitly enabled."""
    skip_e2e = pytest.mark.skip(reason="E2E tests require Docker. Set FLOVYN_E2E_ENABLED=1 to run.")

    for item in items:
        # Skip E2E tests unless explicitly enabled
        if "e2e" in item.keywords and os.environ.get("FLOVYN_E2E_ENABLED", "").lower() not in (
            "1",
            "true",
        ):
            item.add_marker(skip_e2e)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
