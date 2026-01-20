"""Pytest configuration and fixtures."""

import asyncio
import logging

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


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
