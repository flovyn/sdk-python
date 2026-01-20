"""Serialization utilities for Flovyn SDK."""

from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, get_origin

from pydantic import BaseModel

T = TypeVar("T")


class Serializer(ABC, Generic[T]):
    """Abstract base class for serializers."""

    @abstractmethod
    def serialize(self, value: T) -> bytes:
        """Serialize a value to bytes.

        Args:
            value: The value to serialize.

        Returns:
            The serialized bytes.
        """
        ...

    @abstractmethod
    def deserialize(self, data: bytes, type_hint: type[T]) -> T:
        """Deserialize bytes to a value.

        Args:
            data: The bytes to deserialize.
            type_hint: The expected type of the value.

        Returns:
            The deserialized value.
        """
        ...


class JsonSerde(Serializer[Any]):
    """JSON serializer using standard library json module."""

    def serialize(self, value: Any) -> bytes:
        """Serialize a value to JSON bytes."""
        return json.dumps(value, default=self._default_encoder).encode("utf-8")

    def deserialize(self, data: bytes, type_hint: type[T]) -> T:
        """Deserialize JSON bytes to a value."""
        parsed = json.loads(data.decode("utf-8"))
        return self._convert_to_type(parsed, type_hint)

    def _default_encoder(self, obj: Any) -> Any:
        """Default encoder for non-JSON-serializable objects."""
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _convert_to_type(self, value: Any, type_hint: type[T]) -> T:
        """Convert a parsed JSON value to the expected type."""
        # Handle None
        if value is None:
            return None  # type: ignore[return-value]

        # Handle basic types
        if type_hint in (str, int, float, bool, type(None)):
            return value  # type: ignore[no-any-return]

        # Handle dict/list without specific type hints
        if type_hint is dict or type_hint is list:
            return value  # type: ignore[no-any-return]

        # Handle Any
        if type_hint is Any:
            return value  # type: ignore[no-any-return]

        # Handle generic types (dict[K, V], list[T], etc.)
        origin = get_origin(type_hint)
        if origin is dict:
            return value  # type: ignore[no-any-return]
        if origin is list:
            return value  # type: ignore[no-any-return]

        # Handle dataclasses
        if dataclasses.is_dataclass(type_hint):
            if isinstance(value, dict):
                return type_hint(**value)
            return value  # type: ignore[no-any-return]

        # Default: return as-is
        return value  # type: ignore[no-any-return]


class PydanticSerde(Serializer[Any]):
    """Pydantic-based serializer with full validation."""

    def serialize(self, value: Any) -> bytes:
        """Serialize a value to JSON bytes using Pydantic."""
        if isinstance(value, BaseModel):
            return value.model_dump_json().encode("utf-8")
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            return json.dumps(dataclasses.asdict(value)).encode("utf-8")
        else:
            return json.dumps(value).encode("utf-8")

    def deserialize(self, data: bytes, type_hint: type[T]) -> T:
        """Deserialize JSON bytes to a value using Pydantic."""
        json_str = data.decode("utf-8")

        # Handle Pydantic models
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint.model_validate_json(json_str)

        # Handle dataclasses
        if dataclasses.is_dataclass(type_hint):
            parsed = json.loads(json_str)
            return type_hint(**parsed)

        # Handle basic types and containers
        parsed = json.loads(json_str)

        if type_hint in (str, int, float, bool, type(None), Any):
            return parsed  # type: ignore[no-any-return]

        origin = get_origin(type_hint)
        if origin in (dict, list):
            return parsed  # type: ignore[no-any-return]

        return parsed  # type: ignore[no-any-return]


class AutoSerde(Serializer[Any]):
    """Auto-detecting serializer that chooses the best strategy.

    Detection order:
    1. Pydantic BaseModel → use model_dump_json/model_validate_json
    2. dataclass → use dataclasses.asdict/constructor
    3. dict/list/primitives → use json module
    """

    def __init__(self) -> None:
        self._pydantic_serde = PydanticSerde()
        self._json_serde = JsonSerde()

    def serialize(self, value: Any) -> bytes:
        """Serialize a value using auto-detection."""
        if isinstance(value, BaseModel):
            return self._pydantic_serde.serialize(value)
        return self._json_serde.serialize(value)

    def deserialize(self, data: bytes, type_hint: type[T]) -> T:
        """Deserialize bytes using auto-detection based on type hint."""
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return self._pydantic_serde.deserialize(data, type_hint)
        return self._json_serde.deserialize(data, type_hint)


# Default serializer instance
_default_serde: Serializer[Any] = AutoSerde()


def get_default_serde() -> Serializer[Any]:
    """Get the default serializer."""
    return _default_serde


def set_default_serde(serde: Serializer[Any]) -> None:
    """Set the default serializer."""
    global _default_serde
    _default_serde = serde


def serialize(value: Any, serde: Serializer[Any] | None = None) -> bytes:
    """Serialize a value using the specified or default serializer.

    Args:
        value: The value to serialize.
        serde: Optional serializer to use (uses default if not specified).

    Returns:
        The serialized bytes.
    """
    if serde is None:
        serde = _default_serde
    return serde.serialize(value)


def deserialize(data: bytes, type_hint: type[T], serde: Serializer[Any] | None = None) -> T:
    """Deserialize bytes using the specified or default serializer.

    Args:
        data: The bytes to deserialize.
        type_hint: The expected type of the value.
        serde: Optional serializer to use (uses default if not specified).

    Returns:
        The deserialized value.
    """
    if serde is None:
        serde = _default_serde
    return serde.deserialize(data, type_hint)  # type: ignore[no-any-return]
