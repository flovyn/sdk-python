"""Unit tests for serialization."""

from dataclasses import dataclass

from pydantic import BaseModel

from flovyn.serde import AutoSerde, JsonSerde, PydanticSerde


class SampleModel(BaseModel):
    name: str
    value: int


@dataclass
class SampleDataclass:
    name: str
    value: int


class TestJsonSerde:
    def test_serialize_dict(self) -> None:
        serde = JsonSerde()
        data = {"key": "value", "number": 42}
        result = serde.serialize(data)
        assert result == b'{"key": "value", "number": 42}'

    def test_deserialize_dict(self) -> None:
        serde = JsonSerde()
        data = b'{"key": "value", "number": 42}'
        result = serde.deserialize(data, dict)
        assert result == {"key": "value", "number": 42}

    def test_roundtrip_list(self) -> None:
        serde = JsonSerde()
        data = [1, 2, 3, "four"]
        serialized = serde.serialize(data)
        result = serde.deserialize(serialized, list)
        assert result == data


class TestPydanticSerde:
    def test_serialize_model(self) -> None:
        serde = PydanticSerde()
        model = SampleModel(name="test", value=42)
        result = serde.serialize(model)
        assert b'"name":"test"' in result
        assert b'"value":42' in result

    def test_deserialize_model(self) -> None:
        serde = PydanticSerde()
        data = b'{"name": "test", "value": 42}'
        result = serde.deserialize(data, SampleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_roundtrip_model(self) -> None:
        serde = PydanticSerde()
        model = SampleModel(name="roundtrip", value=123)
        serialized = serde.serialize(model)
        result = serde.deserialize(serialized, SampleModel)
        assert result == model


class TestAutoSerde:
    def test_auto_detects_pydantic(self) -> None:
        serde = AutoSerde()
        model = SampleModel(name="auto", value=99)
        serialized = serde.serialize(model)
        result = serde.deserialize(serialized, SampleModel)
        assert result == model

    def test_auto_detects_dict(self) -> None:
        serde = AutoSerde()
        data = {"key": "value"}
        serialized = serde.serialize(data)
        result = serde.deserialize(serialized, dict)
        assert result == data
