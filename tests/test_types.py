import pytest
from hypothesis import given
from hypothesis.strategies import dictionaries, text

from few_shots.types import Shot, dump_io_value, id_io_value, is_io_value


@given(value=text())
def test_str_is_io_value(value: str):
    assert is_io_value(value)


@given(value=dictionaries(text(), text()))
def test_dict_is_io_value(value: dict[str, str]):
    assert is_io_value(value)


def test_data_key():
    data = {"b": 2, "a": 1}
    assert dump_io_value(data) == '{"a":1,"b":2}'


def test_data_hash(snapshot):
    data = {"test": "value"}
    assert id_io_value(data) == snapshot


@given(inputs=dictionaries(text(), text()))
def test_shot_custom_id(inputs: dict[str, str]):
    outputs = {"output": "result"}
    shot = Shot(inputs, outputs, "test_id")
    assert shot.inputs == inputs
    assert shot.outputs == outputs
    assert shot.key == dump_io_value(inputs)
    assert shot.id == "test_id"


@given(inputs=dictionaries(text(), text()))
def test_shot_auto_id(inputs: dict[str, str]):
    outputs = {"output": "result"}
    shot = Shot(inputs, outputs)
    assert shot.key == dump_io_value(inputs)
    assert shot.id == id_io_value(inputs)
