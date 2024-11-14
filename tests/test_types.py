from few_shots.types import Shot, dump_io_value, id_io_value, is_io_value


def test_is_io_value():
    assert is_io_value("test") is True
    assert is_io_value({"key": "value"}) is True
    assert is_io_value(123) is False
    assert is_io_value(["test"]) is False


def test_data_key():
    data = {"b": 2, "a": 1}
    assert dump_io_value(data) == '{"a":1,"b":2}'


def test_data_hash(snapshot):
    data = {"test": "value"}
    assert id_io_value(data) == snapshot


def test_init_with_id():
    shot = Shot({"input": "test"}, {"output": "result"}, "test_id")
    assert shot.inputs == {"input": "test"}
    assert shot.outputs == {"output": "result"}
    assert shot.id == "test_id"


def test_init_without_id():
    inputs = {"input": "test"}
    shot = Shot(inputs, {"output": "result"})
    assert shot.id == id_io_value(inputs)


def test_key_property():
    inputs = {"input": "test"}
    shot = Shot(inputs, {"output": "result"})
    assert shot.key == dump_io_value(inputs)
