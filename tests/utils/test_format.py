from best_shot.types import Shot
from best_shot.utils.format import shots_to_messages, to_str


def test_to_str_string():
    assert to_str("test") == "test"


def test_to_str_dict():
    data = {"key": "value"}
    assert to_str(data) == '{"key":"value"}'


def test_shots_to_messages_empty():
    assert shots_to_messages([]) == []


def test_shots_to_messages_single_shot():
    shot = Shot(inputs="test input", outputs="test output")
    expected = [
        {"role": "user", "content": "test input"},
        {"role": "assistant", "content": "test output"},
    ]
    assert shots_to_messages([shot]) == expected


def test_shots_to_messages_multiple_shots():
    shots = [
        Shot(inputs="input 1", outputs="output 1"),
        Shot(inputs="input 2", outputs="output 2"),
    ]
    expected = [
        {"role": "user", "content": "input 1"},
        {"role": "assistant", "content": "output 1"},
        {"role": "user", "content": "input 2"},
        {"role": "assistant", "content": "output 2"},
    ]
    assert shots_to_messages(shots) == expected


def test_shots_to_messages_dict_inputs():
    shot = Shot(
        inputs={"context": "some context", "prompt": "test input"},
        outputs="test output",
    )
    expected = [
        {
            "content": '{"context":"some context","prompt":"test input"}',
            "role": "user",
        },
        {"content": "test output", "role": "assistant"},
    ]
    assert shots_to_messages([shot]) == expected


def test_shots_to_messages_dict_outputs():
    shot = Shot(
        inputs="test input",
        outputs={"metadata": {"confidence": 0.9}, "response": "test output"},
    )
    expected = [
        {"content": "test input", "role": "user"},
        {
            "content": '{"metadata":{"confidence":0.9},"response":"test output"}',
            "role": "assistant",
        },
    ]
    assert shots_to_messages([shot]) == expected


def test_shots_to_messages_dict_both():
    shot = Shot(inputs={"prompt": "test input"}, outputs={"response": "test output"})
    expected = [
        {"content": '{"prompt":"test input"}', "role": "user"},
        {"content": '{"response":"test output"}', "role": "assistant"},
    ]
    assert shots_to_messages([shot]) == expected
