import pytest
from ml_tracker.utils import read_json, write_json


@pytest.fixture
def temp_json_file(tmp_path):
    return tmp_path / "test.json"


def test_write_and_read_json(temp_json_file):
    data = {"a": 1}
    assert write_json(temp_json_file, data) is True
    assert read_json(temp_json_file) == data


def test_write_invalid_input_json(temp_json_file):
    with pytest.raises(TypeError):
        write_json(temp_json_file, "not a dict")
