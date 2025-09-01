import os
import json
import pytest
from ml_tracker.logger import Experiment

@pytest.fixture
def clean_dir(tmp_path):
    """Fixture that gives a clean temporary directory for testing."""
    return tmp_path

def test_log_params_create_file(clean_dir):  # <-- accept fixture here
    exp = Experiment(name="test_exp", base_dir=str(clean_dir))  # str() needed for path

    run_id = "12345"
    params = {"lr": 0.01, "epochs": 10}

    result = exp.log_params(params, run_id)

    # ✅ check method returned True
    assert result is True

    # ✅ check file exists
    json_path = os.path.join(clean_dir, "test_exp", run_id, "params.json")
    assert os.path.exists(json_path)

    # ✅ check file content is correct
    with open(json_path, "r") as f:
        saved = json.load(f)
    assert saved == params
