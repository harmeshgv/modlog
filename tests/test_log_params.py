import json
import pytest
from ml_tracker.run import RUN
from ml_tracker.experiment import EXPERIMENT

def test_log_params_with_dict(tmp_path):
    run = RUN("001", run_path=tmp_path)
    params = {"lr": 0.01, "epochs": 10}
    assert run.log_params(params) is True

    log_path = tmp_path / "log_params.json"
    assert log_path.exists()
    data = json.loads(log_path.read_text())
    assert data == params

def test_log_params_with_object(tmp_path):
    class DummyModel:
        def get_params(self):
            return {"alpha": 0.1, "beta": 5}

    run = RUN("002", run_path=tmp_path)
    model = DummyModel()
    assert run.log_params(model) is True

    log_path = tmp_path / "log_params.json"
    data = json.loads(log_path.read_text())
    assert data == model.get_params()

def test_log_params_invalid_input(tmp_path):
    run = RUN("003", run_path=tmp_path)
    with pytest.raises(ValueError):
        run.log_params([1, 2, 3])

def test_log_params_complete_test(tmp_path):
    params = {"lr": 0.01, "epochs": 10}
    exp = EXPERIMENT("exp_001", base_dir=tmp_path)
    run_001 = exp.start_run("run_001")
    run_001.log_params(params)

    log_path = tmp_path / "exp_001" / "run_001" / "log_params.json"
    assert log_path.exists()
    data = json.loads(log_path.read_text())
    assert data == params
