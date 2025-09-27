import json
import pytest
from ml_tracker.core.run import RUN


@pytest.fixture
def temp_run_dir(tmp_path):
    return tmp_path / "run001"


def test_log_and_dis_params(temp_run_dir):
    run = RUN("run001", temp_run_dir)
    params = {"lr": 0.01, "epochs": 10}
    assert run.log_params(params) is True

    loaded = run.get_params()
    assert params == loaded


def test_parms_invalid_input(temp_run_dir):
    run = RUN("run001", temp_run_dir)
    with pytest.raises(ValueError):
        run.log_params("not a dict")


def test_log_and_dis_metrics(temp_run_dir):
    run = RUN("run001", temp_run_dir)
    metrics = {"accuracy_score": 0.9876534}
    assert run.log_metrics(metrics) is True

    loaded = run.get_metrics()
    assert metrics == loaded


def test_metrics_invalid_input(temp_run_dir):
    run = RUN("run001", temp_run_dir)
    with pytest.raises(ValueError):
        run.log_metrics("not a dict")
