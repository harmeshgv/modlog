import json
import pytest
from ml_tracker.core.run import RUN
import numpy as np

RUN_ID = "run001"
PARAMS = {"lr": 0.01, "epochs": 10}
METRICS = {"accuracy_score": 0.9876534}
TRUE_VALUES = np.array([1, 2, 3, 4])
PRED_VALUES = np.array([1.1, 1.9, 3.2, 3.8])


@pytest.fixture
def temp_run_dir(tmp_path):
    return tmp_path / "run001"


def test_log_and_dis_params(temp_run_dir):
    run = RUN(RUN_ID, temp_run_dir)
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


def test_set_run_type(temp_run_dir):
    run = RUN("run001", temp_run_dir, task_type="regression")
    assert run.task_type == "regression"


def test_log_predictions_with_task_type(temp_run_dir):
    run = RUN("run001", temp_run_dir, task_type="regression")

    assert run.log_predictions(TRUE_VALUES, PRED_VALUES, set_metrics=True) is True

    loaded_y_true, loaded_y_preds = run.get_predictions(as_array=True)
    assert np.array_equal(loaded_y_true, TRUE_VALUES)
    assert np.array_equal(loaded_y_preds, PRED_VALUES)

    metrics = run.get_metrics()
    assert "regression" in metrics
    assert "MSE" in metrics["regression"]
    assert "MAE" in metrics["regression"]
    assert "R2" in metrics["regression"]
