import pytest
from ml_tracker.regression.run_utils import _prepare_metrics
import numpy as np


def test_prepare_metrics():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    metrics = _prepare_metrics(y_true, y_pred)

    assert "regression" in metrics
    assert "MSE" in metrics["regression"]
    assert "MAE" in metrics["regression"]
    assert "R2" in metrics["regression"]

    # Optional: check numerical values roughly
    assert metrics["regression"]["MSE"] >= 0
    assert metrics["regression"]["MAE"] >= 0
