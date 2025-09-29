import os
from ml_tracker.utils.io import write_json, read_json
from ml_tracker.regression.run_utils import (
    _prepare_metrics as prepare_regression_metrics,
)
from ml_tracker.classification.run_utils import (
    _prepare_metrics as prepare_classification_metrics,
)
from pathlib import Path
import numpy as np


class RUN:
    def __init__(self, run_id: str, run_path: Path, task_type: str | None = None):
        self.run_path = run_path
        self.run_id = run_id
        self.task_type = task_type

    def log_params(self, params: dict) -> bool:
        """
        Description :

        Args:

        Returns:

        Raises:
        """
        if not isinstance(params, dict):
            if hasattr(params, "get_params"):
                params = params.get_params()
            else:
                raise ValueError("params must be a dict or have a get_params() method")

        os.makedirs(self.run_path, exist_ok=True)
        log_path = os.path.join(self.run_path, "log_params.json")
        return write_json(log_path, params)

    def get_params(self) -> dict:
        """
        Description :

        Args:

        Returns:

        Raises:
        """
        log_path = os.path.join(self.run_path, "log_params.json")
        return read_json(log_path)

    def log_metrics(self, metrics: dict) -> bool:
        """
        Description :

        Args:

        Returns:

        Raises:
        """
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dict")

        os.makedirs(self.run_path, exist_ok=True)
        metrics_path = os.path.join(self.run_path, "log_metrics.json")
        return write_json(metrics_path, metrics)

    def get_metrics(self) -> dict:
        """
        Description :

        Args:

        Returns:

        Raises:
        """
        metrics_path = os.path.join(self.run_path, "log_metrics.json")
        return read_json(metrics_path)

    def log_data_info(self, data: dict) -> bool:
        """
        Description :

        Args:

        Returns:

        Raises:
        """

        os.makedirs(self.run_path, exist_ok=True)

        data_path = os.path.join(self.run_path, "data.json")
        return write_json(data_path, data)

    def get_data_info(self) -> dict:
        """
        Description :

        Args:

        Returns:

        Raises:
        """
        data_path = os.path.join(self.run_path, "data.json")
        return read_json(data_path)

    def log_predictions(
        self, y_true, y_preds, task_type: str | None = None, set_metrics: bool = False
    ) -> bool:

        predictions_path = os.path.join(self.run_path, "predictions.json")
        y_true_serializable = y_true.tolist() if hasattr(y_true, "tolist") else y_true
        y_preds_serializable = (
            y_preds.tolist() if hasattr(y_preds, "tolist") else y_preds
        )
        if self.task_type is None and task_type is None:
            raise ValueError(
                "task_type must be specified when creating the run or passed to this method"
            )

        if self.task_type.lower() == "regression" and set_metrics == True:
            metrics = prepare_regression_metrics(y_true, y_preds)

        elif self.task_type.lower() == "classification" and set_metrics == True:
            metrics = prepare_classification_metrics(y_true, y_preds)

        return (
            write_json(
                predictions_path,
                {"y_true": y_true_serializable, "y_preds": y_preds_serializable},
            )
            and self.log_metrics(metrics)
            if set_metrics
            else True
        )

    def get_predictions(self, as_array: bool = False) -> dict:
        predictions_path = os.path.join(self.run_path, "predictions.json")
        predictions = read_json(predictions_path)
        if as_array:
            return np.array(predictions["y_true"]), np.array(predictions["y_preds"])
        return predictions
