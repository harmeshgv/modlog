import os
from ml_tracker.utils.io import write_json, read_json
from pathlib import Path


class RUN:
    def __init__(self, run_id: str, run_path: Path):
        self.run_path = run_path
        self.run_id = run_id

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
        write_json(log_path, params)
        return True

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
        try:
            write_json(metrics_path, metrics)
        except IOError as e:
            raise RuntimeError(f"Failed to write log file at {metrics_path}") from e
        return True

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

    def log_predictions(self, y_true, y_preds, task_type) -> dict:
        if task_type.lower() == "regression":
            return {}
        elif task_type.lower() == "classification":
            return {}
