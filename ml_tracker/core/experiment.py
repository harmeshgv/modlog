from datetime import datetime
from pathlib import Path
from ml_tracker.utils import write_json
from ml_tracker.core.run import RUN
import os


class EXPERIMENT:
    def __init__(self, name: str, base_dir: str | Path = "experiments"):
        self.name = name
        self.base_dir = Path(base_dir)
        self.expir = self.base_dir / self.name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self._runs = {}

    def start_run(self, run_id: str | None = None) -> RUN:
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_id in self._runs:
            raise ValueError(f"run {run_id} aready exists for this experiment")

        run_path = self.exp_dir / run_id
        run = RUN(run_id, run_path)
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> RUN:
        return self._runs[run_id]

    def list_run(self):
        return list(self._runs.keys())
