from datetime import datetime
from .utils import write_json
from .run import RUN
import os

class EXPERIMENT:
    def __init__(self, name: str, base_dir: str = "experiments"):
        self.name = name
        self.base_dir = base_dir

        self.expir = os.path.join(base_dir, self.name)
        os.makedirs(self.expir, exist_ok=True)

    def start_run(self, run_id: str):
        run_path = os.path.join(self.expir , run_id)
        os.makedirs(run_path, exist_ok=True)
        return RUN(run_id, run_path)
    
        