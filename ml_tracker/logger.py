from datetime import datetime
import uuid
from .utils import write_json
import os


class Experiment:
    def __init__(self, name: str, base_dir: str = "experiments"):
        self.name = name
        self.base_dir = base_dir

        self.expir = os.path.join(base_dir, self.name)
        os.makedirs(self.expir, exist_ok=True)

    def log_params(self, params, run_id):
        params_path = os.path.join(self.expir, run_id)   # ✅ correct folder path
        os.makedirs(params_path, exist_ok=True)
        json_path = os.path.join(params_path, "params.json")  # ✅ file path

        write_json(json_path, params)   # ✅ save to file instead of folder
        print("params logged in perfectly")

        return True
