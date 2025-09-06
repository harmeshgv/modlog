import os
from .utils import write_json

class RUN:
    def __init__(self, run_id, run_path):
        self.run_path = run_path
        self.run_id = run_id

    def log_params(self, params):
        if not isinstance(params, dict):
            if hasattr(params, "get_params"):
                params = params.get_params()
            else:
                raise ValueError("params must be a dict or have a get_params() method")
            
        os.makedirs(self.run_path, exist_ok=True)  
        log_path = os.path.join(self.run_path, "log_params.json")
        write_json(log_path, params)
        return True
