import pytest
from ml_tracker.experiment import EXPERIMENT

def test_create_experiment(tmp_path):
    """Test that an Experiment creates its directory correctly."""
    exp = EXPERIMENT(name="test_exp", base_dir=str(tmp_path))
    exp_path = tmp_path / "test_exp"  # Using Path object
    assert exp_path.exists()

def test_create_runs(tmp_path):
    exp = EXPERIMENT(name="test_exp", base_dir=str(tmp_path))
    run_1 = "1"

    exp.start_run(run_1)
    run_1_path = tmp_path / "test_exp" / run_1
    assert run_1_path.exists()

