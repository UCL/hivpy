import os.path

import pytest
import yaml

from hivpy.experiment import create_experiment, run_experiment


@pytest.fixture
def sample_experiment_params():
    """
    Sample experiment parameters for testing.
    """
    filepath = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample.yaml')
    with open(filepath, 'r') as sample_file:
        sample_config = yaml.safe_load(sample_file)
    return sample_config


def test_dummy_workflow(tmp_path, sample_experiment_params):
    """
    Check that we can run a sample experiment from start to end.

    This is only a placeholder test and should be replaced when we have
    implemented actual functionality.
    """
    d = tmp_path / "log"
    d.mkdir()
    # redirect file logging to tmp file (so it is deleted)
    sample_experiment_params["LOGGING"]["log_directory"] = str(d)
    dummy_config = create_experiment(sample_experiment_params)
    run_experiment(dummy_config)
