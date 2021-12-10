import os.path
from configparser import ConfigParser

import pytest

from hivpy.experiment import create_experiment, run_experiment


@pytest.fixture
def sample_experiment_params():
    """Sample experiment parameters for testing."""
    parser = ConfigParser()
    parser.read(
        os.path.join(os.path.dirname(__file__), 'fixtures', 'sample.conf'))
    print(list(parser.items()))
    return parser


def test_dummy_workflow(tmp_path, sample_experiment_params):
    """Check that we can run a sample experiment from start to end.

    This is only a placeholder test and should be replaced when we have
    implemented actual functionality.
    """
    d = tmp_path / "log"
    d.mkdir()
    # redirect file logging to tmp file (so it is deleted)
    sample_experiment_params["LOGGING"]["LOG_DIRECTORY"] = str(d)
    dummy_config = create_experiment(sample_experiment_params)
    run_experiment(dummy_config)
