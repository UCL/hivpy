import os.path

import pytest

from hivpy.config import ExperimentConfig
from hivpy.experiment import run_experiment


@pytest.fixture
def sample_config_file():
    """Sample experiment parameters for testing."""
    config_file = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample.conf')
    return config_file


def test_dummy_workflow(sample_config_file):
    """Check that we can run a sample experiment from start to end.

    This is only a placeholder test and should be replaced when we have
    implemented actual functionality.
    """
    dummy_config = ExperimentConfig.from_file(sample_config_file)
    run_experiment(dummy_config)
