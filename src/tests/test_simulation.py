from datetime import timedelta, date

import pytest

from hivpy import run_simulation, SimulationConfig

@pytest.fixture
def sample_pop():
    return {}

def test_simulation_no_change(sample_pop):
    """Test that nothing happens when the end date is the same as the start."""
    date_bound = date.today()
    config = SimulationConfig(start=date_bound, end=date_bound)
    last_pop = run_simulation(sample_pop, config)
    assert last_pop == sample_pop


def test_error_end_before_start():
    """Ensure that we throw an error if the end date is before the start."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    wrong_config = SimulationConfig(start=today, end=yesterday)
    with pytest.raises(Exception):
        run_simulation(sample_pop, wrong_config)


def test_error_end_before_first_step():
    """Ensure that we throw an error if the simulation would end before the first step."""
    start = date.today()
    end = start + timedelta(days=30)
    step = timedelta(days=90)
    wrong_config = SimulationConfig(start=start, end=end, step=step)
    with pytest.raises(Exception):
        run_simulation(sample_pop, wrong_config)
