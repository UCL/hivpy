from datetime import timedelta, date

import pytest

from hivpy import run_simulation, SimulationConfig, SimulationException
from hivpy.population import Population

@pytest.fixture
def sample_pop():
    return Population(100, date.today())


def test_simulation_no_change(sample_pop):
    """Test that nothing happens when the end date is the same as the start."""
    date_bound = date.today()
    config = SimulationConfig(start_date=date_bound, stop_date=date_bound,
                              time_step=timedelta(days=0))
    last_pop, _ = run_simulation(sample_pop, config)
    # TODO The way the code is currentlywritten, this will always pass because
    # we update the population in-place. For a meaningful test, we should
    # copy the initial population or change the population evolution code.
    assert last_pop == sample_pop


def test_error_end_before_start():
    """Ensure that we throw an error if the end date is before the start."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    with pytest.raises(SimulationException):
        SimulationConfig(start_date=today, stop_date=yesterday)


def test_error_end_before_first_step():
    """Ensure that we throw an error if the simulation would end before the first step."""
    start = date.today()
    end = start + timedelta(days=30)
    step = timedelta(days=90)
    with pytest.raises(SimulationException):
        SimulationConfig(start_date=start, stop_date=end, time_step=step)


def test_can_track(sample_pop):
    """Check that we can get outputs from tracked attributes."""
    start = sample_pop.date
    step = timedelta(days=30)
    end = start + step
    config = SimulationConfig(start, end, step, ['num_alive'])
    pop, results = run_simulation(sample_pop, config)
    assert 'num_alive' in results


def test_error_tracking_nonexistent(sample_pop):
    """Check that we error if trying to track attributes that don't exist."""
    start = sample_pop.date
    step = timedelta(days=30)
    end = start + step
    config = SimulationConfig(start, end, step, ['NOT_AN_ATTR'])
    with pytest.raises(SimulationException,
                       match="Unrecognised tracked attribute"):
        run_simulation(sample_pop, config)
