from datetime import timedelta, date

import pytest

from hivpy import run_simulation, SimulationConfig, SimulationException
from hivpy import population
from hivpy.population import Population


def test_simulation_no_change():
    """Test that nothing happens when the end date is the same as the start."""
    size = 100
    date_bound = date.today()
    initial_pop = Population(size, date.today())
    config = SimulationConfig(start_date=date_bound, stop_date=date_bound,
                              time_step=timedelta(days=0), population_size=size)
    last_pop, _ = run_simulation(config)
    # TODO The way the code is currentlywritten, this will always pass because
    # we update the population in-place. For a meaningful test, we should
    # copy the initial population or change the population evolution code.

    # TODO We should implement a better equality check for populations
    assert (last_pop.date == initial_pop.date
            and last_pop.size == initial_pop.size)


def test_error_end_before_start():
    """Ensure that we throw an error if the end date is before the start."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    with pytest.raises(SimulationException):
        SimulationConfig(start_date=today, stop_date=yesterday,
                         population_size=100)


def test_error_end_before_first_step():
    """Ensure that we throw an error if the simulation would end before the first step."""
    start = date.today()
    end = start + timedelta(days=30)
    step = timedelta(days=90)
    with pytest.raises(SimulationException):
        SimulationConfig(start_date=start, stop_date=end, time_step=step,
                         population_size=100)


def test_can_track():
    """Check that we can get outputs from tracked attributes."""
    size = 100
    start = date.today()
    step = timedelta(days=30)
    end = start + step
    config = SimulationConfig(size, start, end, step, ['num_alive'])
    pop, results = run_simulation(config)
    assert 'num_alive' in results


def test_error_tracking_nonexistent():
    """Check that we error if trying to track attributes that don't exist."""
    size = 100
    start = date.today()
    step = timedelta(days=30)
    end = start + step
    config = SimulationConfig(size, start, end, step, ['NOT_AN_ATTR'])
    with pytest.raises(SimulationException,
                       match="Unrecognised tracked attribute"):
        run_simulation(config)


def test_death_occurs():
    """Check that the number of people alive always decreases."""
    # FIXME This will not necessarily be true once we add in births
    size = 10000
    start = date.today()
    step = timedelta(days=30)
    end = start + 100 * step
    config = SimulationConfig(size, start, end, step, ['num_alive'])
    pop, results = run_simulation(config)
    # Check that the number alive never grows... (some steps may have 0 deaths)
    assert all(results.num_alive.diff()[1:] <= 0)
    # ...and that there is at least one death overall!
    assert results.num_alive[-1] < results.num_alive[0]
