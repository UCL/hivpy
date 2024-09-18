import pytest

from hivpy import SimulationConfig, SimulationException
from hivpy.common import date, timedelta
from hivpy.simulation import SimulationHandler

# from hivpy.population import Population


# def test_simulation_no_change():
#     """Test that nothing happens when the end date is the same as the start."""
#     size = 100
#     date_bound = date.today()
#     initial_pop = Population(size, date.today())
#     config = SimulationConfig(start_date=date_bound, stop_date=date_bound,
#                               time_step=timedelta(days=90), population_size=size)
#     last_pop, _ = run_simulation(config)
#     # TODO The way the code is currentlywritten, this will always pass because
#     # we update the population in-place. For a meaningful test, we should
#     # copy the initial population or change the population evolution code.
#
#     # TODO We should implement a better equality check for populations
#     assert (last_pop.date == initial_pop.date
#             and last_pop.size == initial_pop.size)


def test_error_end_before_start(tmp_path):
    """
    Ensure that we throw an error if the end date is before the start.
    """
    today = date(1989, 1)
    yesterday = today - timedelta(days=30)
    with pytest.raises(SimulationException):
        SimulationConfig(start_date=today, stop_date=yesterday, output_dir=tmp_path,
                         graph_outputs=[], population_size=100)


def test_error_end_before_first_step(tmp_path):
    """
    Ensure that we throw an error if the simulation would end before the first step.
    """
    start = date(1989, 1)
    end = start + timedelta(days=30)
    step = timedelta(days=90)
    with pytest.raises(SimulationException):
        SimulationConfig(start_date=start, stop_date=end, output_dir=tmp_path, graph_outputs=[],
                         time_step=step, population_size=100)


def test_death_occurs(tmp_path):
    """
    Check that the number of people alive always decreases.
    """
    # FIXME This will not necessarily be true once we add in births
    size = 10000
    start = date(1989, 1)
    step = timedelta(days=90)
    end = start + 200 * step
    config = SimulationConfig(size, start, end, tmp_path, [], step)
    simulation_handler = SimulationHandler(config)
    simulation_handler.run()
    pop = simulation_handler.population
    # Check that the number alive never grows... (some steps may have 0 deaths)
    # assert all(results.num_alive.diff()[1:] <= 0)
    num_not_dead = len(pop.data)
    assert (num_not_dead < size)
    num_dead = size - num_not_dead
    assert (num_dead >= 1)
    # assert sum(simulation_handler.output.output_stats["Deaths (tot)"]) == num_dead
    # ...and that there is at least one death overall!
    # FIXME This is not guaranteed at the moment because of the values used
    # assert results.num_alive[-1] < results.num_alive[0]


def test_error_intervention_before_start(tmp_path):
    """
    Ensure that we throw an error if the intervention date is before the start.
    """
    start = date(1989, 1)
    end = date(1995, 1)
    intervention = start - timedelta(days=365)
    with pytest.raises(SimulationException):
        SimulationConfig(start_date=start, stop_date=end, output_dir=tmp_path,
                         graph_outputs=[], intervention_date=intervention, population_size=100)


def test_error_intervention_after_end(tmp_path):
    """
    Ensure that we throw an error if the intervention date is after the end.
    """
    start = date(1989, 1)
    end = date(1995, 1)
    intervention = end + timedelta(days=365)
    with pytest.raises(SimulationException):
        SimulationConfig(start_date=start, stop_date=end, output_dir=tmp_path,
                         graph_outputs=[], intervention_date=intervention, population_size=100)


def test_intervention_option(tmp_path):
    """
    Assert that the option number is implemented
    In this case for the sexual worker program start date
    """
    size = 1000
    start = date(1989, 1)
    step = timedelta(days=90)
    end = date(1995, 1)
    intervention = date(1992, 1)
    option = 1
    config = SimulationConfig(size, start, end, tmp_path, [], step, intervention, option)
    simulation_handler = SimulationHandler(config)

    simulation_handler.run()

    modified_date = simulation_handler.modified_population.sexual_behaviour.sw_program_start_date
    assert modified_date == intervention


def test_recurrent_intervention(tmp_path):
    """
    Assert that the intervention is being updated when required
    And with a different option number
    """
    size = 1000
    start = date(1989, 1)
    step = timedelta(days=90)
    end = date(1995, 1)
    intervention = date(1992, 1)
    option = 1
    repeat_interv = True
    config = SimulationConfig(size, start, end, tmp_path, [], step, intervention, option, repeat_interv)
    simulation_handler = SimulationHandler(config)

    simulation_handler.run()

    assert simulation_handler.modified_population.circumcision.policy_intervention_year == end
