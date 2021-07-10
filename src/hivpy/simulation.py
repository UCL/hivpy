from dataclasses import dataclass
from datetime import date, timedelta



@dataclass
class SimulationConfig:
    """A class holding the parameters required for running a simulation."""
    start_date: date
    stop_date: date
    time_step: timedelta = timedelta(days=90)


def run_simulation(population, config):
    """Run a single simulation for the given population and time bounds."""
    date = config.start_date
    time_step = config.time_step
    while date < config.stop_date:
        population = population.evolve(time_step)
        date = date + time_step
    return population

