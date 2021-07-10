from dataclasses import dataclass
from datetime import date, timedelta

from .common import SimulationException


@dataclass
class SimulationConfig:
    """A class holding the parameters required for running a simulation."""
    start_date: date
    stop_date: date
    time_step: timedelta = timedelta(days=90)

    def _validate(self):
        """Make sure the values passed in make sense."""
        try:
            assert self.stop_date >= self.start_date + self.time_step
        except AssertionError:
            raise SimulationException("Invalid simulation configuration.")

    def __post_init__(self):
        """This is called automatically during construction."""
        self._validate()


def run_simulation(population, config):
    """Run a single simulation for the given population and time bounds."""
    date = config.start_date
    time_step = config.time_step
    while date < config.stop_date:
        population = population.evolve(time_step)
        date = date + time_step
    return population

