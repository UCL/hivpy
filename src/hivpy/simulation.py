from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

import pandas as pd

from .common import SimulationException
from .population import Population


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


class SimulationHandler:
    """A class for handling executing a simulation and accessing results."""
    config: SimulationConfig
    population: Population
    results: pd.DataFrame
    
    def __init__(self, config):
        self.config = config
        self.population = None
        self.results = None

    def set_population(self, population):
        """Specify what population the simulation should use."""
        self.population = population
        # Reset the results if another simulation had been run
        self.results = None

    def _validate_tracked(self, population, tracked):
        for attribute in tracked:
            if not population.has_attribute(attribute):
                raise SimulationException(
                    f"Unrecognised tracked attribute: {attribute}")

    def run(self, tracked):
        self._validate_tracked(self.population, tracked)
        # Store the tracking results in a dataframe, with one row per date
        results = pd.DataFrame(columns=tracked)
        # Start the simulation
        date = self.config.start_date
        assert date == self.population.date
        time_step = self.config.time_step
        while date < self.config.stop_date:
            # Advance the population
            self.population = self.population.evolve(time_step)
            date = date + time_step
            # Record the values of the tracked attributes
            if tracked:  # we need this because we can't set an empty row
                results.loc[date] = {
                    attr: self.population.get(attr) for attr in tracked
                }


def run_simulation(population, config, tracked_attrs=[]):
    """Run a single simulation for the given population and time bounds.
    
    This is a convenience method to avoid using SimulationHandler directly.
    """
    handler = SimulationHandler(config)
    handler.set_population(population)
    handler.run(tracked_attrs)
    return (handler.population, handler.results)