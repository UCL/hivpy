from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List
import logging
import pandas as pd

from .exceptions import SimulationException
from .population import Population


@dataclass
class SimulationConfig:
    """A class holding the parameters required for running a simulation."""
    population_size: int
    start_date: date
    stop_date: date
    time_step: timedelta = timedelta(days=90)
    tracked: List[str] = field(default_factory=list)

    def _validate(self):
        """Make sure the values passed in make sense."""
        try:
            assert self.stop_date >= self.start_date + self.time_step
        except AssertionError:
            raise SimulationException("Invalid simulation configuration.")

    def __post_init__(self):
        """This is called automatically during construction."""
        self._validate()

    def track(self, attribute_name):
        """Track an additional attribute during simulation."""
        # TODO Check if already tracked (or conver tracked to a set?)
        self.tracked.append(attribute_name)


class SimulationHandler:
    """A class for handling executing a simulation and accessing results."""
    simulation_config: SimulationConfig
    population: Population
    results: pd.DataFrame
    
    def __init__(self, simulation_config):
        self.simulation_config = simulation_config
        self.results = None
        self._initialise_population()

    def _initialise_population(self):
        self.population = Population(self.simulation_config.population_size, self.simulation_config.start_date)


    def _validate_tracked(self, population):
        for attribute in self.simulation_config.tracked:
            if not population.has_attribute(attribute):
                raise SimulationException(
                    f"Unrecognised tracked attribute: {attribute}")

    def run(self):
        self._validate_tracked(self.population)
        # Store the tracking results in a dataframe, with one row per date
        tracked_attrs = self.simulation_config.tracked
        results = pd.DataFrame(columns=tracked_attrs)
        # Start the simulation
        date = self.simulation_config.start_date
        assert date == self.population.date
        time_step = self.simulation_config.time_step
        while date < self.simulation_config.stop_date:
            logging.info("Timestep %s\n",date)
            # Advance the population
            self.population = self.population.evolve(time_step)
            date = date + time_step
            # Record the values of the tracked attributes
            if tracked_attrs:  # we need this because we can't set an empty row
                results.loc[date] = {
                    attr: self.population.get(attr) for attr in tracked_attrs
                }
        logging.info("finished")


def run_simulation(simulation_config):
    """Run a single simulation for the given population and time bounds.
    
    This is a convenience method to avoid using SimulationHandler directly.
    """
    handler = SimulationHandler(simulation_config)
    handler.run()
    return (handler.population, handler.results)
