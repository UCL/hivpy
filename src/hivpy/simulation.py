import logging
import operator
import string
from datetime import datetime

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .demographics import SexType
from .exceptions import SimulationException
from .population import Population
from .sexual_behaviour import selector


class SimulationOutput:
    file_path: string
    output_stats: dict

    def _init_output_field(self, key, default):
        self.output_stats[key] = np.array([default]*self.num_steps)

    def _init_output_fields(self, key_default_pairs):
        for (k, d) in key_default_pairs:
            self._init_output_field(k, d)

    def __init__(self, simulation_config: SimulationConfig):
        self.num_steps = int((simulation_config.stop_date -
                             simulation_config.start_date) / simulation_config.time_step)
        self.output_stats = {"Date": np.array([simulation_config.start_date]*self.num_steps)}
        self._init_output_fields([("HIV prevalence (tot)", 0.0),
                                  ("HIV prevalence (male)", 0.0),
                                  ("HIV prevalence (female)", 0.0),
                                  ("HIV infections (tot)", 0),
                                  ("Population (over 15)", 0)])
        self.file_path = "output/simulation_output." + \
            str(datetime.now().strftime("%Y%m%d-%H%M%S")) + ".csv"
        self.age_min = 15
        self.age_max = 100
        self.age_step = 10
        self.step = 0

    def _update_date(self, date):
        self.output_stats["Date"][self.step] = date

    def _ratio(self, subpop, pop):
        if sum(pop) != 0:
            return sum(subpop)/sum(pop)
        else:
            return 0

    def _update_HIV_prevalence(self, pop_data):
        # Update total HIV cases and population
        over_15_idx = selector(pop_data, age=(operator.gt, 15))
        HIV_pos_idx = selector(pop_data, HIV_status=(operator.eq, True))
        self.output_stats["HIV prevalence (tot)"][self.step] = self._ratio(HIV_pos_idx, over_15_idx)
        self.output_stats["HIV infections (tot)"][self.step] = sum(HIV_pos_idx)
        self.output_stats["Population (over 15)"][self.step] = sum(over_15_idx)

        # Update HIV prevalence by sex
        men_idx = selector(pop_data, sex=(operator.eq, SexType.Male))
        self.output_stats["HIV prevalence (male)"][self.step] = (
            self._ratio(HIV_pos_idx & men_idx, men_idx & over_15_idx))
        women_idx = selector(pop_data, sex=(operator.eq, SexType.Female))
        self.output_stats["HIV prevalence (female)"][self.step] = self._ratio(
            HIV_pos_idx & women_idx, women_idx & over_15_idx)

        # Update HIV prevalence by age
        for age_bound in range(self.age_min, self.age_max, self.age_step):
            key = "HIV prevalence (" + str(age_bound) + "-" + str(age_bound+(self.age_step-1)) + ")"
            age_idx = (pop_data["age"] >= age_bound) & (
                pop_data["age"] < (age_bound + self.age_step))
            if(key not in self.output_stats.keys()):
                self._init_output_field(key, 0.0)
            self.output_stats[key][self.step] = self._ratio(HIV_pos_idx & age_idx, age_idx)

    def update_summary_stats(self, date, pop_data):
        self._update_date(date)
        self._update_HIV_prevalence(pop_data)
        self.step += 1

    def write_output(self):
        df = pd.DataFrame(self.output_stats)
        df.to_csv(self.file_path, mode='w')


class SimulationHandler:
    """A class for handling executing a simulation and accessing results."""
    simulation_config: SimulationConfig
    population: Population
    results: pd.DataFrame

    def __init__(self, simulation_config):
        self.simulation_config = simulation_config
        self.results = None
        self._initialise_population()
        self.output = SimulationOutput(self.simulation_config)

    def _initialise_population(self):
        self.population = Population(self.simulation_config.population_size,
                                     self.simulation_config.start_date)

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
            logging.info("Timestep %s\n", date)
            # Advance the population
            self.population = self.population.evolve(time_step)
            date = date + time_step
            self.output.update_summary_stats(date, self.population.data)
            # Record the values of the tracked attributes
            if tracked_attrs:  # we need this because we can't set an empty row
                results.loc[date] = {
                    attr: self.population.get(attr) for attr in tracked_attrs
                }
        logging.info("finished")
        self.output.write_output()
        self.results = results


def run_simulation(simulation_config):
    """Run a single simulation for the given population and time bounds.

    This is a convenience method to avoid using SimulationHandler directly.
    """
    handler = SimulationHandler(simulation_config)
    handler.run()
    return (handler.population, handler.results)
