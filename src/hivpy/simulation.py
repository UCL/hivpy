from __future__ import annotations

import logging
import operator
from datetime import datetime

import numpy as np
import pandas as pd

import hivpy.column_names as col

from .common import SexType
from .config import SimulationConfig
from .population import Population


class SimulationOutput:

    def _init_output_field(self, key, default):
        self.output_stats[key] = np.array([default]*self.num_steps)

    def _init_output_fields(self, key_default_pairs):
        for (k, d) in key_default_pairs:
            self._init_output_field(k, d)

    def __init__(self, simulation_config: SimulationConfig):
        self.num_steps = int((simulation_config.stop_date -
                             simulation_config.start_date) / simulation_config.time_step) + 1
        self.output_stats = {"Date": np.array([simulation_config.start_date]*self.num_steps)}
        self._init_output_fields([("HIV prevalence (tot)", 0.0),
                                  ("HIV prevalence (male)", 0.0),
                                  ("HIV prevalence (female)", 0.0),
                                  ("HIV infections (tot)", 0),
                                  ("Population (over 15)", 0),
                                  ("Deaths (tot)", 0.0),
                                  ("PrEP usage overall", 0.0),
                                  ("PrEP usage (men)", 0.0),
                                  ("PrEP usage (women)", 0.0)])
        self.age_min = 15
        self.age_max = 100
        self.age_step = 10
        self.step = 0

    def _update_date(self, date):
        self.latest_date = date
        self.output_stats["Date"][self.step] = self.latest_date

    def _ratio(self, subpop, pop):
        if len(pop) != 0:
            return len(subpop)/len(pop)
        else:
            return 0

    def _update_HIV_prevalence(self, pop: Population):
        # Update total HIV cases and population
        over_15_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15)])
        HIV_pos_idx = pop.get_sub_pop([(col.HIV_STATUS, operator.eq, True)])
        self.output_stats["HIV prevalence (tot)"][self.step] = self._ratio(HIV_pos_idx, over_15_idx)
        self.output_stats["HIV infections (tot)"][self.step] = len(HIV_pos_idx)
        self.output_stats["Population (over 15)"][self.step] = len(over_15_idx)

        # Update HIV prevalence by sex
        men_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
        self.output_stats["HIV prevalence (male)"][self.step] = (
            self._ratio(pop.get_sub_pop_intersection(men_idx, HIV_pos_idx),
                        pop.get_sub_pop_intersection(men_idx, over_15_idx)))
        women_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])
        self.output_stats["HIV prevalence (female)"][self.step] = (
            self._ratio(pop.get_sub_pop_intersection(women_idx, HIV_pos_idx),
                        pop.get_sub_pop_intersection(women_idx, over_15_idx)))

        # Update HIV prevalence by age
        for age_bound in range(self.age_min, self.age_max, self.age_step):
            key = f"HIV prevalence ({age_bound}-{age_bound+(self.age_step-1)})"
            age_idx = pop.get_sub_pop([(col.AGE, operator.ge, age_bound),
                                       (col.AGE, operator.lt, age_bound+self.age_step)])
            if (key not in self.output_stats.keys()):
                self._init_output_field(key, 0.0)
            self.output_stats[key][self.step] = self._ratio(pop.get_sub_pop_intersection(HIV_pos_idx, age_idx),
                                                            age_idx)

    def _update_deaths(self, pop: Population):
        died_this_step = pop.get_sub_pop([(col.DATE_OF_DEATH, operator.eq, self.latest_date)])
        self.output_stats["Deaths (tot)"][self.step] = len(died_this_step)

    def _update_prep_stats(self, pop_data):
        sexually_active = (pop_data["age"] >= 15) & (pop_data["age"] < 65)
        all_on_prep = selector(pop_data, on_PrEP=(operator.eq, True)) & sexually_active
        men_idx = selector(pop_data, sex=(operator.eq, SexType.Male)) & sexually_active
        women_idx = selector(pop_data, sex=(operator.eq, SexType.Female)) & sexually_active
        self.output_stats["PrEP usage overall"][self.step] = self._ratio(all_on_prep, sexually_active)
        self.output_stats["PrEP usage (men)"][self.step] = self._ratio(all_on_prep & men_idx, men_idx)
        self.output_stats["PrEP usage (women)"][self.step] = self._ratio(all_on_prep & women_idx, women_idx)

    def update_summary_stats(self, date, pop_data):
        self._update_date(date)
        self._update_HIV_prevalence(pop_data)
        self._update_deaths(pop_data)
        self._update_prep_stats(pop_data.data)
        self.step += 1

    def write_output(self, output_path):
        df = pd.DataFrame(self.output_stats)
        df.to_csv(output_path, index_label="Time Step", mode='w')


class SimulationHandler:
    """
    A class for handling executing a simulation and outputting results.
    """
    simulation_config: SimulationConfig
    population: Population

    def __init__(self, simulation_config):
        self.simulation_config = simulation_config
        self._initialise_population()
        self.output = SimulationOutput(self.simulation_config)
        self.output_path = simulation_config.output_dir / (
            "simulation_output_" + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + ".csv")

    def _initialise_population(self):
        self.population = Population(self.simulation_config.population_size,
                                     self.simulation_config.start_date)

    def run(self):
        # Start the simulation
        date = self.simulation_config.start_date
        assert date == self.population.date
        time_step = self.simulation_config.time_step
        while date <= self.simulation_config.stop_date:
            logging.info("Timestep %s\n", date)
            # Advance the population
            self.population = self.population.evolve(time_step)
            self.output.update_summary_stats(date, self.population)
            date = date + time_step
        logging.info("finished")
        self.output.write_output(self.output_path)
