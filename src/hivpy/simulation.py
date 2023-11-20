from __future__ import annotations

import logging
import operator
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from titlecase import titlecase

import hivpy.column_names as col

from .common import SexType
from .config import SimulationConfig
from .population import Population


class SimulationOutput:

    def __init__(self, simulation_config: SimulationConfig):
        # current step
        self.step = 0
        # age boundaries
        self.age_min = 15
        self.age_max = 100
        self.age_step = 10
        # determine output columns
        output_columns = ["Date", "HIV prevalence (tot)", "HIV prevalence (male)",
                          "HIV prevalence (female)", "HIV infections (tot)",
                          "Population (over 15)", "Births (tot)", "Deaths (tot)"]
        for age_bound in range(self.age_min, self.age_max, self.age_step):
            key = f"Population ({age_bound}-{age_bound+(self.age_step-1)})"
            output_columns.insert(4+int(age_bound/10)*2, key)
            key = f"HIV prevalence ({age_bound}-{age_bound+(self.age_step-1)})"
            output_columns.insert(3+int(age_bound/10), key)
        # determine number of output rows
        self.num_steps = int((simulation_config.stop_date -
                             simulation_config.start_date) / simulation_config.time_step) + 1
        # store output information as a dataframe
        self.output_stats = pd.DataFrame(index=range(self.num_steps), columns=output_columns)

    def _update_date(self, date):
        self.latest_date = date
        self.output_stats.loc[self.step, "Date"] = self.latest_date

    def _ratio(self, subpop, pop):
        if len(pop) != 0:
            return len(subpop)/len(pop)
        else:
            return 0

    def _update_HIV_prevalence(self, pop: Population):
        # Update total HIV cases and population
        over_15_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15)])
        HIV_pos_idx = pop.get_sub_pop([(col.HIV_STATUS, operator.eq, True)])
        self.output_stats.loc[self.step, "HIV prevalence (tot)"] = self._ratio(HIV_pos_idx, over_15_idx)
        self.output_stats.loc[self.step, "HIV infections (tot)"] = len(HIV_pos_idx)
        self.output_stats.loc[self.step, "Population (over 15)"] = len(over_15_idx)

        # Update HIV prevalence by sex
        men_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
        self.output_stats.loc[self.step, "HIV prevalence (male)"] = (
            self._ratio(pop.get_sub_pop_intersection(men_idx, HIV_pos_idx),
                        pop.get_sub_pop_intersection(men_idx, over_15_idx)))
        women_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])
        self.output_stats.loc[self.step, "HIV prevalence (female)"] = (
            self._ratio(pop.get_sub_pop_intersection(women_idx, HIV_pos_idx),
                        pop.get_sub_pop_intersection(women_idx, over_15_idx)))

        # Update HIV prevalence and population by age
        for age_bound in range(self.age_min, self.age_max, self.age_step):
            age_idx = pop.get_sub_pop([(col.AGE, operator.ge, age_bound),
                                       (col.AGE, operator.lt, age_bound+self.age_step)])
            key = f"Population ({age_bound}-{age_bound+(self.age_step-1)})"
            self.output_stats.loc[self.step, key] = len(age_idx)
            key = f"HIV prevalence ({age_bound}-{age_bound+(self.age_step-1)})"
            self.output_stats.loc[self.step, key] = self._ratio(pop.get_sub_pop_intersection(HIV_pos_idx,
                                                                                             age_idx), age_idx)

    def _update_births(self, pop: Population, time_step):
        born_this_step = pop.get_sub_pop([(col.AGE, operator.ge, 0.25),
                                          (col.AGE, operator.lt, 0.25 + time_step.days / 365)])
        self.output_stats.loc[self.step, "Births (tot)"] = len(born_this_step)

    def _update_deaths(self, pop: Population):
        died_this_step = pop.get_sub_pop([(col.DATE_OF_DEATH, operator.eq, self.latest_date)])
        self.output_stats.loc[self.step, "Deaths (tot)"] = len(died_this_step)

    def update_summary_stats(self, date, pop: Population, time_step):
        self._update_date(date)
        self._update_HIV_prevalence(pop)
        self._update_births(pop, time_step)
        self._update_deaths(pop)
        self.step += 1

    def write_output(self, output_path):
        self.output_stats.to_csv(output_path, index_label="Time Step", mode='w')

    def graph_output(self, output_dir, output_stats, graph_outputs):

        for out in graph_outputs:
            if out in output_stats.columns:

                plt.subplots()
                plt.plot(output_stats["Date"], output_stats[out])
                title_out = titlecase(out)

                plt.xlabel("Date")
                plt.ylabel(title_out)
                plt.title("{0} Over Time".format(title_out))
                plt.savefig(os.path.join(output_dir, "{0} Over Time".format(title_out)))


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
        self.output_dir = simulation_config.output_dir / (
            "simulation_output_" + str(datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.output_path = self.output_dir / (
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
            self.output.update_summary_stats(date, self.population, time_step)
            date = date + time_step
        logging.info("finished")
        # Store results
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output.write_output(self.output_path)
        self.output.graph_output(self.output_dir, self.output.output_stats, self.simulation_config.graph_outputs)
