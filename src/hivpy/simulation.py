from __future__ import annotations

import logging
import operator
import os
from datetime import datetime, timedelta

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
                          "HIV prevalence (female)", "HIV prevalence (sex worker)",
                          "HIV prevalence (15-49)", "Circumcision (15-49)", "HIV infections (tot)",
                          "CD4 count (under 200)", "CD4 count (200-500)", "CD4 count (over 500)",
                          "Population (over 15)", "Long term partner (15-64)",
                          "Short term partners (15-64)", "Over 5 short term partners (15-64)",
                          "Sex worker (ratio)", "Giving birth (ratio)", "Infected newborns (ratio)",
                          "Births (tot)", "Deaths (tot)", "Deaths (over 15, male)",
                          "Deaths (over 15, female)", "Deaths (20-59, male)", "Deaths (20-59, female)"]
        for age_bound in range(self.age_min, self.age_max, self.age_step):
            # inserted after 'Population (over 15)' column
            key = f"Population ({age_bound}-{age_bound+(self.age_step-1)})"
            output_columns.insert(10+int(age_bound/10)*2, key)
            # inserted after 'HIV prevalence (15-49)' column
            key = f"HIV prevalence ({age_bound}-{age_bound+(self.age_step-1)})"
            output_columns.insert(5+int(age_bound/10), key)
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

        # Update HIV prevalence in female sex workers
        sex_workers_idx = pop.get_sub_pop([(col.SEX_WORKER, operator.eq, True)])
        self.output_stats.loc[self.step, "Sex worker (ratio)"] = self._ratio(sex_workers_idx, women_idx)
        self.output_stats.loc[self.step, "HIV prevalence (sex worker)"] = (
            self._ratio(pop.get_sub_pop_intersection(sex_workers_idx, HIV_pos_idx), sex_workers_idx))

        # Update HIV prevalence and population by age
        age_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15),
                                   (col.AGE, operator.lt, 50)])
        self.output_stats.loc[self.step, "HIV prevalence (15-49)"] = (
            self._ratio(pop.get_sub_pop_intersection(age_idx, HIV_pos_idx), age_idx))

        for age_bound in range(self.age_min, self.age_max, self.age_step):
            age_idx = pop.get_sub_pop([(col.AGE, operator.ge, age_bound),
                                       (col.AGE, operator.lt, age_bound+self.age_step)])
            key = f"Population ({age_bound}-{age_bound+(self.age_step-1)})"
            self.output_stats.loc[self.step, key] = len(age_idx)
            key = f"HIV prevalence ({age_bound}-{age_bound+(self.age_step-1)})"
            self.output_stats.loc[self.step, key] = (
                self._ratio(pop.get_sub_pop_intersection(HIV_pos_idx, age_idx), age_idx))

    def _update_CD4_count(self, pop: Population):
        # Update number of people with given CD4 counts
        cd4_under_200_idx = pop.get_sub_pop([(col.CD4, operator.lt, 200)])
        cd4_200_to_500_idx = pop.get_sub_pop([(col.CD4, operator.ge, 200),
                                              (col.CD4, operator.le, 500)])
        cd4_over_500_idx = pop.get_sub_pop([(col.CD4, operator.gt, 500)])
        self.output_stats.loc[self.step, "CD4 count (under 200)"] = len(cd4_under_200_idx)
        self.output_stats.loc[self.step, "CD4 count (200-500)"] = len(cd4_200_to_500_idx)
        self.output_stats.loc[self.step, "CD4 count (over 500)"] = len(cd4_over_500_idx)

    def _update_circumcision(self, pop: Population):
        # Update proportion of circumcised men
        men_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
        circumcised_idx = pop.get_sub_pop([(col.CIRCUMCISED, operator.eq, True),
                                           (col.AGE, operator.ge, 15),
                                           (col.AGE, operator.lt, 50)])
        self.output_stats.loc[self.step, "Circumcision (15-49)"] = self._ratio(circumcised_idx, men_idx)

    def _update_partners(self, pop: Population):
        # Update proportion of people with long term partners
        age_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15),
                                   (col.AGE, operator.lt, 65)])
        ltp_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15),
                                   (col.AGE, operator.lt, 65),
                                   (col.LONG_TERM_PARTNER, operator.eq, True)])
        self.output_stats.loc[self.step, "Long term partner (15-64)"] = self._ratio(ltp_idx, age_idx)
        # Update proportion of people with short term partners
        stp_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15),
                                   (col.AGE, operator.lt, 65),
                                   (col.NUM_PARTNERS, operator.ge, 1)])
        self.output_stats.loc[self.step, "Short term partners (15-64)"] = self._ratio(stp_idx, age_idx)
        # Update proportion of people with 5+ short term partners
        stp_over_5_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15),
                                          (col.AGE, operator.lt, 65),
                                          (col.NUM_PARTNERS, operator.ge, 5)])
        self.output_stats.loc[self.step, "Over 5 short term partners (15-64)"] = self._ratio(stp_over_5_idx, age_idx)

    def _update_births(self, pop: Population, time_step):
        # Update total births
        born_this_step = pop.get_sub_pop([(col.AGE, operator.ge, 0.25),
                                          (col.AGE, operator.lt, 0.25 + time_step.days / 365)])
        self.output_stats.loc[self.step, "Births (tot)"] = len(born_this_step)

        # Update proportion of women giving birth and infected children
        women_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])
        giving_birth_this_step = pop.get_sub_pop([(col.PREGNANT, operator.eq, True),
                                                  (col.LAST_PREGNANCY_DATE, operator.le,
                                                   pop.date - timedelta(days=270))])
        self.output_stats.loc[self.step, "Giving birth (ratio)"] = self._ratio(giving_birth_this_step,
                                                                               women_idx)
        infected_newborns = pop.get_sub_pop([(col.INFECTED_BIRTH, operator.eq, True)])
        self.output_stats.loc[self.step, "Infected newborns (ratio)"] = self._ratio(infected_newborns,
                                                                                    born_this_step)

    def _update_deaths(self, pop: Population):
        # Update total deaths
        died_this_step = pop.get_sub_pop([(col.DATE_OF_DEATH, operator.eq, self.latest_date)])
        self.output_stats.loc[self.step, "Deaths (tot)"] = len(died_this_step)

        # Update deaths by sex and age
        over_15_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15)])
        aged_20_to_59_idx = pop.get_sub_pop([(col.AGE, operator.ge, 20),
                                             (col.AGE, operator.lt, 60)])
        men_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
        women_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])

        self.output_stats.loc[self.step, "Deaths (over 15, male)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(men_idx, over_15_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "Deaths (over 15, female)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(women_idx, over_15_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "Deaths (20-59, male)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(men_idx, aged_20_to_59_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "Deaths (20-59, female)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(women_idx, aged_20_to_59_idx),
                                             died_this_step)))

    def update_summary_stats(self, date, pop: Population, time_step):
        self._update_date(date)
        self._update_HIV_prevalence(pop)
        self._update_CD4_count(pop)
        self._update_circumcision(pop)
        self._update_partners(pop)
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
