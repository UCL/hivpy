from __future__ import annotations

import math
import operator
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from titlecase import titlecase

import hivpy.column_names as col

from .common import AND, COND, OR, SexType, date, timedelta
# from .config import SimulationConfig
from .population import Population


class SimulationOutput:

    def __init__(self, start_date=date(1989, 1, 1), stop_date=date(2025, 1, 1), time_step=timedelta(months=3)):
        # current step
        self.step = 0
        # age boundaries
        self.age_min = 15
        self.age_max_active = 65
        self.age_max = 100
        self.age_step = 10

        # for HIV status outputs
        self.infected_ltp = 0
        self.infected_stp = 0
        self.infected_primary_infection = 0

        # for pregnancy outputs
        self.infected_newborns = 0
        # TODO
        self.anc_tests = 0
        self.labdel_tests = 0
        self.postdel_tests = 0

        self._init_df(start_date, stop_date, time_step)

    def _init_df(self, start_date, stop_date, time_step):
        # determine output columns
        output_columns = ["Date", "HIV prevalence (tot)", "HIV prevalence (male)",
                          "HIV prevalence (female)", "HIV prevalence (sex worker)",
                          "HIV prevalence (15-49)", "Circumcision (15-49)", "HIV infections (tot)",
                          "Infected by long term partner", "Infected by short term partner",
                          "Infected by primary infection", "CD4 count (under 200)",
                          "CD4 count (200-500)", "CD4 count (over 500)",
                          "Population (over 15)", "Long term partner (15-64)",
                          "Short term partners (15-64)", "Over 5 short term partners (15-64)",
                          "Partner sex balance (male)", "Partner sex balance (female)",
                          "Sex worker (ratio)", "Births (ratio)", "Infected newborns (ratio)",
                          "ANC tests (tot)", "Labour and delivery tests (tot)",
                          "Post-delivery tests (tot)", "Births to infected women (tot)",
                          "Births (tot)", "Deaths (tot)", "Deaths (over 15, male)",
                          "Deaths (over 15, female)", "Deaths (20-59, male)", "Deaths (20-59, female)",
                          "HIV deaths (tot)", "HIV deaths (over 15, male)",
                          "HIV deaths (over 15, female)", "HIV deaths (20-59, male)", "HIV deaths (20-59, female)",
                          "Non-HIV deaths (tot)", "Non-HIV deaths (over 15, male)",
                          "Non-HIV deaths (over 15, female)", "Non-HIV deaths (20-59, male)",
                          "Non-HIV deaths (20-59, female)"]

        for age_bound in range(self.age_min, self.age_max, self.age_step):
            if age_bound < self.age_max_active:
                # inserted after 'Partner sex balance (female)'
                key = f"Partner sex balance ({age_bound}-{age_bound+(self.age_step-1)}, female)"
                output_columns.insert(14+int(age_bound/10)*6, key)
                # inserted after 'Partner sex balance (male)'
                key = f"Partner sex balance ({age_bound}-{age_bound+(self.age_step-1)}, male)"
                output_columns.insert(14+int(age_bound/10)*5, key)
            # inserted after 'Population (over 15)' column
            key = f"Population ({age_bound}-{age_bound+(self.age_step-1)})"
            output_columns.insert(11+int(age_bound/10)*4, key)
            # inserted after 'HIV prevalence (15-49)' column
            key = f"HIV incidence ({age_bound}-{age_bound+(self.age_step-1)}, female)"
            output_columns.insert(3+int(age_bound/10)*3, key)
            key = f"HIV incidence ({age_bound}-{age_bound+(self.age_step-1)}, male)"
            output_columns.insert(4+int(age_bound/10)*2, key)
            key = f"HIV prevalence ({age_bound}-{age_bound+(self.age_step-1)})"
            output_columns.insert(5+int(age_bound/10), key)

        # determine number of output rows
        self.num_steps = int((stop_date - start_date) / time_step) + 1
        # store output information as a dataframe
        self.output_stats = pd.DataFrame(index=range(self.num_steps), columns=output_columns)

    def _update_date(self, date):
        self.latest_date = date
        self.output_stats.loc[self.step, "Date"] = self.latest_date

    def _ratio(self, subpop, pop):
        if type(pop) is int:
            if pop != 0:
                if type(subpop) is int:
                    return subpop/pop
                return len(subpop)/pop
        elif len(pop) != 0:
            if type(subpop) is int:
                return subpop/len(pop)
            return len(subpop)/len(pop)
        return 0

    def _log(self, val):
        if val > 0:
            return math.log(val)
        return None

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

    def _update_HIV_incidence(self, pop: Population):
        # Update HIV incidence by sex and age group
        primary_infection_idx = pop.get_sub_pop([(col.IN_PRIMARY_INFECTION, operator.eq, True)])
        men_idx = pop.get_sub_pop(AND(COND(col.SEX, operator.eq, SexType.Male),
                                      OR(COND(col.NUM_PARTNERS, operator.ge, 1),
                                         COND(col.LONG_TERM_PARTNER, operator.eq, True))))
        women_idx = pop.get_sub_pop(AND(COND(col.SEX, operator.eq, SexType.Female),
                                        OR(COND(col.NUM_PARTNERS, operator.ge, 1),
                                           COND(col.LONG_TERM_PARTNER, operator.eq, True))))

        for age_bound in range(self.age_min, self.age_max, self.age_step):
            age_idx = pop.get_sub_pop([(col.AGE, operator.ge, age_bound),
                                       (col.AGE, operator.lt, age_bound+self.age_step)])
            key = f"HIV incidence ({age_bound}-{age_bound+(self.age_step-1)}, male)"
            total = pop.get_sub_pop_intersection(men_idx, age_idx)
            self.output_stats.loc[self.step, key] = (
                self._ratio(pop.get_sub_pop_intersection(primary_infection_idx, total), total))
            key = f"HIV incidence ({age_bound}-{age_bound+(self.age_step-1)}, female)"
            total = pop.get_sub_pop_intersection(women_idx, age_idx)
            self.output_stats.loc[self.step, key] = (
                self._ratio(pop.get_sub_pop_intersection(primary_infection_idx, total), total))

    def _update_CD4_count(self, pop: Population):
        # Update number of people with given CD4 counts
        cd4_under_200_idx = pop.get_sub_pop([(col.CD4, operator.lt, 200)])
        cd4_200_to_500_idx = pop.get_sub_pop([(col.CD4, operator.ge, 200),
                                              (col.CD4, operator.le, 500)])
        cd4_over_500_idx = pop.get_sub_pop([(col.CD4, operator.gt, 500)])
        self.output_stats.loc[self.step, "CD4 count (under 200)"] = len(cd4_under_200_idx)
        self.output_stats.loc[self.step, "CD4 count (200-500)"] = len(cd4_200_to_500_idx)
        self.output_stats.loc[self.step, "CD4 count (over 500)"] = len(cd4_over_500_idx)

    def _update_infections(self, date, pop: Population):
        # Update infection ratios of long vs short term partners
        recently_infected_idx = pop.get_sub_pop([(col.DATE_HIV_INFECTION, operator.eq, date)])
        self.output_stats.loc[self.step, "Infected by long term partner"] = (
                self._ratio(self.infected_ltp, recently_infected_idx))
        self.output_stats.loc[self.step, "Infected by short term partner"] = (
                self._ratio(self.infected_stp, recently_infected_idx))
        self.output_stats.loc[self.step, "Infected by primary infection"] = (
                self._ratio(self.infected_primary_infection, recently_infected_idx))

        # Reset infection counters
        self.infected_ltp = 0
        self.infected_stp = 0
        self.infected_primary_infection = 0

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

    def _update_partner_sex_balance(self, pop: Population):
        # Update short term partner sex balance statistics
        men_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
        women_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])
        active_idx = pop.get_sub_pop([(col.NUM_PARTNERS, operator.gt, 0)])
        active_men = pop.get_sub_pop_intersection(active_idx, men_idx)
        active_women = pop.get_sub_pop_intersection(active_idx, women_idx)
        # Get flattened lists of partner age groups (values 0-4)
        women_stp_age_list = pop.get_variable(col.STP_AGE_GROUPS, active_women).values
        women_stp_age_list = (np.concatenate(women_stp_age_list).ravel() if len(women_stp_age_list) > 0
                              else women_stp_age_list).tolist()
        men_stp_age_list = pop.get_variable(col.STP_AGE_GROUPS, active_men).values
        men_stp_age_list = (np.concatenate(men_stp_age_list).ravel() if len(men_stp_age_list) > 0
                            else men_stp_age_list).tolist()

        self.output_stats.loc[self.step, "Partner sex balance (male)"] = self._log(
            self._ratio(pop.get_variable(col.NUM_PARTNERS, active_men).sum(),
                        pop.get_variable(col.NUM_PARTNERS, active_women).sum()))
        self.output_stats.loc[self.step, "Partner sex balance (female)"] = self._log(
            self._ratio(pop.get_variable(col.NUM_PARTNERS, active_women).sum(),
                        pop.get_variable(col.NUM_PARTNERS, active_men).sum()))

        # Update short term partner sex balance statistics by age group
        for age_bound in range(self.age_min, self.age_max_active, self.age_step):
            age_group = int(age_bound/10)-1
            age_idx = pop.get_sub_pop([(col.AGE, operator.ge, age_bound),
                                       (col.AGE, operator.lt, age_bound+self.age_step)])
            men_of_age = pop.get_sub_pop_intersection(age_idx, active_men)
            women_of_age = pop.get_sub_pop_intersection(age_idx, active_women)

            key = f"Partner sex balance ({age_bound}-{age_bound+(self.age_step-1)}, male)"
            # Count occurrences of current age group
            women_stp_num = women_stp_age_list.count(age_group)
            self.output_stats.loc[self.step, key] = self._log(
                self._ratio(pop.get_variable(col.NUM_PARTNERS, men_of_age).sum(), women_stp_num))

            key = f"Partner sex balance ({age_bound}-{age_bound+(self.age_step-1)}, female)"
            # Count occurrences of current age group
            men_stp_num = men_stp_age_list.count(age_group)
            self.output_stats.loc[self.step, key] = self._log(
                self._ratio(pop.get_variable(col.NUM_PARTNERS, women_of_age).sum(), men_stp_num))

    def _update_births(self, pop: Population, time_step):
        # Update total births
        born_this_step = pop.get_sub_pop([(col.AGE, operator.ge, 0.25),
                                          (col.AGE, operator.lt, 0.25 + time_step.month / 12)])
        self.output_stats.loc[self.step, "Births (tot)"] = len(born_this_step)

        # Update proportion of women giving birth and infected children
        women_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])
        giving_birth_this_step = pop.get_sub_pop([(col.PREGNANT, operator.eq, True),
                                                  (col.LAST_PREGNANCY_DATE, operator.le,
                                                   pop.date - timedelta(days=270))])
        self.output_stats.loc[self.step, "Births (ratio)"] = self._ratio(giving_birth_this_step, women_idx)
        self.output_stats.loc[self.step, "Births to infected women (tot)"] = len(
            pop.get_sub_pop_intersection(pop.get_sub_pop([(col.HIV_STATUS, operator.eq, True)]),
                                         giving_birth_this_step))
        self.output_stats.loc[self.step, "Infected newborns (ratio)"] = self._ratio(
            self.infected_newborns, born_this_step)
        self.output_stats.loc[self.step, "ANC tests (tot)"] = self.anc_tests
        self.output_stats.loc[self.step, "Labour and delivery tests (tot)"] = self.labdel_tests
        self.output_stats.loc[self.step, "Post-delivery tests (tot)"] = self.postdel_tests

        # Reset infection counter
        self.infected_newborns = 0
        self.anc_tests = 0
        self.labdel_tests = 0
        self.postdel_tests = 0

    def record_HIV_deaths(self, pop: Population, deaths: pd.Series):
        # Update total deaths
        self.output_stats.loc[self.step, "HIV deaths (tot)"] = sum(deaths)
        died_this_step = deaths[deaths].index   # indices of all the people for whom death is true

        # Update deaths by sex and age
        over_15_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15)])
        aged_20_to_59_idx = pop.get_sub_pop([(col.AGE, operator.ge, 20),
                                             (col.AGE, operator.lt, 60)])
        men_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
        women_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])

        self.output_stats.loc[self.step, "HIV deaths (over 15, male)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(men_idx, over_15_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "HIV deaths (over 15, female)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(women_idx, over_15_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "HIV deaths (20-59, male)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(men_idx, aged_20_to_59_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "HIV deaths (20-59, female)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(women_idx, aged_20_to_59_idx),
                                             died_this_step)))

    def record_non_HIV_deaths(self, pop: Population, deaths: pd.Series):
        # Update total deaths
        self.output_stats.loc[self.step, "Non-HIV deaths (tot)"] = sum(deaths)
        died_this_step = deaths[deaths].index   # indices of all the people for whom death is true

        # Update deaths by sex and age
        over_15_idx = pop.get_sub_pop([(col.AGE, operator.ge, 15)])
        aged_20_to_59_idx = pop.get_sub_pop([(col.AGE, operator.ge, 20),
                                             (col.AGE, operator.lt, 60)])
        men_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
        women_idx = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])

        self.output_stats.loc[self.step, "Non-HIV deaths (over 15, male)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(men_idx, over_15_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "Non-HIV deaths (over 15, female)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(women_idx, over_15_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "Non-HIV deaths (20-59, male)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(men_idx, aged_20_to_59_idx),
                                             died_this_step)))
        self.output_stats.loc[self.step, "Non-HIV deaths (20-59, female)"] = (
            len(pop.get_sub_pop_intersection(pop.get_sub_pop_intersection(women_idx, aged_20_to_59_idx),
                                             died_this_step)))

        # Update death totals; could just do this at the end rather than every time step!
        for s in ["(tot)", "(over 15, male)", "(over 15, female)", "(20-59, male)", "(20-59, female)"]:
            self.output_stats.loc[self.step, ("Deaths " + s)] = self.output_stats.loc[self.step, ("HIV deaths " + s)] +\
                                                                self.output_stats.loc[self.step, ("Non-HIV deaths "+s)]

    def update_summary_stats(self, date, pop: Population, time_step):
        self._update_date(date)
        self._update_HIV_prevalence(pop)
        self._update_HIV_incidence(pop)
        self._update_CD4_count(pop)
        self._update_infections(date, pop)
        self._update_circumcision(pop)
        self._update_partners(pop)
        self._update_partner_sex_balance(pop)
        self._update_births(pop, time_step)
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
                plt.close()


# output dataframe initialised by simulation handler
simulation_output = SimulationOutput()
