import importlib.resources
import operator as op

import hivpy.column_names as col

from .common import AND, COND, OR, rng, timedelta, floatToDate, diff_years
from .hiv_testing_data import HIVTestingData


class HIVTestingModule:

    def __init__(self, **kwargs):

        # init hiv testing data
        with importlib.resources.path("hivpy.data", "hiv_testing.yaml") as data_path:
            self.ht_data = HIVTestingData(data_path)

        self.date_start_testing = floatToDate(self.ht_data.date_start_testing)
        self.date_rate_testing_incr = floatToDate(self.ht_data.date_rate_testing_incr)
        self.init_rate_first_test = self.ht_data.init_rate_first_test
        self.eff_max_freq_testing = self.ht_data.eff_max_freq_testing
        self.test_scenario = self.ht_data.test_scenario
        self.no_test_if_np0 = self.ht_data.no_test_if_np0

        self.prob_anc_test_trim1 = self.ht_data.prob_anc_test_trim1
        self.prob_anc_test_trim2 = self.ht_data.prob_anc_test_trim2.sample()
        self.prob_anc_test_trim3 = self.ht_data.prob_anc_test_trim3
        self.prob_test_postdel = self.ht_data.prob_test_postdel

        self.prob_test_non_hiv_symptoms = self.ht_data.prob_test_non_hiv_symptoms
        self.prob_test_who4 = self.ht_data.prob_test_who4
        self.prob_test_tb = self.ht_data.prob_test_tb
        self.prob_test_non_tb_who3 = self.ht_data.prob_test_non_tb_who3
        self.test_targeting = self.ht_data.test_targeting.sample()
        self.date_general_testing_plateau = floatToDate(self.ht_data.date_general_testing_plateau.sample())
        self.date_targeted_testing_plateau = floatToDate(self.ht_data.date_targeted_testing_plateau)
        self.an_lin_incr_test = self.ht_data.an_lin_incr_test.sample()
        self.incr_test_rate_sympt = self.ht_data.incr_test_rate_sympt.sample()

        # eff_max_freq_testing is used as an index to pick the correct
        # minimum number of days to wait between tests from this list
        self.months_to_wait = [12, 6, 3]  # 12 months, 6 months, 3 months

        self.rate_first_test = 0
        self.rate_rep_test = 0
        # TODO: meant to be a personal variable but currently always just set to test_targeting
        self.eff_test_targeting = self.test_targeting
        # scenario has a high chance to change eff_test_targeting
        if self.test_scenario == 1:
            r = rng.uniform()
            if r < 0.45:
                self.eff_test_targeting = 2
            elif r < 0.9:
                self.eff_test_targeting = 5
        # FIXME: move this to a yaml later
        self.covid_disrup_affected = False
        self.testing_disrup_covid = False
        # sex workers regularly test every 6 months
        self.sw_test_regularly = False

    def update_hiv_testing(self, pop, time_step: timedelta):
        """
        Update which individuals in the population have been tested.
        COVID disruption is factored in.
        """
        # mark people for testing
        # anc > hiv symptomatic > non hiv symptomatic > vmmc > self testing > general > prep
        self.test_mark_anc(pop, time_step)
        self.test_mark_hiv_symptomatic(pop)
        self.test_mark_non_hiv_symptomatic(pop)
        self.test_mark_vmmc(pop, time_step)
        self.test_mark_general_pop(pop)

        # apply testing to marked population
        marked_population = pop.get_sub_pop([(col.TEST_MARK, op.eq, True)])
        self.apply_test_outcomes_to_sub_pop(pop, marked_population)

    def test_mark_hiv_symptomatic(self, pop):
        """
        Mark HIV symptomatic individuals to undergo testing this time step.

        Note: If the simulation is started after date_targeted_testing_plateau
        (or if date_start_testing is set to be after date_targeted_testing_plateau)
        the HIV symptomatic testing probabilities will not be updated
        and this function may not work as expected.
        """
        # testing occurs after a certain year
        if (pop.date > self.date_start_testing):

            # date needed for a function passed to transform_group
            self.date = pop.date

            # update symptomatic test probabilities
            if pop.date <= self.date_targeted_testing_plateau:
                self.prob_test_who4 = min(0.9, self.prob_test_who4 * self.incr_test_rate_sympt)
                self.prob_test_tb = min(0.8, self.prob_test_tb * self.incr_test_rate_sympt)
                self.prob_test_non_tb_who3 = min(0.7, self.prob_test_non_tb_who3 * self.incr_test_rate_sympt)

            # undiagnosed population not scheduled for testing
            not_diag_tested_pop = pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, False),
                                                   (col.TEST_MARK, op.eq, False)])

            if len(not_diag_tested_pop) > 0:
                # mark people for testing
                marked = pop.transform_group([col.ADC, col.TB_INITIAL_INFECTION, col.NON_TB_WHO3],
                                             self.calc_symptomatic_testing_outcomes,
                                             sub_pop=not_diag_tested_pop)
                # set outcomes
                pop.set_present_variable(col.TEST_MARK, marked, not_diag_tested_pop)

    def calc_symptomatic_testing_outcomes(self, adc, tb_initial_infection, non_tb_who3, size):
        """
        Uses the symptomatic test probability for a given group
        of symptoms to select individuals marked to be tested.
        """
        prob_test = self.calc_symptomatic_prob_test(adc, tb_initial_infection, non_tb_who3)
        # outcomes
        r = rng.uniform(size=size)
        marked = r < prob_test

        return marked

    def calc_symptomatic_prob_test(self, adc, tb_initial_infection, non_tb_who3):
        """
        Calculates the probability of being tested for a group
        with specific symptoms and returns it. Presence of an AIDS
        defining condition (ADC; any WHO4), tuberculosis (TB), and a
        non-TB WHO3 disease all affect groupings and test probability.
        """
        # assume asymptomatic by default
        prob_test = 0
        # presence of ADC
        if adc:
            prob_test = self.prob_test_who4
        # presence of TB, no ADC
        elif tb_initial_infection:
            prob_test = self.prob_test_tb
        # presence of a non-TB WHO3 disease, no ADC or TB
        elif non_tb_who3 and not tb_initial_infection:
            prob_test = self.prob_test_non_tb_who3

        return prob_test

    def test_mark_non_hiv_symptomatic(self, pop):
        """
        Mark non-HIV symptomatic individuals to undergo testing this time step.
        """
        # testing occurs after a certain year if there is no covid disruption
        if ((pop.date >= self.date_start_testing)
           & (not (self.covid_disrup_affected | self.testing_disrup_covid))):

            # undiagnosed (last time step) population not scheduled for testing
            not_diag_tested_pop = pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, False),
                                                   (col.TEST_MARK, op.eq, False)])

            if len(not_diag_tested_pop) > 0:
                # mark people for testing
                r = rng.uniform(size=len(not_diag_tested_pop))
                s = rng.uniform(size=len(not_diag_tested_pop))
                marked = ((r < (self.prob_test_non_tb_who3 + self.prob_test_who4)/2) &
                          (s < self.prob_test_non_hiv_symptoms))
                # set outcomes
                pop.set_present_variable(col.TEST_MARK, marked, not_diag_tested_pop)

    def test_mark_vmmc(self, pop, time_step):
        """
        Mark recently circumcised individuals to undergo testing this time step.
        """
        # those that just got circumcised and weren't tested last time step get tested now
        testing_population = pop.get_sub_pop(AND(COND(col.VMMC, op.eq, True),
                                                 COND(col.CIRCUMCISION_DATE, op.eq, pop.date),
                                                 OR(COND(col.LAST_TEST_DATE, op.lt, pop.date - time_step),
                                                    COND(col.LAST_TEST_DATE, op.eq, None)),
                                                 COND(col.TEST_MARK, op.eq, False)))
        # mark people for testing
        if len(testing_population) > 0:
            pop.set_present_variable(col.TEST_MARK, True, testing_population)

    def test_mark_anc(self, pop, time_step):
        """
        Mark which pregnant women are tested while in antenatal care.
        COVID disruption is factored in.
        """
        # conduct up to three tests in anc during pregnancy if there is no covid disruption
        if not (self.covid_disrup_affected | self.testing_disrup_covid):
            # get population at the end of the first trimester
            first_trimester_pop = pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, False),
                                                   (col.ANC, op.eq, True),
                                                   (col.LAST_PREGNANCY_DATE, op.le, pop.date
                                                    - timedelta(days=90)),
                                                   (col.LAST_PREGNANCY_DATE, op.gt, pop.date
                                                    - (timedelta(days=90) + time_step))])
            self.update_sub_pop_test_mark(pop, first_trimester_pop, self.prob_anc_test_trim1)

            # get population at the end of the second trimester
            second_trimester_pop = pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, False),
                                                    (col.ANC, op.eq, True),
                                                    (col.LAST_PREGNANCY_DATE, op.le, pop.date
                                                     - timedelta(days=180)),
                                                    (col.LAST_PREGNANCY_DATE, op.gt, pop.date
                                                     - (timedelta(days=180) + time_step))])
            self.update_sub_pop_test_mark(pop, second_trimester_pop, self.prob_anc_test_trim2)

            # get population at the end of the third trimester
            third_trimester_pop = pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, False),
                                                   (col.ANC, op.eq, True),
                                                   (col.LAST_PREGNANCY_DATE, op.le, pop.date
                                                    - timedelta(days=270))])
            self.update_sub_pop_test_mark(pop, third_trimester_pop, self.prob_anc_test_trim3)

            # get post-delivery population tested during the previous time step
            post_delivery_pop = pop.get_sub_pop([(col.HIV_DIAGNOSED, op.eq, False),
                                                 (col.LAST_TEST_DATE, op.eq, pop.date - time_step),
                                                 (col.LAST_PREGNANCY_DATE, op.eq, pop.date
                                                  - (timedelta(days=270) + time_step))])
            self.update_sub_pop_test_mark(pop, post_delivery_pop, self.prob_test_postdel)

    def update_sub_pop_test_mark(self, pop, sub_pop, prob_test):
        """
        Update the test mark of a sub-population based on a given probability.
        """
        if len(sub_pop) > 0:
            # mark people for testing
            r = rng.uniform(size=len(sub_pop))
            marked = r < prob_test
            # set outcomes
            pop.set_present_variable(col.TEST_MARK, True, sub_pop=pop.apply_bool_mask(marked, sub_pop))

    def test_mark_general_pop(self, pop):
        """
        Mark general population to undergo testing this time step.
        """
        # testing occurs after a certain year if there is no covid disruption
        if ((pop.date >= self.date_rate_testing_incr)
           & (not (self.covid_disrup_affected | self.testing_disrup_covid))):

            # mark sex workers for testing first
            self.test_mark_sex_workers(pop)

            # update testing probabilities
            self.rate_rep_test = (min(pop.date, self.date_general_testing_plateau) - 
                                            self.date_rate_testing_incr).years() * self.an_lin_incr_test
            self.rate_first_test = self.init_rate_first_test + self.rate_rep_test

            # general population ready for testing
            testing_population = pop.get_sub_pop(AND(COND(col.HARD_REACH, op.eq, False),
                                                     COND(col.AGE, op.ge, 15),
                                                     COND(col.HIV_DIAGNOSED, op.eq, False),
                                                     OR(COND(col.LAST_TEST_DATE, op.le, pop.date -
                                                             timedelta(months=self.months_to_wait[
                                                                 self.eff_max_freq_testing])),
                                                        COND(col.LAST_TEST_DATE, op.eq, None)),
                                                     COND(col.TEST_MARK, op.eq, False)))

            if len(testing_population) > 0:
                # mark people for testing
                marked = pop.transform_group([col.EVER_TESTED, col.NP_LAST_TEST, col.NSTP_LAST_TEST],
                                             self.calc_testing_outcomes, sub_pop=testing_population)
                # set outcomes
                pop.set_present_variable(col.TEST_MARK, marked, testing_population)

    def test_mark_sex_workers(self, pop):
        """
        Mark sex workers to undergo testing this time step.
        """
        # testing occurs if sex workers regularly test every 6 months
        if self.sw_test_regularly:
            # sex workers ready for testing
            testing_population = pop.get_sub_pop(AND(COND(col.SEX_WORKER, op.eq, True),
                                                     COND(col.HIV_DIAGNOSED, op.eq, False),
                                                     OR(COND(col.LAST_TEST_DATE, op.le, pop.date -
                                                             timedelta(days=180)),
                                                        COND(col.LAST_TEST_DATE, op.eq, None)),
                                                     COND(col.TEST_MARK, op.eq, False)))
            # mark people for testing
            if len(testing_population) > 0:
                pop.set_present_variable(col.TEST_MARK, True, testing_population)

    def calc_testing_outcomes(self, repeat_tester, np_last_test, nstp_last_test, size):
        """
        Uses the HIV test probability for either
        first-time or repeat testers to return testing outcomes.
        """
        prob_test = self.calc_prob_test(repeat_tester, np_last_test, nstp_last_test)
        # outcomes
        r = rng.uniform(size=size)
        tested = r < prob_test

        return tested

    def calc_prob_test(self, repeat_tester, np_last_test, nstp_last_test):
        """
        Calculates the probability of an individual getting
        tested for HIV based on whether they are a first-time
        or repeat tester and returns it. The number of
        condomless sex partners since a person's last test
        also affects their testing probability.
        """
        # get base probability (assumes repeat tester by default)
        prob_test = self.rate_rep_test
        # first-time testers may have a different base probability
        if not repeat_tester:
            prob_test = self.rate_first_test

        # adjust according to number of partners
        if np_last_test == 0:
            if self.no_test_if_np0:
                prob_test = 0
            else:
                prob_test /= self.eff_test_targeting
        elif nstp_last_test >= 1:
            prob_test *= self.eff_test_targeting

        return min(prob_test, 1)

    def apply_test_outcomes_to_sub_pop(self, pop, sub_pop):
        """
        Sets HIV testing outcomes for a given sub-population
        and resets number of partners since last test.
        """
        # set ever tested
        pop.set_present_variable(col.EVER_TESTED, True, sub_pop)
        # set last test date
        pop.set_present_variable(col.LAST_TEST_DATE, pop.date, sub_pop)
        # "reset" dummy partner columns
        pop.set_present_variable(col.NSTP_LAST_TEST, 0, sub_pop)
        pop.set_present_variable(col.NP_LAST_TEST, 0, sub_pop)
        # exhaust test marks
        pop.set_present_variable(col.TEST_MARK, False, sub_pop)
