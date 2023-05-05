import importlib.resources
import operator as op
from datetime import timedelta

import hivpy.column_names as col

from .common import rng
from .hiv_testing_data import HIVTestingData


class HIVTestingModule:

    def __init__(self, **kwargs):

        # init hiv testing data
        with importlib.resources.path("hivpy.data", "hiv_testing.yaml") as data_path:
            self.ht_data = HIVTestingData(data_path)

        self.date_start_anc_testing = self.ht_data.date_start_anc_testing
        self.date_start_testing = self.ht_data.date_start_testing
        self.init_rate_first_test = self.ht_data.init_rate_first_test
        self.eff_max_freq_testing = self.ht_data.eff_max_freq_testing
        self.test_scenario = self.ht_data.test_scenario
        self.no_test_if_np0 = self.ht_data.no_test_if_np0
        self.test_targeting = self.ht_data.test_targeting.sample()
        self.date_test_rate_plateau = self.ht_data.date_test_rate_plateau.sample()
        self.an_lin_incr_test = self.ht_data.an_lin_incr_test.sample()

        # eff_max_freq_testing is used as an index to pick the correct
        # minimum number of days to wait between tests from this list
        self.days_to_wait = [365, 180, 90]  # 12 months, 6 months, 3 months

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

    def update_hiv_testing(self, pop):
        """
        Update which individuals in the population have been tested.
        COVID disruption is factored in.
        """
        # testing occurs after a certain year if there is no covid disruption
        if ((pop.date.year >= self.date_start_testing)
           & ((not self.covid_disrup_affected) | (not self.testing_disrup_covid))):

            # update testing probabilities
            self.rate_first_test = self.init_rate_first_test + (min(pop.date.year, self.date_test_rate_plateau)
                                                                - self.date_start_testing) \
                                                                * self.an_lin_incr_test
            self.rate_rep_test = (min(pop.date.year, self.date_test_rate_plateau)
                                  - self.date_start_testing) * self.an_lin_incr_test

            # get population ready for testing
            testing_population = pop.get_sub_pop([(col.HARD_REACH, op.eq, False),
                                                  (col.AGE, op.ge, 15),
                                                  (col.DATE_OF_DEATH, op.eq, None),
                                                  (col.HIV_STATUS, op.eq, False),
                                                  [(col.LAST_TEST_DATE, op.le, pop.date -
                                                    timedelta(days=self.days_to_wait[self.eff_max_freq_testing])),
                                                   (col.LAST_TEST_DATE, op.eq, None)]
                                                  ])

            # first time testers
            untested_population = pop.apply_bool_mask(~pop.get_variable(col.EVER_TESTED, testing_population),
                                                      testing_population)
            # repeat testers
            prev_tested_population = pop.apply_bool_mask(pop.get_variable(col.EVER_TESTED, testing_population),
                                                         testing_population)

            if len(untested_population) > 0:
                # test first time testers
                tested = pop.transform_group([col.EVER_TESTED, col.NP_LAST_TEST, col.NSTP_LAST_TEST],
                                             self.calc_testing_outcomes,
                                             sub_pop=untested_population)
                # set outcomes
                pop.set_present_variable(col.EVER_TESTED, tested, sub_pop=untested_population)
                # set last test date
                pop.set_present_variable(col.LAST_TEST_DATE, pop.date,
                                         sub_pop=pop.apply_bool_mask(tested, untested_population))
                # "reset" dummy partner columns
                pop.set_present_variable(col.NSTP_LAST_TEST, 0,
                                         sub_pop=pop.apply_bool_mask(tested, untested_population))
                pop.set_present_variable(col.NP_LAST_TEST, 0,
                                         sub_pop=pop.apply_bool_mask(tested, untested_population))

            if len(prev_tested_population) > 0:
                # test repeat testers
                tested = pop.transform_group([col.EVER_TESTED, col.NP_LAST_TEST, col.NSTP_LAST_TEST],
                                             self.calc_testing_outcomes,
                                             sub_pop=prev_tested_population)
                # set last test date
                pop.set_present_variable(col.LAST_TEST_DATE, pop.date,
                                         sub_pop=pop.apply_bool_mask(tested, prev_tested_population))
                # "reset" dummy partner columns
                pop.set_present_variable(col.NSTP_LAST_TEST, 0,
                                         sub_pop=pop.apply_bool_mask(tested, prev_tested_population))
                pop.set_present_variable(col.NP_LAST_TEST, 0,
                                         sub_pop=pop.apply_bool_mask(tested, prev_tested_population))

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

        return prob_test

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
