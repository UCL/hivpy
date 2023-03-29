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

        self.date_start_testing = self.ht_data.date_start_testing
        self.init_rate_first_test = self.ht_data.init_rate_first_test
        self.date_test_rate_plateau = self.ht_data.date_test_rate_plateau.sample()
        self.an_lin_incr_test = self.ht_data.an_lin_incr_test.sample()

        self.rate_first_test = 0
        self.rate_rep_test = 0
        # FIXME: move this to a yaml later
        self.covid_disrup_affected = False
        self.testing_disrup_covid = False

    def update_hiv_testing(self, pop):
        """
        Update which individuals in the population have been tested.
        COVID disruption is factored in.
        """
        # testing occurs after a certain year if there is no covid disruption
        if ((pop.date.year >= (self.date_start_testing + 5.5))
           & ((not self.covid_disrup_affected) | (not self.testing_disrup_covid))):

            # update testing probabilities
            self.rate_first_test = self.init_rate_first_test + (min(pop.date.year, self.date_test_rate_plateau)
                                                                - (self.date_start_testing + 5.5)) \
                                                                * self.an_lin_incr_test
            self.rate_rep_test = (min(pop.date.year, self.date_test_rate_plateau)
                                  - (self.date_start_testing + 5.5)) * self.an_lin_incr_test

            # get population ready for testing
            testing_population = pop.get_sub_pop([(col.HARD_REACH, op.eq, False),
                                                  (col.AGE, op.ge, 15),
                                                  (col.DATE_OF_DEATH, op.eq, None),
                                                  [(col.LAST_TEST_DATE, op.le, pop.date - timedelta(days=180)),
                                                   (col.LAST_TEST_DATE, op.eq, None)]
                                                  ])

            # first time testers
            untested_population = pop.apply_bool_mask(~pop.data.loc[testing_population, col.EVER_TESTED],
                                                      testing_population)
            # repeat testers
            prev_tested_population = pop.apply_bool_mask(pop.data.loc[testing_population, col.EVER_TESTED],
                                                         testing_population)

            r = rng.uniform(size=len(untested_population))
            tested = r < self.rate_first_test
            # outcomes
            pop.set_present_variable(col.EVER_TESTED, tested, sub_pop=untested_population)
            # set last test date
            pop.set_present_variable(col.LAST_TEST_DATE, pop.date,
                                     sub_pop=pop.apply_bool_mask(tested, untested_population))

            r = rng.uniform(size=len(prev_tested_population))
            tested = r < self.rate_rep_test
            # outcomes
            pop.set_present_variable(col.EVER_TESTED, tested, sub_pop=prev_tested_population)
            # set last test date
            pop.set_present_variable(col.LAST_TEST_DATE, pop.date,
                                     sub_pop=pop.apply_bool_mask(tested, prev_tested_population))
