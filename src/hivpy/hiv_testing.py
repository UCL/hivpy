import hivpy.column_names as col

class HIVTestingModule:

    def __init__(self, **kwargs):
        # FIXME: move this to a yaml later
        self.date_start_testing = 2003.5
        self.covid_disrup_affected = False
        self.testing_disrup_covid = False

    def update_hiv_testing(self, pop):
        """
        Update which individuals in the population have been tested.
        COVID disruption is factored in.
        """
        # testing occurs after a certain year if there is no covid disruption
        if ((pop.date.year > (self.date_start_testing + 5.5))
            & (not self.covid_disrup_affected) & (not self.testing_disrup_covid)):
            # get population that is not hard to reach
            reachable_population = pop.data[(~pop.data[col.HARD_REACH])
                                            & (pop.data[col.AGE] > 0)  # FIXME: is this the correct age range?
                                            & (pop.data[col.DATE_OF_DEATH].isnull())]
            # first time testers
            untested_population = reachable_population[~reachable_population[col.EVER_TESTED]]
            # repeat testers
            prev_tested_population = reachable_population[reachable_population[col.EVER_TESTED]]
