import datetime

import pandas as pd

import hivpy.column_names as col

from .circumcision import CircumcisionModule
from .demographics import DemographicsModule
from .hiv_status import HIVStatusModule
from .sexual_behaviour import SexualBehaviourModule

HIV_APPEARANCE = datetime.date(1989, 1, 1)


class Population:
    """
    A set of individuals with particular characteristics.
    """
    size: int  # how many individuals to create in total
    data: pd.DataFrame  # the underlying data
    params: dict  # population-level parameters
    date: datetime.date  # current date
    HIV_introduced: bool  # whether HIV has been introduced yet

    def __init__(self, size, start_date):
        """
        Initialise a population of the given size.
        """
        self.size = size
        self.date = start_date
        self.demographics = DemographicsModule()
        self.sexual_behaviour = SexualBehaviourModule()
        self.circumcision = CircumcisionModule()
        self.hiv_status = HIVStatusModule()
        self.HIV_introduced = False
        self._sample_parameters()
        self._create_population_data()

    def _sample_parameters(self):
        """
        Randomly determine the uncertain population-level parameters.
        """
        # Example: Each person will have a predetermined max age,
        # which will come from a normal distribution. The mean of
        # that distrubition is chosen randomly for each population.
        # avg_max_age = random.choices([80, 85, 90], [0.4, 0.4, 0.2])
        # self.params = {
        #     'avg_max_age': avg_max_age,
        # }

    def _create_population_data(self):
        """
        Populate the data frame with initial values.
        """
        # NB This is a prototype. We should use the new numpy random interface:
        # https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
        self.data = pd.DataFrame({
            col.SEX: self.demographics.initialise_sex(self.size),
            col.AGE: self.demographics.initialise_age(self.size),
            col.DATE_OF_DEATH: [None] * self.size
        })
        self.data[col.CIRCUMCISED] = False
        self.data[col.CIRCUMCISION_DATE] = None
        self.data[col.VMMC] = False
        self.data[col.HARD_REACH] = False
        self.data[col.HIV_STATUS] = self.hiv_status.initial_HIV_status(self.data)
        self.data[col.HIV_DIAGNOSIS_DATE] = None
        self.data[col.NUM_PARTNERS] = 0
        self.data[col.LONG_TERM_PARTNER] = False
        self.data[col.LTP_LONGEVITY] = 0
        self.demographics.initialise_hard_reach(self.data)
        if self.circumcision.vmmc_disrup_covid:
            self.circumcision.init_birth_circumcision_born(self.data, self.date)
        else:
            self.circumcision.init_birth_circumcision_all(self.data, self.date)
        self.sexual_behaviour.init_sex_behaviour_groups(self.data)
        self.sexual_behaviour.init_risk_factors(self.data)
        self.sexual_behaviour.num_short_term_partners(self)
        self.sexual_behaviour.assign_stp_ages(self)
        # TEMP
        self.hiv_status.set_dummy_viral_load(self)
        # If we are at the start of the epidemic, introduce HIV into the population.
        if self.date >= HIV_APPEARANCE and not self.HIV_introduced:
            self.data[col.HIV_STATUS] = self.hiv_status.introduce_HIV(self.data)
            self.HIV_introduced = True

    def transform_group(self, param_list, func, use_size=True, sub_pop=None):
        """
        Groups the data by a list of parameters and applies a function to each grouping.

        `param_list` is a list of names of columns by which you want to group the data. The order
        must match the order of arguments taken by the function `func`. \n
        `func` is a function which takes the values of those columns for a group (and optionally
        the size of the group, which should be the last argument) and returns a value or array of
        values of the size of the group. \n
        `use_size` is true by default, but should be set to false if `func` does not take the size
        of the group as an argument. \n
        `sub_pop` is `None` by default, in which case the transform acts upon the entire dataframe.
        If `sub_pop` is defined, then it acts only on the part of the dataframe defined
        by `data.loc[sub_pop]`.
        """
        # HIV_STATUS is just a dummy column to allow us to use the transform method
        def general_func(g):
            if len(param_list) == 1:
                args = [g.name]
            else:
                args = list(g.name)
            if (use_size):
                args.append(g.size)
            return func(*args)
        if sub_pop is not None:
            df = self.data.loc[sub_pop]
        else:
            df = self.data
        return df.groupby(param_list)[col.HIV_STATUS].transform(general_func)

    def evolve(self, time_step: datetime.timedelta):
        """
        Advance the population by one time step.
        """
        # Does nothing just yet except advance the current date, track ages
        # and set death dates.
        self.data.age += time_step.days / 365  # Very naive!
        # Record who has reached their max age
        died_this_period = self.demographics.determine_deaths(self)
        self.data.loc[died_this_period, col.DATE_OF_DEATH] = self.date

        if self.circumcision.vmmc_disrup_covid:
            self.circumcision.update_birth_circumcision(self.data, time_step, self.date)
        self.circumcision.update_vmmc(self)
        # Get the number of sexual partners this time step
        self.sexual_behaviour.update_sex_behaviour(self)
        self.hiv_status.update_HIV_status(self)

        # If we are at the start of the epidemic, introduce HIV into the population.
        if self.date >= HIV_APPEARANCE and not self.HIV_introduced:
            self.data[col.HIV_STATUS] = self.hiv_status.introduce_HIV(self.data)
            self.HIV_introduced = True

        # We should think about whether we want to return a copy or evolve
        # the population in-place. We will likely need a copy at some point.
        self.date += time_step
        return self
