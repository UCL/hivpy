import datetime
import functools
from typing import Callable, Dict

import pandas as pd

from .common import SexType
from .demographics import DemographicsModule
from .hiv_status import HIVStatusModule
from .sexual_behaviour import SexualBehaviourModule


class Population:
    """A set of individuals with particular characteristics."""
    size: int  # how many individuals to create in total
    data: pd.DataFrame  # the underlying data
    params: dict  # population-level parameters
    date: datetime.date  # current date
    attributes: Dict[str, Callable]  # aggregate measures across the population

    def __init__(self, size, start_date):
        """Initialise a population of the given size."""
        self.size = size
        self.date = start_date
        self.demographics = DemographicsModule()
        self.sexual_behaviour = SexualBehaviourModule()
        self.hiv_status = HIVStatusModule()
        self._sample_parameters()
        self._create_population_data()
        self._create_attributes()

    def _sample_parameters(self):
        """Randomly determine the uncertain population-level parameters."""
        # Example: Each person will have a predetermined max age,
        # which will come from a normal distribution. The mean of
        # that distrubition is chosen randomly for each population.
        # avg_max_age = random.choices([80, 85, 90], [0.4, 0.4, 0.2])
        # self.params = {
        #     'avg_max_age': avg_max_age,
        # }

    def _create_population_data(self):
        """Populate the data frame with initial values."""
        # NB This is a prototype. We should use the new numpy random interface:
        # https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
        date_of_death = [None] * self.size
        self.data = pd.DataFrame({
            'sex': self.demographics.initialize_sex(self.size),
            'age': self.demographics.initialise_age(self.size),
            'date_of_death': date_of_death
        })
        self.data['HIV_status'] = self.hiv_status.initial_HIV_status(self.data)
        self.data['num_partners'] = 0
        self.sexual_behaviour.init_sex_behaviour_groups(self.data)
        self.sexual_behaviour.init_risk_factors(self.data)
        self.sexual_behaviour.num_short_term_partners(self.data)

    def _create_attributes(self):
        """Determine what aggregate measures can be computed and how."""
        attributes = {}

        # Helper functions
        def count_alive(population_data):
            """Count how many people don't have a set date of death yet."""
            return self.size - population_data.date_of_death.count()

        def count_sex_alive(population_data, sex):
            """Count how many people of the given sex are alive."""
            assert sex in SexType
            # Should also make sure they are alive! (always true for now)
            alive = population_data.date_of_death is not None
            return population_data[alive].sex.value_counts()[sex]

        attributes["num_alive"] = count_alive
        attributes["num_male"] = functools.partial(count_sex_alive, sex="male")
        attributes["num_female"] = functools.partial(count_sex_alive, sex="female")
        self.attributes = attributes

    def has_attribute(self, attribute_name):
        """Return whether the named attribute exists in the population."""
        return attribute_name in self.attributes

    def get(self, attribute_name):
        """Get the value of the named attribute at the current date."""
        return self.attributes[attribute_name](self.data)

    def evolve(self, time_step: datetime.timedelta):
        """Advance the population by one time step."""
        # Does nothing just yet except advance the current date, track ages
        # and set death dates.
        self.data.age += time_step.days / 365  # Very naive!
        # Record who has reached their max age
        died_this_period = self.demographics.determine_deaths(self.data)
        self.data.loc[died_this_period, "date_of_death"] = self.date

        # Get the number of sexual partners this time step
        self.sexual_behaviour.update_sex_behaviour(self.data)
        self.hiv_status.update_HIV_status(self.data)
        # We should think about whether we want to return a copy or evolve
        # the population in-place. We will likely need a copy at some point.
        self.date += time_step
        return self
