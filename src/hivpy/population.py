import datetime
import random
from typing import Callable, Dict

import numpy as np
import pandas as pd

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
        self._sample_parameters()
        self._create_population_data()
        self._create_attributes()

    def _sample_parameters(self):
        """Randomly determine the uncertain population-level parameters."""
        # Example: Each person will have a predetermined max age,
        # which will come from a normal distribution. The mean of
        # that distrubition is chosen randomly for each population. 
        avg_max_age = random.choices([80, 85, 90], [0.4, 0.4, 0.2])
        self.params = {
            'avg_max_age': avg_max_age
        }

    def _create_population_data(self):
        """Populate the data frame with initial values."""
        # NB This is a prototype. We should use the new numpy random interface:
        # https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
        max_age = self.params['avg_max_age'] + 2 * np.random.randn(self.size)
        # For now, age will be between 0 and max_age for each person
        # (i.e. we're assuming everyone is born at the start of the simulation)
        age = np.random.random(self.size) * max_age
        date_of_death = [None] * self.size
        self.data = pd.DataFrame({
            'max_age': max_age.astype(int),
            'age': age.astype(int),
            'date_of_death': date_of_death
        })

    def _create_attributes(self):
        """Determine what aggregate measures can be computed and how."""
        attributes = {}

        # Helper functions
        def count_alive(population_data):
            """Count how many people don't have a set date of death yet."""
            return self.size - population_data.date_of_death.count()

        attributes["num_alive"] = count_alive
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
        self.data.age += time_step.days / 365 # Very naive!
        # Record who has reached their max age
        died_this_period = self.data.age >= self.data.max_age
        self.data.loc[died_this_period, "date_of_death"] = self.date
        # We should think about whether we want to return a copy or evolve
        # the population in-place. We will likely need a copy at some point.
        self.date += time_step
        return self
        