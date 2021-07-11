import random

import numpy as np
import pandas as pd

class Population:
    """A set of individuals with particular characteristics."""
    size: int  # how many individuals to create in total
    data: pd.DataFrame  # the underlying data
    params: dict  # population-level parameters

    def __init__(self, size):
        """Initialise a population of the given size."""
        self.size = size
        self._sample_parameters()
        self._create_population_data()

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
        self.data = pd.DataFrame({
            'max_age': max_age.astype(int)
        })
        