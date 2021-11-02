import numpy as np
from pandas.api.types import CategoricalDtype

SexType = CategoricalDtype(["female", "male"])


# Default values
# Can wrap those in something that ensures they have a description?
FEMALE_RATIO = 0.52


class DemographicsModule:

    def __init__(self, **kwargs):
        params = {
            "female_ratio": FEMALE_RATIO
        }
        # allow setting some parameters explicitly
        # could be useful if we have another method for more complex initialization,
        # e.g. from a config file
        for param,  value in kwargs.items():
            assert param in params, f"{param} is not related to this module."
            params[param] = value
        self.params = params

    def initialize_sex(self, count):
        sex_distribution = (
            self.params['female_ratio'], 1 - self.params['female_ratio'])
        return np.random.choice(SexType.categories, count, sex_distribution)
