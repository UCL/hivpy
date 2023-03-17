import numpy as np
import yaml

from hivpy.exceptions import DataLoadException

from .common import DiscreteChoice


class HIVTestingData:
    """
    Class to hold and interpret HIV testing data loaded from the yaml file.
    """

    # TODO: This is ripped directly from sex_behaviour_data.py,
    # we should make a new data reader module to store functions like this.
    def _get_discrete_dist(self, *keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]
        vals = np.array(dist_data["Value"])
        probs = np.array(dist_data["Probability"])
        probs /= sum(probs)
        return DiscreteChoice(vals, probs)

    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.data = yaml.safe_load(file)
        try:
            self.date_start_testing = self.data["date_start_testing"]
            self.date_test_rate_plateau = self._get_discrete_dist("date_test_rate_plateau")
            self.an_lin_incr_test = self._get_discrete_dist("an_lin_incr_test")
        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
