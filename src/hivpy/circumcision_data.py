import yaml
import numpy as np

from hivpy.exceptions import DataLoadException

from .common import DiscreteChoice


class CircumcisionData:
    """Class to hold and interpret circumcision data loaded from the yaml file"""

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
            self.circ_inc_rate = self._get_discrete_dist("circ_inc_rate")
            self.rel_incr_circ_post_2013 = self._get_discrete_dist("rel_incr_circ_post_2013")
            self.circ_inc_15_19 = self._get_discrete_dist("circ_inc_15_19")
            self.circ_red_20_30 = self._get_discrete_dist("circ_red_20_30")
            self.circ_red_30_50 = self._get_discrete_dist("circ_red_30_50")
            self.prob_birth_circ = self._get_discrete_dist("prob_birth_circ")
        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
