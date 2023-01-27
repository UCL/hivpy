import numpy as np
import yaml

from hivpy.exceptions import DataLoadException

from .common import DiscreteChoice


class PregnancyData:
    """
    Class to hold and interpret pregnancy data loaded from the yaml file.
    """

    # TODO: This is ripped directly from sex_behaviour_data.py,
    # we should make a new data reader module to store functions like this.
    def _setup_probabilty_dist(self, prob_dict):
        if ("Range" in prob_dict):
            min = prob_dict["Range"][0]
            max = prob_dict["Range"][1]
            N = max - min + 1
            return DiscreteChoice(np.arange(min, max+1, 1), np.array([1./N]*N))
        else:
            values = np.array(prob_dict["Value"])
            probs = np.array(prob_dict["Probability"])
            probs /= sum(probs)
            return DiscreteChoice(values, probs)

    def _get_discrete_dist_list(self, *keys):
        dist_list = self.data
        for k in keys:
            dist_list = dist_list[k]
        return np.array([self._setup_probabilty_dist(x) for x in dist_list])

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
            self.can_be_pregnant = self.data["can_be_pregnant"]
            self.fold_preg = self.data["fold_preg"]
            self.inc_cat = self._get_discrete_dist("inc_cat")
            self.rate_birth_with_infected_child = self._get_discrete_dist("rate_birth_with_infected_child")
            self.init_num_children_distributions = self._get_discrete_dist_list("init_num_children_distributions")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
