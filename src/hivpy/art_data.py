import numpy as np

from hivpy.exceptions import DataLoadException

from .common import SexType, rng
from .data_reader import DataReader


class ARTData(DataReader):
    """
    Class to hold and interpret sexual behaviour data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.lower_future_art_coverage = self._get_discrete_dist("lower_future_art_coverage")
            self.base_prob_init_ART = self._get_discrete_dist("base_prob_init_ART")
            self.base_prob_switch_line = self._get_discrete_dist("prob_switch_line")
            self.prob_vl_measurement_done = self._get_discrete_dist("prob_vl_measurement_done")
            self.reduced_interrupt_poc_vl = self._get_discrete_dist("reduced_interrupt_poc_vl")
            self.base_rate_interruption = self._get_discrete_dist("base_rate_interruption")
            self.prob_lost_ART = self._get_discrete_dist("prob_lost_ART")
            self.base_rate_restart_ART = self._get_discrete_dist("rate_restart_ART")
            self.prob_supply_interrupted = self.data["prob_supply_interrupted"]
            self.prob_supply_resumed = self.data["prob_supply_resumed"]
            self.base_rate_lost = self._get_discrete_dist["rate_lost"]
            self.base_rate_return = self._get_discrete_dist["rate_return"]
            self.base_rate_return_lencab = self._get_discrete_dist["rate_return_lencab"]

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
