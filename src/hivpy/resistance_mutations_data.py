from hivpy.exceptions import DataLoadException

from .data_reader import DataReader


class ResistanceMutationsData(DataReader):
    """
    Class to hold and interpret resistance mutations data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.min_vl_on_art = self.data["min_vl_on_art"]
            self.vl_stdev_on_art = self.data["vl_stdev_on_art"]
            self.cd4_recovery_pi_factor = self.data["cd4_recovery_pi_factor"]
            self.cd4_recovery_female_factor = self.data["cd4_recovery_female_factor"]
            self.cd4_stdev_on_art = self.data["cd4_stdev_on_art"]

            self.mutation_risk_change = self._get_discrete_dist("mutation_risk_change")
            self.risk_change_tams_resist = self.data["risk_change_tams_resist"]
            self.risk_change_151_resist = self.data["risk_change_151_resist"]
            self.ten_resist_rate = self._get_discrete_dist("ten_resist_rate")
            self.dol_resist_rate = self._get_discrete_dist("dol_resist_rate")
            self.len_resist_rate = self._get_discrete_dist("len_resist_rate")
            self.incr_len_resist = self.data["incr_len_resist"]
            self.cab_resist_factor = self._get_discrete_dist("cab_resist_factor")
            self.risk_change_cab_resist = self._get_discrete_dist("risk_change_cab_resist")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
