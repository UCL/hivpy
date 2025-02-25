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
            self.risk_change_cab_resist = self._get_discrete_dist("risk_change_cab_resist")

            self.resist_rate_tams_higher = self.data["resist_rate_tams_higher"]
            self.resist_rate_tams_lower = self.data["resist_rate_tams_lower"]
            self.resist_rate_nev_higher = self.data["resist_rate_nev_higher"]
            self.resist_rate_nev_lower = self.data["resist_rate_nev_lower"]
            self.resist_rate_efa_higher = self.data["resist_rate_efa_higher"]
            self.resist_rate_efa_lower = self.data["resist_rate_efa_lower"]
            self.resist_rate_lpr_higher = self.data["resist_rate_lpr_higher"]
            self.resist_rate_lpr_lower = self.data["resist_rate_lpr_lower"]

            self.resist_rate_zdv = self.data["resist_rate_zdv"]
            self.resist_rate_3tc = self.data["resist_rate_3tc"]
            self.resist_rate_dar = self.data["resist_rate_dar"]
            self.resist_rate_taz = self.data["resist_rate_taz"]
            self.resist_rate_isl = self.data["resist_rate_isl"]
            self.resist_rate_ten = self._get_discrete_dist("resist_rate_ten")
            self.resist_rate_dol = self._get_discrete_dist("resist_rate_dol")
            self.resist_rate_len = self._get_discrete_dist("resist_rate_len")

            self.incr_len_resist = self.data["incr_len_resist"]
            self.cab_resist_factor = self._get_discrete_dist("cab_resist_factor")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
