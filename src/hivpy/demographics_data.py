from hivpy.exceptions import DataLoadException

from .common import SexType
from .data_reader import DataReader


class DemographicsData(DataReader):
    """
    Class to hold and interpret demographics data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.female_ratio = self.data["female_ratio"]
            self.death_age_limits = self.data["death_rates"]["Age_limits"]
            self.death_rates = {
                SexType.Female: [0] + self.data["death_rates"]["Female"],
                SexType.Male: [0] + self.data["death_rates"]["Male"]
            }

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
