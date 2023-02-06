import yaml

from .common import SexType


class DemographicsData:
    """
    A class for holding demographics-related data loaded from a file.
    """

    def __init__(self, filename):
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)

        self.female_ratio = data["female_ratio"]

        self.death_age_limits = data["death_rates"]["Age_limits"]
        self.death_rates = {
            SexType.Female: [0] + data["death_rates"]["Female"],
            SexType.Male: [0] + data["death_rates"]["Male"]
        }
