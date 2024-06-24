import numpy as np

from hivpy.exceptions import DataLoadException

from .common import SexType, rng
from .data_reader import DataReader


class ArtData(DataReader):
    """
    Class to hold and interpret sexual behaviour data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.lower_future_art_coverage = self._get_discrete_dist("lower_future_art_coverage")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
