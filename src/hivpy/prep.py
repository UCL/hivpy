from enum import IntEnum


class PrEPType(IntEnum):
    Oral = 0
    Cabotegravir = 1  # injectable
    Lenacapavir = 2  # injectable
    VaginalRing = 3


class PrEPModule:

    def __init__(self, **kwargs):
        # FIXME: move these to data file
        self.rate_test_onprep_any = 1
        self.prep_willing_threshold = 0.2
