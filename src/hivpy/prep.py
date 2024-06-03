from enum import IntEnum


class PrEPType(IntEnum):
    Oral = 1
    Injectable = 2
    VaginalRing = 3


class PrEPModule:

    def __init__(self, **kwargs):
        # FIXME: move these to data file
        self.rate_test_onprep_any = 1
        self.prep_willing_threshold = 0.2
