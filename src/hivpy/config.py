import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List

from .exceptions import SimulationException

LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


@dataclass
class OutputConfig:
    output_dir: str
    logfile: str
    loglevel: str

    def start_logging(self):
        logging.basicConfig(filename=self.logfile, level=LEVELS[self.loglevel])
        logging.info("starting experiment")
        print("Starting the simulation. Please, consult the logfile at "+self.logfile)


@dataclass
class SimulationConfig:
    """A class holding the parameters required for running a simulation."""
    population_size: int
    start_date: date
    stop_date: date
    time_step: timedelta = timedelta(days=90)
    tracked: List[str] = field(default_factory=list)

    def _validate(self):
        """Make sure the values passed in make sense."""
        try:
            assert self.stop_date >= self.start_date + self.time_step
        except AssertionError:
            raise SimulationException("Invalid simulation configuration.")

    def __post_init__(self):
        """This is called automatically during construction."""
        self._validate()

    def track(self, attribute_name):
        """Track an additional attribute during simulation."""
        # TODO Check if already tracked (or conver tracked to a set?)
        self.tracked.append(attribute_name)


@dataclass
class ExperimentConfig:
    simulation_config: SimulationConfig
    output_config: OutputConfig
