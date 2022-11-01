from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .exceptions import SimulationException


@dataclass
class LoggingConfig:
    log_dir: str
    logfile: str
    consoleLogging: bool = True
    consoleLogLevel: str = 'WARNING'
    fileLogging: bool = True
    fileLogLevel: str = 'DEBUG'


@dataclass
class SimulationConfig:
    """A class holding the parameters required for running a simulation."""
    population_size: int
    start_date: date
    stop_date: date
    output_dir: Path
    time_step: timedelta = timedelta(days=90)

    def _validate(self):
        """Make sure the values passed in make sense."""
        try:
            assert self.stop_date >= self.start_date + self.time_step
            assert self.time_step > timedelta(days=0)
        except AssertionError:
            raise SimulationException("Invalid simulation configuration.")

    def __post_init__(self):
        """This is called automatically during construction."""
        self._validate()


@dataclass
class ExperimentConfig:
    simulation_config: SimulationConfig
    logging_config: LoggingConfig
