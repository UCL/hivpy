import logging
from dataclasses import dataclass, field
from os import path
from pathlib import Path

from .common import date, timedelta
from .exceptions import SimulationException

LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


@dataclass
class LoggingConfig:
    log_dir: str
    logfile: str
    consoleLogging: bool = True
    consoleLogLevel: str = 'WARNING'
    fileLogging: bool = True
    fileLogLevel: str = 'DEBUG'

    def start_logging(self):
        # file logging
        file = path.join(self.log_dir, self.logfile)
        logging.root.setLevel(logging.DEBUG)
        file_logger = logging.FileHandler(file, 'w')
        file_formatter = logging.Formatter('%(asctime)s %(name)-15s %(levelname)-10s %(message)s',
                                           datefmt='%y-%d-%m %H:%M:%S')
        file_logger.setFormatter(file_formatter)
        file_logger.setLevel(self.fileLogLevel)
        logging.getLogger(name=None).addHandler(file_logger)
        # console logging
        console_logger = logging.StreamHandler()
        console_formatter = logging.Formatter('%(name)-15s %(levelname)-10s %(message)s')
        console_logger.setFormatter(console_formatter)
        console_logger.setLevel(self.consoleLogLevel)
        logging.getLogger(name=None).addHandler(console_logger)

        print("Starting the simulation. Please, consult the logfile at "+self.logfile)


@dataclass
class SimulationConfig:
    """
    A class holding the parameters required for running a simulation.
    """
    population_size: int
    start_date: date
    stop_date: date
    output_dir: Path
    graph_outputs: list
    time_step: timedelta = field(default_factory=lambda: timedelta(days=90))
    intervention_date: date = None
    intervention_option: int = 0

    def _validate(self):
        """
        Make sure the values passed in make sense.
        """
        try:
            assert self.stop_date >= self.start_date + self.time_step
            assert self.time_step > timedelta(days=0)
            if self.intervention_date:
                assert self.intervention_date >= self.start_date + self.time_step
                assert self.intervention_date <= self.stop_date - self.time_step
        except AssertionError:
            raise SimulationException("Invalid simulation configuration.")

    def __post_init__(self):
        """
        This is called automatically during construction.
        """
        self._validate()


@dataclass
class ExperimentConfig:
    simulation_config: SimulationConfig
    logging_config: LoggingConfig
