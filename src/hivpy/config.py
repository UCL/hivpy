import logging
import os
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

    @classmethod
    def from_file(cls, output_section):
        outputdir = output_section['OUTPUT_DIRECTORY']
        logfilename = output_section['LOGOUTPUT_NAME']
        log_level = output_section['LOG_LEVEL']
        logpath = os.path.join(outputdir, logfilename)
        return cls(outputdir, logpath, log_level)

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

    @classmethod
    def from_file(cls, simulation_section):
        """Create a configuration from the contents of a file."""
        try:
            start_date = date(int(simulation_section['START_YEAR']), 1, 1)
            end_date = date(int(simulation_section['END_YEAR']), 12, 31)
            population_size = int(simulation_section['POPULATION'])
            interval = timedelta(days=int(simulation_section['TIME_INTERVAL_DAYS']))
            return cls(population_size, start_date, end_date, interval)
        except ValueError as err:
            print('Error parsing the experiment parameters {}'.format(err))
        except KeyError as kerr:
            print('Error extracting values from the parameter set {}'.format(kerr))
        return None

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

    @classmethod
    def from_file(cls, file_config):
        """Create a configuration from the contents of a file."""
        simulation_config = SimulationConfig.from_file(file_config['EXPERIMENT'])
        output_config = OutputConfig.from_file(file_config['OUTPUT'])
        return cls(simulation_config, output_config)
