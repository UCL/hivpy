import logging
from os import path

from .config import LoggingConfig

LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

_ROOT_NAME = "hivpy"


def start_logging(config: LoggingConfig):
    root_logger = logging.getLogger(name=_ROOT_NAME)

    # file logging
    file = path.join(config.log_dir, config.logfile)
    root_logger.setLevel(logging.DEBUG)
    file_logger = logging.FileHandler(file, 'w')
    file_formatter = logging.Formatter('%(asctime)s %(name)-15s %(levelname)-10s %(message)s',
                                       datefmt='%y-%d-%m %H:%M:%S')
    file_logger.setFormatter(file_formatter)
    file_logger.setLevel(config.fileLogLevel)
    root_logger.addHandler(file_logger)
    # console logging
    console_logger = logging.StreamHandler()
    console_formatter = logging.Formatter('%(name)-15s %(levelname)-10s %(message)s')
    console_logger.setFormatter(console_formatter)
    console_logger.setLevel(config.consoleLogLevel)
    root_logger.addHandler(console_logger)

    print("Starting the simulation. Please, consult the logfile at "+config.logfile)


def get_logger(name=None):
    qualified_logger_name = _ROOT_NAME + (f".{name.lower()}" if name else "")
    return logging.getLogger(qualified_logger_name)


class HIVpyLogger:
    def __init__(self, base_logger=None):
        self._logger = base_logger if base_logger is not None else get_logger()

    def write_population(self, population):
        population  # stand-in to satisfy linter
        self._logger.info("This is where the population will go.")
