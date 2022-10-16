import logging
from os import path

LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

_ROOT_NAME = "hivpy"


def setup_logging(log_dir, logfile, fileLogLevel, consoleLogLevel):
    root_logger = logging.getLogger(name=_ROOT_NAME)

    # file logging
    file = path.join(log_dir, logfile)
    root_logger.setLevel(logging.DEBUG)
    file_logger = logging.FileHandler(file, 'w')
    file_formatter = logging.Formatter('%(asctime)s %(name)-15s %(levelname)-10s %(message)s',
                                       datefmt='%y-%d-%m %H:%M:%S')
    file_logger.setFormatter(file_formatter)
    file_logger.setLevel(fileLogLevel)
    root_logger.addHandler(file_logger)
    # console logging
    console_logger = logging.StreamHandler()
    console_formatter = logging.Formatter('%(name)-15s %(levelname)-10s %(message)s')
    console_logger.setFormatter(console_formatter)
    console_logger.setLevel(consoleLogLevel)
    root_logger.addHandler(console_logger)

    print("Starting the simulation. Please, consult the logfile at "+logfile)


def get_logger(name=None):
    qualified_logger_name = _ROOT_NAME + (f".{name.lower()}" if name else "")
    return logging.getLogger(qualified_logger_name)


class PopulationWriter:
    def __init__(self, logger=None):
        self._logger = logger if logger is not None else get_logger()
