import logging
from os import path

LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


class HIVpyLogger(logging.Logger):
    def __init__(self, log_dir, logfile, fileLogLevel, consoleLogLevel):
        self._logger = logging.getLogger(name=None)

        # file logging
        file = path.join(log_dir, logfile)
        logging.root.setLevel(logging.DEBUG)
        file_logger = logging.FileHandler(file, 'w')
        file_formatter = logging.Formatter('%(asctime)s %(name)-15s %(levelname)-10s %(message)s',
                                           datefmt='%y-%d-%m %H:%M:%S')
        file_logger.setFormatter(file_formatter)
        file_logger.setLevel(fileLogLevel)
        self._logger.addHandler(file_logger)
        # console logging
        console_logger = logging.StreamHandler()
        console_formatter = logging.Formatter('%(name)-15s %(levelname)-10s %(message)s')
        console_logger.setFormatter(console_formatter)
        console_logger.setLevel(consoleLogLevel)
        self._logger.addHandler(console_logger)

        print("Starting the simulation. Please, consult the logfile at "+logfile)
