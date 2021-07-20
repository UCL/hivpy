from dataclasses import dataclass
from datetime import date, timedelta
from .population import Population
from .simulation import run_simulation, SimulationConfig
import logging
import os

LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


@dataclass
class ExperimentConfig:
    """Configuration parameters covering all stages of an experiment."""
    population_size: int
    simulation: SimulationConfig

    def initialize_population(self):
        """Create an initial population for use in simulations."""
        return Population(self.population_size, self.simulation.start_date)

@dataclass
class ExperimentOutput:
    output_dir: str
    log_level : str
    log_file : str

def create_experiment_from_config(experiment_param):
    start_date = date(1980,1,1)
    end_date = date(2040, 12, 31)
    interval = timedelta(days=30)
    population = 1000
    try:
        start_date = date(int(experiment_param['START_YEAR']),1 ,1 )
        end_date = date(int(experiment_param['END_YEAR']),12, 31)
        population = int(experiment_param['POPULATION'])
        interval = timedelta(days = int(experiment_param['TIME_INTERVAL_DAYS']))
    except RuntimeError as err:
        print('Error parsing the experiment parameters {}'.format(err))

    # Dummy values for now
    sim_config = SimulationConfig(start_date, end_date, interval)
    return ExperimentConfig(population, sim_config)

def create_output(general_param):
    outputdir = general_param['OUTPUT_DIRECTORY']
    logfilename = general_param['LOGOUTPUT_NAME']
    log_level = general_param['LOG_LEVEL']
    pathname = os.path.join(outputdir, logfilename)


    logging.basicConfig(filename=pathname, level=LEVELS[log_level])
    logging.info("starting experiment")

    return ExperimentOutput(outputdir, log_level, pathname)


def run_experiment(experiment_config):
    """Run an entire experiment.

    An experiment can consist of one or more simulation runs,
    as well as processing steps after those are completed.
    """
    simulation_config = experiment_config.simulation
    population = experiment_config.initialize_population()
    result = run_simulation(population, simulation_config)
