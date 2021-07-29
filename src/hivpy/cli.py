import configparser
import argparse
import pathlib
from .experiment import create_experiment, create_output, run_experiment
from .config import SimulationConfig, OutputConfig, ExperimentConfig

'''
Maybe we may want to register parameters/properties in future?
'''
def register_parameters():
    parser = argparse.ArgumentParser(description="register model parameters")
    parser.add_argument("parameters", type=pathlib.Path, help="register_parameters parameters.csv")
    args = parser.parse_args()
    parameter_filepath = args.parameters

'''
Running a HIV model simulation
Assuming there is a hivpy.conf file the command is
$run_model hivpy.conf
'''
def run_model():
    parser = argparse.ArgumentParser(description="run a simulation")
    parser.add_argument("input", type=pathlib.Path, help="run_model config.conf")
    args = parser.parse_args()
    conf_filename = args.input
    config = configparser.ConfigParser()
    try:
        config.read(conf_filename)
        simulation_config = create_experiment(config['EXPERIMENT'])
        output_config = create_output(config['OUTPUT'])
        experiment_config = ExperimentConfig(simulation_config=simulation_config, output_config=output_config)
        run_experiment(experiment_config)

    except configparser.Error as err:
        print('error parsing the config file {}'.format(err))


if __name__ == '__main__':
    print('calling main')
    run_model()