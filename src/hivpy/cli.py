import argparse
import pathlib

from .config import ExperimentConfig
from .exceptions import ConfigException
from .experiment import run_experiment

'''
Maybe we may want to register parameters/properties in future?
'''
# def register_parameters():
#     parser = argparse.ArgumentParser(description="register model parameters")
#     parser.add_argument("parameters", type=pathlib.Path,
#                         help="register_parameters parameters.csv")
#     args = parser.parse_args()
#     parameter_filepath = args.parameters


def run_model():
    '''
    Running a HIV model simulation
    Assuming there is a hivpy.conf file the command is
    $run_model hivpy.conf
    '''
    parser = argparse.ArgumentParser(description="run a simulation")
    parser.add_argument("input", type=pathlib.Path, help="run_model config.conf")
    args = parser.parse_args()
    conf_filename = args.input
    try:
        experiment_config = ExperimentConfig.from_file(conf_filename)
        run_experiment(experiment_config)
    except ConfigException as err:
        print(err.args[0])


if __name__ == '__main__':
    print('calling main')
    run_model()
