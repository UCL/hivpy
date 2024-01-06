# analysis.py
import yaml
from hivpy.experiment import create_experiment, run_experiment

def running_model(config_filename):
    with open(config_filename, 'r') as file:
        config = yaml.safe_load(file)
        experiment_config = create_experiment(config)
        run_experiment(experiment_config)

config_path = r"C:\Users\w3sth\PycharmProjects\hivpy\hivpy.yaml"
running_model(config_path)


