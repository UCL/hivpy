from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd

from . import output
from .config import SimulationConfig
from .output import SimulationOutput
from .population import Population
from .post_processing import graph_output
import hivpy.column_names as col
from copy import deepcopy


class SimulationHandler:
    """
    A class for handling executing a simulation and outputting results.
    """
    simulation_config: SimulationConfig
    output: SimulationOutput
    population: Population

    def __init__(self, simulation_config):
        self.simulation_config = simulation_config
        output.simulation_output = SimulationOutput(simulation_config.start_date,
                                                    simulation_config.stop_date,
                                                    simulation_config.time_step)
        self.output = output.simulation_output
        self._initialise_population()
        self.output_dir = simulation_config.output_dir / (
            "simulation_output_" + str(datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.output_path = self.output_dir / (
            "simulation_output_" + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + ".csv")
        self.output_path_intervention = self.output_dir / (
            "simulation_output_intervention" + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + ".csv")

    def _initialise_population(self):
        self.population = Population(self.simulation_config.population_size,
                                     self.simulation_config.start_date)

    def intervention(self, pop):
        pop.set_present_variable(col.TEST_MARK, True)
        return pop

    def run(self):

        # Start the simulation
        date = self.simulation_config.start_date
        assert date == self.population.date
        time_step = self.simulation_config.time_step

        if self.simulation_config.intervention_date:
            end_date = self.simulation_config.intervention_date
            message = "Reached intervention year"
        else:
            end_date = self.simulation_config.stop_date
            message = "Finished"

        while date <= end_date:
            logging.info("Timestep %s\n", date)
            # Advance the population
            self.population = self.population.evolve(time_step)
            self.output.update_summary_stats(date, self.population, time_step)
            date = date + time_step
        logging.info(message)

        if self.simulation_config.intervention_date:
            # make deep copy
            self.modified_population = deepcopy(self.population)
            self.intervention_output = deepcopy(self.output)
            # call intervention function
            self.modified_population = self.intervention(self.modified_population)

            while date <= self.simulation_config.stop_date:
                logging.info("Timestep %s\n", date)

                # option 1 intervention
                self.modified_population = self.modified_population.evolve(time_step)
                self.intervention_output.update_summary_stats(date, self.modified_population, time_step)

                # option 2 no intervention
                self.population = self.population.evolve(time_step)
                self.output.update_summary_stats(date, self.population, time_step)

                # continue till end
                date = date + time_step
            logging.info("Finished")

        # Store results
        if not os.path.exists(self.output_dir):
            os.makedirs(os.path.join(self.output_dir, "graph_outputs"))
        self.output.write_output(self.output_path)
        self.output.output_stats["Date"] = pd.to_datetime(
            self.output.output_stats["Date"], format="(%Y, %m, %d)")
        graph_output(os.path.join(self.output_dir, "graph_outputs"), self.output.output_stats,
                     self.simulation_config.graph_outputs)

        if self.simulation_config.intervention_date:
            self.intervention_output.write_output(self.output_path_intervention)
            self.intervention_output.output_stats["Date"] = pd.to_datetime(
                self.intervention_output.output_stats["Date"], format="(%Y, %m, %d)")
            graph_output(os.path.join(self.output_dir, "graph_outputs"), self.intervention_output.output_stats,
                         self.simulation_config.graph_outputs)
