from __future__ import annotations

import logging
import os
from copy import deepcopy
from datetime import datetime

import pandas as pd

from . import output
from .config import SimulationConfig
from .output import SimulationOutput
from .population import Population
from .post_processing import graph_output


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
            "simulation_output_intervention_i" + str(datetime.now().strftime("%Y%m%d-%H%M%S")) + ".csv")

    def _initialise_population(self):
        self.population = Population(self.simulation_config.population_size,
                                     self.simulation_config.start_date)

    def intervention(self, pop, option):
        # Negative intervention options have been reserved for demonstration / testing
        if option == -1:
            pop.sexual_behaviour.sw_program_start_date = pop.date - self.simulation_config.time_step
        elif option == -2 and pop.date == datetime(2002, 1, 1):
            pop.circumcision.policy_intervention_year = pop.date
        return pop

    def run(self):

        # Start the simulation
        date = self.simulation_config.start_date
        assert date == self.population.date
        time_step = self.simulation_config.time_step

        if self.simulation_config.intervention_date:
            end_date = self.simulation_config.intervention_date
            interv_option = self.simulation_config.intervention_option
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
            self.modified_population = self.intervention(self.modified_population, interv_option)

            while date <= self.simulation_config.stop_date:
                logging.info("Timestep %s\n", date)

                # intervention
                self.modified_population = self.modified_population.evolve(time_step)
                self.intervention_output.update_summary_stats(date, self.modified_population, time_step)
                # repeat intervention according to option number
                if self.simulation_config.recurrent_intervention:
                    self.modified_population = self.intervention(self.modified_population, interv_option)

                # no intervention
                self.population = self.population.evolve(time_step)
                self.output.update_summary_stats(date, self.population, time_step)

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
