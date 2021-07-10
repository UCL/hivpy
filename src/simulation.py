def run_simulation(population, config):
    """Run a single simulation for the given population and time bounds."""
    date = config.start_date
    time_step = config.time_step
    while date < config.stop_date:
        population = population.evolve(time_step)
        date = date + time_step


