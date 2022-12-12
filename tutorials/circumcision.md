## Circumcision Tutorial

Welcome to the circumcision module. This code deals with circumcision at birth and voluntary medical male circumcision (VMMC) intervention. The most relevant files are listed below:

- `src/hivpy/circumcision.py` - The circumcision module.
- `src/tests/test_circumcision.py` - Tests for the circumcision module.
- `src/hivpy/data/circumcision.yaml` - Circumcision data and variables.
- `src/hivpy/circumcision_data.py` - A class for storing data loaded from `circumcision.yaml`.

If there are any circumcision-related variables you would like to change before running your simulation, please change them in `circumcision.yaml`.

### Population Initialisation

Birth circumcision is initialised when population data is first created. There are currently two methods for initialising birth circumcision:

- `init_birth_circumcision_all` - Initialises circumcision at birth for the entire male population, both born and unborn. Circumcision dates for all born individuals are recorded as the date of the start of the simulation, but cirucmcision dates for unborn individuals are calculated by finding the dates at which each individual's age would be 0.25.
- `init_birth_circumcision_born` - Initialises circumcision at birth for all born males of *age >= 0.25*. All circumcised individuals get assigned a circumcision date of the start of the simulation. This method requires the use of `update_birth_circumcision` during a population's evolve step, which updates birth circumcision for newly born males of *age >= 0.25* and *age - `time_step` < 0.25* during each time step (assuming ages have already been incremented this time step). During the update, newly circumcised males have the date of the current time step set as their circumcision date.

The method that initialises birth circumcision all at once is much faster at determining birth circumcision outcomes for the entire population than the second due to the high cumulative time taken by the update method, however it does not factor in COVID disruption. On the other hand, the second method's updates can take new information into account during a simulation, and is thus capable of factoring in COVID disruption. As such, a birth circumcision update only goes ahead if *`covid_disrup_affected`* and *`vmmc_disrup_covid`* are both False, otherwise the circumcision probability is 0.

If you would like to change how birth circumcision is initialised, you need to edit `_create_population_data` in `src/hivpy/population.py`.

To initialise circumcision for all males at once, make sure that the following line is present:
```python
self.circumcision.init_birth_circumcision_all(self.data, self.date)
```
Alternatively, to initialise circumcision only for born males, make sure the line above is replaced with the following:
```python
self.circumcision.init_birth_circumcision_born(self.data, self.date)
```

If using the second initialisation method, you also need to edit `evolve` in `src/hivpy/population.py` such that it features the following line *after* ages have been incremented for the time step:
```python
self.circumcision.update_birth_circumcision(self.data, time_step, self.date)
```

### Population Evolution

VMMC is updated when the population is advanced by one time step during its `evolve`.

There is no VMMC during a given time step if one of the following conditions is fulfilled:
- *`vmmc_disrup_covid` == True*
- *`circ_policy_scenario == 2`* AND *`policy_intervention_year` <= year*
- *`circ_policy_scenario` == 4* AND *`policy_intervention_year` + 5 <= year*

Otherwise, VMMC occurs only if the current *year > `vmmc_start_year`*. Typically uncircumcised males of *10 =< age < 50* are eligible for VMMC, but if *`circ_policy_scenario`* is one of 1, 3, or 4 and *`policy_intervention_year` <= year* then circumcision stops in 10-14 year olds.

Individuals are chosen for VMMC by first finding all eligible males that have not yet been circumcised. If uncircumcised males are present during a given time step, then males are grouped by age [10-19 (or 15-19), 20-29, 30-49] and VMMC outcomes are assigned according to the different VMMC probabilities for each age group.

VMMC probabilities are calculated in the following way:
- An age modifier is determined based on age group and the year used in calculations is capped at *`prob_circ_calc_cutoff_year`* (2019 by default) if the current year exceeds this date.
- If the current date is *`circ_rate_change_year`* (2013 by default) or earlier, the following calculation is used:
```python
prob_circ = (year - vmmc_start_year) * circ_increase_rate * age_mod
```
- If the current date is after *`circ_rate_change_year`* and the 15-19 age group has an age modifier, the probability is calculated with:
```python
prob_circ = ((circ_rate_change_year - vmmc_start_year) + (year - circ_rate_change_year) * circ_rate_change_post_2013 * circ_inc_15_19) * circ_increase_rate
```
- Otherwise, if the current date is after *`circ_rate_change_year`* and the 10-19 (or 15-19) age group does not have an age modifier (other than the default 1), the calculation is:
```python
prob_circ = ((circ_rate_change_year - vmmc_start_year) + (year - circ_rate_change_year) * circ_rate_change_post_2013) * circ_increase_rate * age_mod
```
