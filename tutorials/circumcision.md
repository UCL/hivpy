## Circumcision Tutorial

Welcome to the circumcision module. This code deals with circumcision at birth and voluntary medical male circumcision (VMMC) intervention. The most relevant files are listed below:

- `src/hivpy/circumcision.py` - The circumcision module.
- `src/tests/test_circumcision.py` - Tests for the circumcision module.
- `src/hivpy/data/circumcision.yaml` - Circumcision data and variables.
- `src/hivpy/circumcision_data.py` - A class for storing data loaded from `circumcision.yaml`.

If there are any circumcision-related variables you would like to change before running your simulation, please change them in `circumcision.yaml`.

### Circumcision Data Variables

- *`vmmc_start_year`* - The year during which VMMC intervention begins (2008 by default).
- *`circ_rate_change_year`* - The year after which VMMC rates change and *`circ_rate_change_post_2013`* becomes included in VMMC probability calculations (2013 by default).
- *`prob_circ_calc_cutoff_year`* - The year which caps the current time step date used in VMMC probability calculations (2019 by default).
- *`circ_after_test`* - A boolean that determines whether a negative HIV test can lead to VMMC.
- *`prob_circ_after_test`* - The probability that a negative HIV test leads to VMMC.
- *`covid_disrup_affected`* - A boolean that determines whether disruption due to COVID is factored into the model.
- *`vmmc_disrup_covid`* - A boolean that determines whether COVID disruption affects VMMC intervention.
- *`policy_intervention_year`* - The year after which policy intervention options are modelled (2022 by default).
- *`circ_policy_scenario`* - An integer that represents the simulation of the enactment of a specific policy intervantion option after *`policy_intervention_year`*.
    - **Scenario 0** - Default behaviour.
    - **Scenario 1** - VMMC stops in 10-14 year olds and increases in 15-19 year olds.
    - **Scenario 2** - No further VMMC is carried out.
    - **Scenario 3** - VMMC stops in 10-14 year olds.
    - **Scenario 4** - VMMC stops in 10-14 year olds and and there is no further VMMC 5 years after *`policy_intervention_year`*.
- *`circ_increase_rate`* - The rate at which VMMC increases over time.
- *`circ_rate_change_post_2013`* - The relative increase in VMCC after *`circ_rate_change_year`*.
- *`circ_rate_change_15_19`* - The relative increase in VMMC in 15-19 year olds.
- *`circ_rate_change_20_29`* - The relative decrease in VMMC in 20-29 year olds.
- *`circ_rate_change_30_49`* - The relative decrease in VMMC in 30-49 year olds.
- *`prob_birth_circ`* - The probability of circumcision at birth.

### Population Initialisation

Birth circumcision is initialised when population data is first created. There are currently two methods for initialising birth circumcision:

- `init_birth_circumcision_all` - Initialises circumcision at birth for the entire male population, both born and unborn. Circumcision dates for all born individuals are recorded as the date of the start of the simulation, but cirucmcision dates for unborn individuals are calculated by finding the dates at which each individual's age would be 0.25.
- `init_birth_circumcision_born` - Initialises circumcision at birth for all born males of *age >= 0.25*. All circumcised individuals get assigned a circumcision date of the start of the simulation. This method requires the use of `update_birth_circumcision` during a population's evolve step, which updates birth circumcision for newly born males of *age >= 0.25* and *age - `time_step` < 0.25* during each time step (assuming ages have already been incremented this time step). During the update, newly circumcised males have the date of the current time step set as their circumcision date.

The method that initialises birth circumcision all at once is much faster at determining birth circumcision outcomes for the entire population than the second due to the high cumulative time taken by the update method, however it does not factor in COVID disruption. On the other hand, the second method's updates can take new information into account during a simulation, and is thus capable of factoring in COVID disruption. As such, a birth circumcision update only goes ahead if *`covid_disrup_affected`* and *`vmmc_disrup_covid`* are both False, otherwise the circumcision probability is 0.

#### Changing Birth Circumcision Initialisation

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
- *`circ_policy_scenario` == 2* AND *`policy_intervention_year` <= year*
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

#### Changing VMMC Age Range and Age Groupings

If you would like to change the age range at which VMMC can be applied (*10 =< age < 50* by default), you can do so by editing *`min_vmmc_age`* and *`max_vmmc_age`* in `__init__` in `circumcision.py`. Note that the module is set up such that those of *`min_vmmc_age`* are included as valid candidates for VMMC, while those of *`max_vmmc_age`* are excluded. Additionally, you can edit *`vmmc_cutoff_age`* (*`min_vmmc_age` + 5* i.e. 15 by default) to change the age at which *`circ_policy_scenario`* 1, 3, and 4 stop circumcision in minors.

If you would like to change the age groups ([10-19, 20-29, 30-49] by default) used for calculating different VMMC probabilities, you can edit *`vmmc_age_bound_1`* (20 by default) and *`vmmc_age_bound_2`* (30 by default) in `__init__` to change the boundary ages at which age groups are divided. If you would like to change the *number* of age groups, introduce any new *`vmmc_age_bound_x`* variables as necessary and edit *`age_groups`* in `update_vmmc` to achieve the desired number of groups by adding or removing age bounds. Additionally, `calc_prob_circ` may need to be updated to change the VMMC probability age modifier for any newly added age groups.

#### Adding a New VMMC Scenario or Variable

If you would like to add a new *`circ_policy_scenario`*, there is currently no simple way to do so. You will need to edit `update_vmmc` to exhibit the new branching logic you expect, as well as potentially edit `calc_prob_circ` if your new scenario affects VMMC probability calculations.

If you would like to add a new circumcision data variable, first add it to `circumcision.yaml`. Then add it to `__init__` in `circumcision_data.py`:
```python
self.your_var_name_here = self.data["your_var_name_here"]
```
If your variable is more than just a single value, use an appropriate data reader method instead:
```python
# e.g. reading in a discrete distribution
self.your_var_name_here = self._get_discrete_dist("your_var_name_here")
```
Now add your variable to `__init__` in `circumcision.py`:
```python
self.your_var_name_here = self.c_data.your_var_name_here
```
If your variable needs to be picked from a distribution, use the sample method:
```python
self.your_var_name_here = self.c_data.your_var_name_here.sample()
```
