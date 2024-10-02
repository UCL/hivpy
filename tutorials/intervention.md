## Intervention Tutorial

Intervention is an optional part of the simulation which is triggered at the _year of intervention_; this allows the simulation to fork at this point into two independent simulations which differ in some properties. The changes to the properties of the simulation are determined by the _intervention function_, what can modify properties of the population or any of the simulation modules. After the year on intervention, one simulation will continue with the intervention function applied, and the other will continue with no changes. The intervention function may be applied once at the year of intervention, or every timestep following that point. 

The most relevant files are listed below:

- `src/hivpy/simulation.py` - The simulation module, which contains the intervention code
- `src/tests/test_simulation.py` - Tests for the simulation
- `src/hivpy/hivpy_intervention.yaml` - Sample file for running with intervention-related variables
- `src/tests/test_population.py` - Test for understanding how the population class is modified

If there are any testing-related variables you would like to change before running your simulation, please change them in `hivpy_intervention.yaml` and `simulation.py`.

To make use of the intervention-related option there are two steps that need to be followed:

1) to add/modify the relevent intervention-related variables (intervention year, intervention option etc.) at the configuration file that is used to launch the simulation,
2) to ensure that the relevent intervention option is implemented in the intervention function in `simulation.py`.

These steps are described in more detail below: 

### Modifying the configuration file

The configuration file contains the intervention-related parameters that should be initialised when launching the simulation. These parameters are:
- `intervention_year`: The year were the intervention is set to take place (date)
- `intervention_option`: The option to implement at the intervention year (integer)
- `repeat_intervention`: Whether the intervention will repeat every time step after the year of intervention (True/False). If false, the intervention function will only be called once at the start of the intervention year.

For an example of how these parameters can be used please refer to the sample file `hivpy_intervention.yaml`

To run the simulations using the sample file, the `run_model` command can be run in the terminal:
```bash
run_model hivpy_intervention.yaml
```

### Modifying the simulation module

If an intervention year is set in the configuration file the `intervention` function is called in the simulation module. A synopsis of the process is: if the intervention year exists the simulation runs until the intervention year; a deep copy of the population object is created (this includes all of the modules that it owns such as sexual behaviour, HIV status, etc.); the simulation advances in two seperate outcomes with and without the intervention being implemented. 

The 'intervention' function takes as input the 'intervention_option' provided as a numeric value in the configuration file and updates the population / modules accordingly. 

#### Adding an intervention option in the simulation

To add a new intervention option into the simulation, the 'intervention' function in the simulation module needs to be modified. To do so a condition with the number of the option should be added and the corresponding population sub-module should be accessed. 

**Options with negative numbers (-1, -2) have been reserved for the purposes of testing and providing example code in the intervention function.**

For example, option no. `-1` changes the starting date of the sex worker program setting as starting date the intervention date: 
```bash
if option == -1:
            pop.sexual_behaviour.sw_program_start_date = pop.date - self.simulation_config.time_step
    
```
- `if` statements can be more complex if desired, e.g. `if option in [1, 2, 3, 4]:` can be used to define code which applies to all the options in that list.
- Option code can modify any of the modules or the population itself through the population object (`pop`); in this case we are changing the `sw_program_start_date` by accessing the sexual behaviour module (`sexual_behaviour`).
- Alternative options can be added in a similar way. 

#### Recurrent intervention 
If the `repeat_intervention` is set to True in the configuration file, the intervention is set to repeat for every timestep after the intervention year. This can be useful if there is reason to believe that changes made in the option code may be overridden by other parts of the simulation. 