## Intervention Tutorial

The intervention function forms part of the simulation and allows to assess how the outputs may differ in case of changes in policy during an intervention year. The most relevant files are listed below:

- `src/hivpy/simulation.py` - The simulation module
- `src/tests/test_simulation.py` - Tests for the simulation
- `src/hivpy/hivpy_intervention.yaml` - Sample file for running with intervention-related variables
- `src/tests/test_population.py` - Test for understanding how the population class is modified

If there are any testing-related variables you would like to change before running your simulation, please change them in `hivpy_intervention.yaml` and `simulation.py`.

To make use of the intervention-related option there are two steps that need to be followed:

1) To add the intervention-related variables at the configuration file that is used to launch the simulation,
2) To include the desirable options inside the intervention function of the simulation module.

These steps are described in more detail below: 

### Modifying the configuration file

The configuration file contains the intervention-related parameters that should be initialised when launching the simulation. These parameters are:
- `intervention_year`: The year were the intervention is set to take place (date)
- `intervention_option`: The option to implement at the intervention year (integer)
- `repeat_intervention`: Whether the intervention will repeat before the simulation ends (True/False)

For an example of how these parameters can be used please refer to the sample file `hivpy_intervention.yaml`

To run the simulations using the sample file, the `run_model` command can be run in the terminal:
```bash
run_model hivpy_intervention.yaml
```

### Modifying the simulation module

Once an intervention year is set in the configuration file the `intervention` function is called in the simulation module. A synopsis of the process is: if the intervention year exists the simulation runs until the intervention year; a deep copy of the population class is created; the simulation advances in two seperate outcomes with and without the intervention being implemented. 

The 'intervention' function takes as input the 'intervention_option' provided as a numeric value in the configuration file and updates the population class accordingly. 

#### Adding an intervention option in the simulation
To add a new intervention option into the simulation, the 'intervention' function in the simulation module needs to be modified. To do so a condition with the number of the option should be added and the corresponding population sub-module should be accessed. 

For example, option no. 1 changes the starting date of the sex worker program setting as starting date the intervention date: 
```bash
if option == 1:
            pop.sexual_behaviour.sw_program_start_date = pop.date - self.simulation_config.time_step
    
```
Alternative options can be added in a similar way. 

#### Recurrent intervention 
If the `repeat_intervention` is set to True in the configuration file, the intervention is set to repeat for every timestep after the intervention year. This option is currently hard-coded in the simulation module. The `update_intervention` function takes as input the population that needs to be modified and the intervention option value to be implemented and calls the `intervention` function per timestep. 
