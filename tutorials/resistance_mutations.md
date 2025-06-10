## Resistance Mutations Tutorial

The resistance mutations module tracks viral load, CD4 count, and HIV resistance mutations in HIV+ people. The most relevant files are listed below:

- `src/hivpy/resistance_mutations.py` - The resistance mutations module.
- `src/tests/test_resistance_mutations.py` - Tests for the resistance mutations module.
- `src/hivpy/data/resistance_mutations.yaml` - Resistance mutations data and variables.
- `src/hivpy/resistance_mutations_data.py` - A class for storing data loaded from `resistance_mutations.yaml`.

If there are any mutation-related variables you would like to change before running your simulation, please change them in `resistance_mutations.yaml`.

### Module Overview

When resistance is updated, the HIV+ sub-population is first assigned various indices based on their number of active drugs, continuous ART usage, and ART adherence. These indices are used to look up values in matrices containing viral load, CD4 delta, and new mutation information for the purposes of calculating an individual's current viral load, CD4 count, and probability of a new HIV resistance mutation. Each HIV+ person is also assigned an overall `resistance_index` which is used to access their `active_drug_index`, `cont_on_art_tm1_index`, `adherence_index`, and `adherence_tm1_index`.

The `viral_load_matrix` contains (`a`, `b`, `c`) tuples used to calculate a base viral load value with the expression `a * max_viral_load + b + c * min_vl_on_art`, which is then used to calculate viral load changes this time step. The `cd4_delta_matrix` and `new_mutation_matrix` both contain multiplier values used in the calculation of the change in CD4 levels and new mutation probability for this time step respectively.

A resistance matrix is accessed with `matrix`[`active_drug_index`][`cont_on_art_tm1_index`][`adherence_index`]. Adherence last time step is discounted unless a person has been on ART for 3 <= `cont_on_art_tm1` < 6 months, in which case a matrix is instead accessed with `matrix`[`active_drug_index`][`cont_on_art_tm1_index`][`adherence_index`][`adherence_tm1_index`].

Viral load is calculated first and is affected by an individual's viral load last time step. CD4 count is calculated next and is affected by an individual's age, sex, use of specific ART drugs, as well as CD4 levels last time step, maximum CD4 levels, and individual rate of CD4 recovery on ART. Finally, new resistance mutations are determined for people that clear their probability of acquiring a new HIV mutation, the calculation of which is affected by an individual's use of specific ART drugs and viral load.
