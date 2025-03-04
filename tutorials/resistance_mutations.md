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

Viral load is calculated first and is affected by an individual's viral load last time step. CD4 count is calculated next and is affected by an individual's age, sex, use of specific ART drugs, as well as CD4 levels last time step, maximum CD4 levels, and individual rate of CD4 recovery on ART. Finally, new resistance mutations may be acquired by people who clear their probability of developing a new HIV mutation, the calculation of which is affected by an individual's use of specific ART drugs and viral load. Developing resistance is not guaranteed, but any number of new mutations may arise in an individual based on their current drug regimen and the subsequent resistance probabilities associated with each mutation.

All mutations are tracked with a `MutationStatus` enum, except thymidine analog mutations (TAMs), which are tracked as an integer instead. A person can have up to 6 TAMs, but all other mutations will either be `Absent`, in `Minority`, or in `Majority`. New mutations acquired through this module are automatically added as in `Majority`. Mutations in `Majority` may change to in `Minority` over time, but once a mutation is present in an individual's system it can never become `Absent` again. Only mutations in `Majority` can be transmitted to other people.

### Resistance Mutations Columns

- *`RESISTANCE_INDEX`* - An integer value that serves as an index to identify an individual's resistance profile for various calculations.
- *`RESISTANCE_MUTATIONS`* - An integer value between [0, 28] that tracks the total number of resistance mutations in an individual.
- *`RTTA_MUTATIONS`* - An integer value between [0, 6] that tracks the number of reverse transcriptase gene thymidine analog mutations (TAMs) in an individual.
- *`RT184_MUTATION`* - A MutationStatus enum storing the current presence or absence of the reverse transcriptase gene M184 mutation in an individual.
- *`RT151_MUTATION`* - A MutationStatus enum storing the current presence or absence of the reverse transcriptase gene Q151 mutation in an individual.
- *`RT65_MUTATION`* - A MutationStatus enum storing the current presence or absence of the reverse transcriptase gene K65 mutation in an individual.
- *`RT103_MUTATION`* - A MutationStatus enum storing the current presence or absence of the reverse transcriptase gene K103 mutation in an individual.
- *`RT181_MUTATION`* - A MutationStatus enum storing the current presence or absence of the reverse transcriptase gene Y181 mutation in an individual.
- *`RT190_MUTATION`* - A MutationStatus enum storing the current presence or absence of the reverse transcriptase gene G190 mutation in an individual.
- *`PR32_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P32 mutation in an individual.
- *`PR46_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P46 mutation in an individual.
- *`PR47_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P47 mutation in an individual.
- *`PR50L_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P50L mutation in an individual.
- *`PR50V_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P50V mutation in an individual.
- *`PR54_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P54 mutation in an individual.
- *`PR76_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P76 mutation in an individual.
- *`PR82_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P82 mutation in an individual.
- *`PR84_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P84 mutation in an individual.
- *`PR88_MUTATION`* - A MutationStatus enum storing the current presence or absence of the protease gene P88 mutation in an individual.
- *`IN118_MUTATION`* - A MutationStatus enum storing the current presence or absence of the integrase inhibitor IN118 mutation in an individual.
- *`IN140_MUTATION`* - A MutationStatus enum storing the current presence or absence of the integrase inhibitor IN140 mutation in an individual.
- *`IN148_MUTATION`* - A MutationStatus enum storing the current presence or absence of the integrase inhibitor IN148 mutation in an individual.
- *`IN155_MUTATION`* - A MutationStatus enum storing the current presence or absence of the integrase inhibitor IN155 mutation in an individual.
- *`IN263_MUTATION`* - A MutationStatus enum storing the current presence or absence of the integrase inhibitor IN263 mutation in an individual.
- *`CA66_MUTATION`* - A MutationStatus enum storing the current presence or absence of the capsid gene CA66 mutation in an individual.

### Resistance Mutations Data Variables

- *`active_drug_bins`* - Indexing boundaries of the number of active drugs in a regimen for resistance matrices.
- *`cont_on_art_bins`* - Indexing boundaries of continuous ART usage for resistance matrices.
- *`adherence_bins`* - Indexing boundaries of ART adherence for resistance matrices.
- *`min_vl_on_art`* - The minimum viral load on ART.
- *`vl_stdev_on_art`* - The standard deviation for viral load changes on ART.
- *`hindered_cd4_recovery`* - A lowered base CD4 recovery starting value that is used when a failing NNRTI regimen hinders CD4 recovery.
- *`failed_insti_hinders_cd4_recovery`* - A boolean flag that determines whether a failing INSTI regimen also hinders CD4 recovery.
- *`cd4_recovery_pi_factor`* - A factor that increases CD4 recovery for people taking protease inhibitors.
- *`cd4_recovery_female_factor`* - A factor that increases CD4 recovery for women.
- *`cd4_stdev_on_art`* - The standard deviation for CD4 changes on ART.
- *`mutation_risk_change`* - A constant factor affecting the population's new mutation probabilities.
- *`risk_change_tams_resist`* - A factor affecting the probability of developing new TAMs.
- *`risk_change_151_resist`* - A factor affecting the probability of developing the RT151 mutation.
- *`risk_change_cab_resist`* - A factor affecting the probability of developing the IN118, IN140, IN148, IN155, and IN263 mutations.
- *`resist_rate_tams_higher`* - The higher rate of TAMs acquisition.
- *`resist_rate_tams_lower`* - The lower rate of TAMs acquisition.
- *`resist_rate_nev_higher`* - The higher rate of nevirapine resistance acquisition.
- *`resist_rate_nev_lower`* - The lower rate of nevirapine resistance acquisition.
- *`resist_rate_efa_higher`* - The higher rate of efavirenz resistance acquisition.
- *`resist_rate_efa_lower`* - The lower rate of efavirenz resistance acquisition.
- *`resist_rate_lpr_higher`* - The higher rate of lopinavir resistance acquisition.
- *`resist_rate_lpr_lower`* - The lower rate of lopinavir resistance acquisition.
- *`zdv_resist_rate`* - The rate of zidovudine resistance acquisition.
- *`3tc_resist_rate`* - The rate of lamivudine resistance acquisition.
- *`dar_resist_rate`* - The rate of darunavir resistance acquisition.
- *`taz_resist_rate`* - The rate of atazanavir resistance acquisition.
- *`isl_resist_rate`* - The rate of islatravir resistance acquisition.
- *`ten_resist_rate`* - The rate of tenofovir resistance acquisition.
- *`dol_resist_rate`* - The rate of dolutegravir resistance acquisition.
- *`len_resist_rate`* - The rate of lenacapavir resistance acquisition.
- *`incr_len_resist`* - A multiplier for lenacapavir resistance acquisition.
- *`cab_resist_factor`* - A multiplier for cabotegravir resistance acquisition.
- *`viral_load_matrix`* - A matrix containing (`a`, `b`, `c`) float tuples used to calculate an individual's base viral load (`a * max_viral_load + b + c * min_vl_on_art`) for a given time step.
- *`cd4_delta_matrix`* - A matrix containing floats used to calculate an individual's change in CD4 levels for a given time step.
- *`new_mutation_matrix`* - A matrix containing floats used to calculate an individual's new mutation probability for a given time step.
