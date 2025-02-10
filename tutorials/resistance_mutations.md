## Resistance Mutations Tutorial

The resistance mutations module tracks viral load, CD4 count, and HIV resistance mutations in HIV+ people. The most relevant files are listed below:

- `src/hivpy/resistance_mutations.py` - The resistance mutations module.
- `src/tests/test_resistance_mutations.py` - Tests for the resistance mutations module.

### Module Overview

When resistance is updated, the HIV+ sub-population is first assigned various indices based on their number of active drugs, continuous ART usage, and ART adherence. These indices are used to look up values in matrices containing viral load, CD4 delta, and new mutation information for the purposes of calculating an individual's current viral load, CD4 count, and probability of a new HIV resistance mutation. Each HIV+ person is also assigned an overall `resistance_index` which is used to access their `active_drug_index`, `cont_on_art_index`, `adherence_index`, and `adherence_tm1_index`.

A resistance matrix is accessed with `matrix`[`active_drug_index`][`cont_on_art_index`][`adherence_index`]. Adherence last time step is discounted unless a person has been on ART for 3 <= `cont_on_art` < 6 months, in which case a matrix is instead accessed with `matrix`[`active_drug_index`][`cont_on_art_index`][`adherence_index`][`adherence_tm1_index`].

Viral load is calculated first and is affected by an individual's viral load last time step. CD4 count is calculated next and is affected by an individual's age, sex, use of specific ART drugs, as well as CD4 levels last time step, maximum CD4 levels, and individual rate of CD4 recovery on ART. Finally, new resistance mutations are determined for people that clear their probability of acquiring a new HIV mutation, the calculation of which is affected by an individual's use of specific ART drugs and viral load.
