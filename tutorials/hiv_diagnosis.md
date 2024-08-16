## HIV Diagnosis Tutorial

The HIV diagnosis module tracks the current HIV test type, test sensitivities, as well as loss of care at diagnosis. The most relevant files are listed below:

- `src/hivpy/hiv_diagnosis.py` - The HIV diagnosis module.
- `src/tests/test_hiv_diagnosis.py` - Tests for the HIV diagnosis module.
- `src/hivpy/data/hiv_diagnosis.yaml` - HIV diagnosis data and variables.
- `src/hivpy/hiv_diagnosis_data.py` - A class for storing data loaded from `hiv_diagnosis.yaml`.

If there are any testing-related variables you would like to change before running your simulation, please change them in `hiv_diagnosis.yaml`.

### Module Overview

When HIV diagnosis is updated, people that were tested this time step have a chance to be diagnosed. Effective HIV test sensitivity (and thus the probability of diagnosis) is different for those that are in primary infection and those that are not. Test sensitivity for people in primary infection is based on test type, use of injectable PrEP (Cabotegravir or Lenacapavir), and whether or not PrEP usage began before this time step. Test sensitivity for the general population is affected by test type, use of injectable PrEP, and the duration of infection.

Some people have a chance to disengage with medical care after diagnosis, which is represented by determining loss of care outcomes. Loss of care is affected by sex worker status in both primary infection and general populations, but it is also affected by the presence of short-term partners and HIV-adjacent diseases (ADC, TB, non-TB WHO3) in the general population only.
