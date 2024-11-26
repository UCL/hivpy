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

### HIV Diagnosis Data Variables

- *`hiv_test_type`* - The type of HIV test being administered; one of: Ab (default), NA, Ag/Ab.
- *`init_prep_inj_na`* - A boolean randomly chosen by a coin flip at the start of the simulation to determine whether `prep_inj_na` has a chance to be True.
- *`prep_inj_na`* - A boolean randomly chosen by a coin flip at the start of the simulation if `init_prep_inj_na` is True to determine whether NA tests should be used for those on injectable PrEP in the general population.
- *`test_sens_general`* - The base value for general population HIV test sensitivity.
- *`test_sens_primary_ab`* - Ab test sensitivity for someone in primary infection.
- *`test_sens_prep_inj_primary_ab`* - Ab test sensitivity for someone in primary infection that is using injectable PrEP.
- *`test_sens_prep_inj_3m_ab`* - Ab test sensitivity for someone infected for 3-5 months that is using injectable PrEP.
- *`test_sens_prep_inj_ge6m_ab`* - Ab test sensitivity for someone infected for 6 or more months that is using injectable PrEP.
- *`tests_sens_prep_inj`* - An index used for NA test sensitivity selection.
- *`test_sens_prep_inj_primary_na`* - NA test sensitivity for someone in primary infection that is using injectable PrEP.
- *`test_sens_prep_inj_3m_na`* - NA test sensitivity for someone infected for 3 months that is using injectable PrEP.
- *`test_sens_prep_inj_ge6m_na`* - NA test sensitivity for someone infected for 6 or more months that is using injectable PrEP.
- *`prob_loss_at_diag`* - The base probability of loss of care at diagnosis.
- *`sw_incr_prob_loss_at_diag`* - A multiplier applied to sex workers that may increase their chances of disengaging with care at diagnosis.
- *`higher_newp_less_engagement`* - A boolean value indicating whether people with short-term partners are less likely to be engaged with care.
- *`prob_loss_at_diag_adc_tb`* - The probability of loss of care at diagnosis if the patient is infected with an ADC or TB.
- *`prob_loss_at_diag_non_tb_who3`* - The probability of loss of care at diagnosis if the patient is infected with a non-TB WHO3 disease.
