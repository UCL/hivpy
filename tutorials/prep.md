## PrEP Tutorial

The PrEP module tracks PrEP preferences, willingness, eligibility, and use, including starting, stopping, and restarting PrEP. The most relevant files are listed below:

- `src/hivpy/prep.py` - The PrEP module.
- `src/tests/test_prep.py` - Tests for the PrEP module.
- `src/hivpy/data/prep.yaml` - PrEP data and variables.
- `src/hivpy/prep_data.py` - A class for storing data loaded from `prep.yaml`.

If there are any PrEP-related variables you would like to change before running your simulation, please change them in `prep.yaml`.

### Module Overview

When PrEP is updated, the first thing to be calculated is PrEP propensity for the population aged 15 and over. For each individual, this includes: setting preference values for each type of PrEP, determining which types of PrEP they are willing to take, setting preference ranks for each type of PrEP, and calculating their 'favoured PrEP' – the type of PrEP with the highest preference value someone is willing to take that is also currently available.

Next, PrEP eligibility is determined. Based on the current PrEP strategy, a sub-population of people is selected to be eligible for PrEP usage this time step. Eligible people typically either have short-term partners or have a long-term partner who is not on ART.

Finally, PrEP usage is updated for anyone starting, continuing, stopping, or restarting PrEP. (`Note`: PrEP usage relies on the assumption that HIV diagnosis has already taken place in order to identify people that are HIV positive but have falsely not been diagnosed.)

Individuals can be specifically tested to start PrEP for the first time, but tested people in the general population can also decide to start PrEP. When people continue PrEP usage, they can either continue with their current PrEP or switch to a different type if their favoured PrEP has changed.

PrEP usage can be stopped for two reasons – an individual can choose to stop, or they can become ineligible for PrEP. Temporary ineligibility, where PrEP usage is considered paused rather than stopped outright, can occur due to lack of risk or a change in partnership, but permanent ineligibility is reached when an individual is diagnosed with HIV or when they reach age 65+.

People who have chosen to stop taking PrEP but are still eligible can also choose to restart, but anyone who paused PrEP usage due to a break in eligibility automatically restarts.

### PrEP Columns

- *`R_PREP`* - A semi-permanent random float variable that determines whether an individual is risk informed or suspects they are at risk enough to take PrEP. Rerolled only for ineligible people when determining PrEP eligibility each time step.
- *`PREP_ORAL_PREF`* - A float value between [0, 1] drawn from a beta distribution that determines an individual's preference for oral PrEP.
- *`PREP_CAB_PREF`* - A float value between [0, 1] drawn from a beta distribution that determines an individual's preference for cabotegravir PrEP.
- *`PREP_LEN_PREF`* - A float value between [0, 1] drawn from a beta distribution that determines an individual's preference for lenacapavir PrEP.
- *`PREP_VR_PREF`* - A float value between [0, 1] drawn from a beta distribution that determines an individual's preference for vaginal ring PrEP.
- *`PREP_ORAL_RANK`* - An integer value between [1, 4] representing an individual's ranked PrEP preference for oral PrEP.
- *`PREP_CAB_RANK`* - An integer value between [1, 4] representing an individual's ranked PrEP preference for cabotegravir PrEP.
- *`PREP_LEN_RANK`* - An integer value between [1, 4] representing an individual's ranked PrEP preference for lenacapavir PrEP.
- *`PREP_VR_RANK`* - An integer value between [1, 4] representing an individual's ranked PrEP preference for vaginal ring PrEP.
- *`PREP_ORAL_WILLING`* - A boolean flag signifying whether an individual is willing to use oral PrEP. True if their oral preference value clears a willingness threshold.
- *`PREP_CAB_WILLING`* - A boolean flag signifying whether an individual is willing to use cabotegravir PrEP. True if their cab preference value clears a willingness threshold.
- *`PREP_LEN_WILLING`* - A boolean flag signifying whether an individual is willing to use lenacapavir PrEP. True if their len preference value clears a willingness threshold.
- *`PREP_VR_WILLING`* - A boolean flag signifying whether an individual is willing to use vaginal ring PrEP. True if their vr preference value clears a willingness threshold.
- *`FAVOURED_PREP_TYPE`* - The `PrEPType` with the highest preference value an individual is willing to take that is also currently available. If there is no PrEP available that they are willing to take, this value is set to None.
- *`PREP_ELIGIBLE`* - A boolean flag signifying whether an individual is eligible for PrEP this time step.
- *`PREP_ORAL_TESTED`* - A boolean flag signifying whether an individual has tested for HIV specifically for the purpose of starting oral PrEP. `Note`: Currently dummied.
- *`PREP_CAB_TESTED`* - A boolean flag signifying whether an individual has tested for HIV specifically for the purpose of starting cabotegravir PrEP. `Note`: Currently dummied.
- *`PREP_LEN_TESTED`* - A boolean flag signifying whether an individual has tested for HIV specifically for the purpose of starting lenacapavir PrEP. `Note`: Currently dummied.
- *`PREP_VR_TESTED`* - A boolean flag signifying whether an individual has tested for HIV specifically for the purpose of starting vaginal ring PrEP. `Note`: Currently dummied.
- *`PREP_TYPE`* - The most recent `PrEPType` an individual has used, otherwise None if they have never used PrEP. This column is kept intact upon stopping PrEP usage.
- *`EVER_PREP`* - A boolean flag signifying whether an individual has ever been on PrEP.
- *`FIRST_ORAL_START_DATE`* - The start date of an individual's first ever usage of oral PrEP, otherwise None if they have never used oral PrEP.
- *`FIRST_CAB_START_DATE`* - The start date of an individual's first ever usage of cabotegravir PrEP, otherwise None if they have never used cab PrEP.
- *`FIRST_LEN_START_DATE`* - The start date of an individual's first ever usage of lenacapavir PrEP, otherwise None if they have never used len PrEP.
- *`FIRST_VR_START_DATE`* - The start date of an individual's first ever usage of vaginal ring PrEP, otherwise None if they have never used vr PrEP.
- *`LAST_PREP_START_DATE`* - The start date of an individual's most recent period of PrEP usage.
- *`PREP_JUST_STARTED`* - A boolean flag signifying whether an individual started using PrEP this time step.
- *`LAST_PREP_USE_DATE`* - The date of an individual's most recent PrEP usage (at a time step granularity).
- *`LAST_PREP_STOP_DATE`* - The stop date of an individual's most recent period of PrEP usage. Reset to None if they restart PrEP.
- *`PREP_PAUSED`* - A boolean flag signifying whether an individual has paused their PrEP usage this time step due to temporary ineligibility or lack of risk.
- *`ON_PREP`* - A boolean flag signifying whether an individual is currently taking PrEP this time step.
- *`CONT_ON_PREP`* - A timedelta tracking the total length of continuous PrEP usage (at a time step granularity) of the current type of PrEP based on user intention. Choosing to stop using PrEP or becoming permanently ineligible resets continuity, but pausing PrEP usage will simply freeze this count until PrEP is being actively taken again.
- *`CONT_INTENT_ON_PREP`* - As `CONT_ON_PREP`, but this count will continue to increment even when an individual has paused PrEP usage. Choosing to stop using PrEP or becoming permanently ineligible will reset continuity.
- *`CONT_ACTIVE_ON_PREP`* - A timedelta tracking the actual total length of continuous PrEP usage (at a time step granularity) of the current type of PrEP. Choosing to stop using PrEP and dropping out due to ineligibility (temporary or otherwise) will reset continuity.
- *`CUMULATIVE_PREP_ORAL`* - A timedelta tracking the total length of cumulative oral PrEP usage.
- *`CUMULATIVE_PREP_CAB`* - A timedelta tracking the total length of cumulative cabotegravir PrEP usage.
- *`CUMULATIVE_PREP_LEN`* - A timedelta tracking the total length of cumulative lenacapavir PrEP usage.
- *`CUMULATIVE_PREP_VR`* - A timedelta tracking the total length of cumulative vaginal ring PrEP usage.

### PrEP Data Variables

- *`prep_strategy`* - An integer that determines which sub-population of people is marked as eligible for PrEP.
- *`date_prep_intro`* - An array containing the introduction dates for each type of PrEP. Intended to be accessed through the use of `PrEPType`s as indices (e.g. `date_prep_intro[PrEPType.Oral]`).
- *`cab_available`* - The boolean flag that determines the availability of cabotegravir PrEP. Cab is only available if the current date has reached the cab introduction date and this flag is True.
- *`prob_risk_informed_prep`* - The probability of an individual with an *uninfected* long-term partner who is not on ART to meet the risk-informed PrEP eligibility criteria.
- *`prob_greater_risk_informed_prep`* - As `prob_risk_informed_prep`, but with a higher probability value. Used with certain `prep_strategy` values.
- *`prob_suspect_risk_prep`* - The probability of an individual with an *infected* long-term partner who is not on ART to meet the risk-informed PrEP eligibility criteria.
- *`prep_oral_pref_beta`* - The alpha value used to draw random oral PrEP preference values for the population from a beta distribution.
- *`prep_cab_pref_beta`* - The alpha value used to draw random cabotegravir PrEP preference values for the population from a beta distribution.
- *`prep_len_pref_beta`* - The alpha value used to draw random lenacapavir PrEP preference values for the population from a beta distribution.
- *`prep_vr_pref_beta`* - The alpha value used to draw random vaginal ring PrEP preference values for the population from a beta distribution.
- *`prep_willing_threshold`* - A threshold value that a PrEP preference value must exceed in order for an individual to be willing to take the respective type of PrEP.
- *`vl_prevalence_affects_prep`* - A boolean flag that dictates whether willingness to take PrEP is affected by low unsuppressed viral load prevalence in the population.
- *`vl_prevalence_prep_threshold`* - The threshold at which low unsuppressed viral load prevalence affects PrEP willingness.
- *`rate_test_onprep_any`* - The rate of being tested for HIV while on PrEP.
- *`prob_test_prep_start`* - The probability of being tested for HIV with the intent to start PrEP.
- *`prob_base_prep_start`* - The base probability of starting any type of PrEP for the first time.
- *`prob_oral_prep_start`* - The probability of starting oral PrEP for the first time.
- *`prob_cab_prep_start`* - The probability of starting cabotegravir PrEP for the first time.
- *`prob_len_prep_start`* - The probability of starting lenacapavir PrEP for the first time.
- *`prob_vr_prep_start`* - The probability of starting vaginal ring PrEP for the first time.
- *`prob_oral_prep_stop`* - The probability of choosing to stop taking oral PrEP despite being eligible.
- *`prob_cab_prep_stop`* - The probability of choosing to stop taking cabotegravir PrEP despite being eligible.
- *`prob_len_prep_stop`* - The probability of choosing to stop taking lenacapavir PrEP despite being eligible.
- *`prob_vr_prep_stop`* - The probability of choosing to stop taking vaginal ring PrEP despite being eligible.
- *`prob_prep_restart`* - The probability of restarting any type of PrEP after choosing to stop taking PrEP.
