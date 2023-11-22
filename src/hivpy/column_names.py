"""
Column names in the Population Data-Frame.
"""

SEX = "sex"                                     # common.SexType: SexType.Male for male and SexType.Female for female
AGE = "age"                                     # float: age at the current date
AGE_GROUP = "age_group"                         # int: discrete age grouping based on age
LTP_AGE_GROUP = "ltp_age_group"                 # int: discrete age group for starting / longevity of ltp

RISK = "risk"                                   # float: overall risk value from combined factors
RISK_AGE = "risk_age"                           # float: risk reduction factor based on age
RISK_ADC = "risk_adc"                           # float: risk reduction for AIDS defining condition
RISK_BALANCE = "risk_balance"                   # float: risk reduction factor to re-balance male & female partner numbers
RISK_DIAGNOSIS = "risk_diagnosis"               # float: risk reduction associated with recent HIV diagnosis
RISK_PERSONAL = "risk_personal"                 # float: individual risk reduction applied with a certain probability
RISK_LTP = "risk_long_term_partnered"           # float: risk reduction for people in long term partnerships
RISK_ART_ADHERENCE = "risk_art_adherence"       # float: risk reduction associated with low ART adherence
RISK_INTIIAL = "risk_initial"                   # float: initial risk reduction factor
CIRCUMCISED = "circumcised"                     # bool: True if a man is circumcised
CIRCUMCISION_DATE = "circumcision_date"         # None | datetime.date: date of circumcision if circumcised, o/w None
VMMC = "vmmc"                                   # bool: True if voluntary medical male circumcision was applied
NUM_PARTNERS = "num_partners"                   # float: number of short term condomless sex partners during the current time step
SEX_MIX_AGE_GROUP = "sex_mix_age_group"         # int: Discrete age group for sexual mixing
STP_AGE_GROUPS = "stp_age_groups"               # int array: ages groups of short term partners
SEX_BEHAVIOUR = "sex_behaviour"                 # int: sexual behaviour grouping
SEX_BEHAVIOUR_CLASS = "sex_class"               # sexual_behaviour.SexBehaviourClass (enum): Men, Young Women, Older Women, or Sex Worker
LONG_TERM_PARTNER = "long_term_partner"         # bool: True if the subject has a long term condomless partner
LTP_LONGEVITY = "ltp_longevity"                 # int: categorises longevity of long term partnerships (higher => more stable)
LOW_FERTILITY = "low_fertility"                 # bool: True if a woman is considered to have a 0% chance of pregnancy, o/w False
PREGNANT = "pregnant"                           # bool: True if a woman is currently pregnant
ANC = "anc"                                     # bool: True if in antenatal care
PMTCT = "pmtct"                                 # bool: True if undergoing prevention of mother to child transmission care
LAST_PREGNANCY_DATE = "last_pregnancy_date"     # None | datetime.date: date of most recent pregnancy, o/w None if never pregnant
INFECTED_BIRTH = "infected_birth"               # bool: True if a woman's last child was born with HIV
NUM_CHILDREN = "num_children"                   # int: number of children a woman has
NUM_HIV_CHILDREN = "num_HIV_children"           # int: number of children infected with HIV a woman has
WANT_NO_CHILDREN = "want_no_children"           # bool: True if a woman does not want any more children

SEX_WORKER = "sex_worker"                       # bool: True if person is a sex worker, false otherwise
SW_AGE_GROUP = "sw_age_group"                   # int: categorises sex worker behaviour by age
DATE_STOP_SW = "date_stop_sex_work"             # date: date at which a former sex worker (last) stopped sex work
SW_PROGRAM_VISIT = "sw_program_visit"           # bool: is sex worker engaged with sex worker program?
SW_TEST_6MONTHLY = "sw_test_6monthly"           # bool: sex worker tested for HIV bi-annually
SW_HIGHER_INT = "sw_higher_int"                 # bool: TODO: something sex workers and ART disadvantage?
SW_ART_DISENGAGE = "sw_art_disengage"           # bool: whether sex workers have lower levels of ART engagement
SW_PROGRAM = "sw_program"                       # bool: Is a sex worker program in place?
SW_PROGRAM_EFFECT = "sw_program_effect"         # enum: STRONG or WEAK program efficacy
EVER_SEX_WORKER = "ever_sex_worker"             # bool: if person has ever been a sex worker
AGE_STOP_SEX_WORK = "age_stop_sex_work"         # float: age at which former sex worker (last) stopped sex work
LIFE_SEX_RISK = "life_sex_risk"                 # int: value in (1, 2, 3) indicating risk of adopting sex work

HARD_REACH = "hard_reach"                       # bool: True if person is reluctant to test for HIV (also affects PrEP and VMMC), but will still test if symptomatic or in antenatal care
EVER_TESTED = "ever_tested"                     # bool: True if person has ever been tested for HIV
LAST_TEST_DATE = "last_test_date"               # None | datetime.date: date of last HIV test
NSTP_LAST_TEST = "nstp_last_test"               # int: Number of short term condomless sex partners since last test (DUMMY)
NP_LAST_TEST = "np_last_test"                   # int: Total number of condomless sex partners since last test (DUMMY)
HIV_STATUS = "HIV_status"                       # bool: True if person is HIV positive, o/w False
HIV_DIAGNOSED = "HIV_diagnosed"                 # bool: True if you have had a positive HIV test
HIV_DIAGNOSIS_DATE = "HIV_Diagnosis_Date"       # None | datetime.date: date of HIV diagnosis (to nearest timestep) if HIV+, o/w None
VIRAL_LOAD_GROUP = "viral_load_group"           # int: value 1-6 placing bounds on viral load for an HIV positive person
ART_NAIVE = "art_naive"                         # bool: True if person has never been on antiretroviral therapy

ADC = "AIDS_defining_condition"                 # Bool: presence of AIDS defining condition

DATE_OF_DEATH = "date_of_death"                 # None | datetime.date: date of death if dead, o/w None

ART_ADHERENCE = "art_adherence"                 # DUMMY
