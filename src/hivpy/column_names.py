"""
Column names in the Population Data-Frame.
"""

SEX = "sex"                                 # common.SexType: SexType.Male for male and SexType.Female for female
AGE = "age"                                 # float: age at the current date
AGE_GROUP = "age_group"                     # int: discrete age grouping based on age
LTP_AGE_GROUP = "ltp_age_group"             # int: discrete age group for starting / longevity of ltp

RISK = "risk"                               # float: overall risk value from combined factors
RISK_AGE = "risk_age"                       # float: risk reduction factor based on age
RISK_ADC = "risk_adc"                       # float: risk reduction for AIDS defining condition
RISK_BALANCE = "risk_balance"               # float: risk reduction factor to re-balance male & female partner numbers
RISK_DIAGNOSIS = "risk_diagnosis"           # float: risk reduction associated with recent HIV diagnosis
RISK_PERSONAL = "risk_personal"             # float: individual risk reduction applied with a certain probability
RISK_LTP = "risk_long_term_partnered"       # float: risk reduction for people in long term partnerships
RISK_ART_ADHERENCE = "risk_art_adherence"   # float: risk reduction associated with low ART adherence
RISK_INTIIAL = "risk_initial"               # float: initial risk reduction factor
CIRCUMCISED = "circumcised"                 # bool: True if a man is circumcised
CIRCUMCISION_DATE = "circumcision_date"     # None | datetime.date: date of circumcision if circumcised, o/w None
VMMC = "vmmc"                               # bool: True if voluntary medical male circumcision was applied
NUM_PARTNERS = "num_partners"               # float: number of short term condomless sex partners during the current time step
SEX_MIX_AGE_GROUP = "sex_mix_age_group"     # int: Discrete age group for sexual mixing
STP_AGE_GROUPS = "stp_age_groups"           # int array: ages groups of short term partners
SEX_BEHAVIOUR = "sex_behaviour"             # int: sexual behaviour grouping
LONG_TERM_PARTNER = "long_term_partner"     # bool: True if the subject has a long term condomless partner
LTP_LONGEVITY = "ltp_longevity"             # int: categorises longevity of long term partnerships (higher => more stable)

SEX_WORKER = "sex_worker"                   # bool: True if person is a sex worker, false otherwise
SW_AGE_GROUP = "sw_age_group"               # int: categorises sex worker behaviour by age
DATE_STOP_SW = "date_stop_sex_work"         # date: date at which a former sex worker (last) stopped sex work
SW_PROGRAM_VISIT = "sw_program_visit"       # bool: is sex worker engaged with sex worker program?
SW_TEST_6MONTHLY = "sw_test_6monthly"       # bool: sex worker tested for HIV bi-annually
SW_HIGHER_INT = "sw_higher_int"             # bool: TODO: something sex workers and ART disadvantage?
SW_ART_DISENGAGE = "sw_art_disengage"       # bool: whether sex workers have lower levels of ART engagement
SW_PROGRAM = "sw_program"                   # bool: Is a sex worker program in place?
SW_PROGRAM_EFFECT = "sw_program_effect"     # enum: STRONG or WEAK program efficacy

HARD_REACH = "hard_reach"                   # bool: True if person is reluctant to test for HIV (also affects PrEP and VMMC), but will still test if symptomatic or in antenatal care
HIV_STATUS = "HIV_status"                   # bool: True if person is HIV positive, o/w False
HIV_DIAGNOSIS_DATE = "HIV_Diagnosis_Date"   # None | datetime.date: date of HIV diagnosis (to nearest timestep) if HIV+, o/w None
VIRAL_LOAD_GROUP = "viral_load_group"       # int: value 1-6 placing bounds on viral load for an HIV positive person

ADC = "AIDS_defining_condition"             # Bool: presence of AIDS defining condition

DATE_OF_DEATH = "date_of_death"             # None | datetime.date: date of death if dead, o/w None

ART_ADHERENCE = "art_adherence"        # DUMMY
