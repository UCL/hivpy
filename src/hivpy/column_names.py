"""
Column names in the Population Data-Frame.
"""

SEX = "sex"                                     # common.SexType: SexType.Male for male and SexType.Female for female
AGE = "age"                                     # float: age at the current date
AGE_GROUP = "age_group"                         # int: discrete age grouping based on age
LTP_AGE_GROUP = "ltp_age_group"                 # int: discrete age group for starting / longevity of ltp

RISK = "risk"                                   # float: overall risk value from combined factors
RISK_AGE = "risk_age"                           # float: risk factor based on age
RISK_ADC = "risk_adc"                           # float: risk for AIDS defining condition
RISK_BALANCE = "risk_balance"                   # float: risk factor to re-balance male & female partner numbers
RISK_DIAGNOSIS = "risk_diagnosis"               # float: risk associated with recent HIV diagnosis
RISK_PERSONAL = "risk_personal"                 # float: individual risk reduction applied with a certain probability
RISK_LTP = "risk_long_term_partnered"           # float: risk reduction for people in long term partnerships
RISK_ART_ADHERENCE = "risk_art_adherence"       # float: risk reduction associated with low ART adherence
RISK_INITIAL = "risk_initial"                   # float: initial risk reduction factor
CIRCUMCISED = "circumcised"                     # bool: True if a man is circumcised
CIRCUMCISION_DATE = "circumcision_date"         # None | datetime.date: date of circumcision if circumcised, o/w None
VMMC = "vmmc"                                   # bool: True if voluntary medical male circumcision was applied

NUM_PARTNERS = "num_partners"                   # float: number of short term condomless sex partners during the current time step
SEX_MIX_AGE_GROUP = "sex_mix_age_group"         # int: discrete age group for sexual mixing
STP_AGE_GROUPS = "stp_age_groups"               # int array: age groups of short term partners
SEX_BEHAVIOUR = "sex_behaviour"                 # int: sexual behaviour grouping
SEX_BEHAVIOUR_CLASS = "sex_class"               # sexual_behaviour.SexBehaviourClass(enum): Men, Young Women, Older Women, or Sex Workers
LONG_TERM_PARTNER = "long_term_partner"         # bool: True if the subject has a long term condomless partner
LTP_LONGEVITY = "ltp_longevity"                 # int: categorises longevity of long term partnerships (higher => more stable)
LOW_FERTILITY = "low_fertility"                 # bool: True if a woman is considered to have a 0% chance of pregnancy, o/w False
PREGNANT = "pregnant"                           # bool: True if a woman is currently pregnant
ANC = "anc"                                     # bool: True if in antenatal care
PMTCT = "pmtct"                                 # bool: True if undergoing prevention of mother to child transmission care
LAST_PREGNANCY_DATE = "last_pregnancy_date"     # None | datetime.date: date of most recent pregnancy, o/w None if never pregnant
NUM_CHILDREN = "num_children"                   # int: number of children a woman has
NUM_HIV_CHILDREN = "num_HIV_children"           # int: number of children infected with HIV a woman has
WANT_NO_CHILDREN = "want_no_children"           # bool: True if a woman does not want any more children

SEX_WORKER = "sex_worker"                       # bool: True if person is a sex worker, o/w False
SW_AGE_GROUP = "sw_age_group"                   # int: categorises sex worker behaviour by age
DATE_STOP_SW = "date_stop_sex_work"             # date: date at which a former sex worker (last) stopped sex work
SW_PROGRAM_VISIT = "sw_program_visit"           # bool: True if sex worker is engaged with sex worker program
SW_TEST_6MONTHLY = "sw_test_6monthly"           # bool: True if sex worker is tested for HIV bi-annually
SW_HIGHER_INT = "sw_higher_int"                 # bool: TODO: something sex workers and ART disadvantage?
SW_ART_DISENGAGE = "sw_art_disengage"           # bool: True if sex worker has a lower levels of ART engagement
SW_PROGRAM = "sw_program"                       # bool: True if a sex worker program is in place
SW_PROGRAM_EFFECT = "sw_program_effect"         # enum: STRONG or WEAK sex worker program efficacy
EVER_SEX_WORKER = "ever_sex_worker"             # bool: True if person has ever been a sex worker
AGE_STOP_SEX_WORK = "age_stop_sex_work"         # float: age at which former sex worker (last) stopped sex work
LIFE_SEX_RISK = "life_sex_risk"                 # int: value in (1, 2, 3) indicating risk of adopting sex work
STI = "sti"                                     # bool: True if has sexually transmitted infection (non HIV), false o/w (TODO: DUMMIED)

HARD_REACH = "hard_reach"                       # bool: True if person is reluctant to test for HIV (also affects PrEP and VMMC), but will still test if symptomatic or in antenatal care
EVER_TESTED = "ever_tested"                     # bool: True if person has ever been tested for HIV
LAST_TEST_DATE = "last_test_date"               # None | datetime.date: date of last HIV test
NSTP_LAST_TEST = "nstp_last_test"               # int: number of short term condomless sex partners since last test (DUMMY)
NP_LAST_TEST = "np_last_test"                   # int: total number of condomless sex partners since last test (DUMMY)
HIV_STATUS = "HIV_status"                       # bool: True if person is HIV positive, o/w False
DATE_HIV_INFECTION = "date_HIV_infection"       # None | date: date of HIV infection if HIV+, o/w None
IN_PRIMARY_INFECTION = "in_primary_infection"   # bool: True if a person contracted HIV within 3 months of the current date, o/w False
HIV_INFECTION_GE6M = "HIV_infection_ge6m"       # bool: True is a person has been infected with HIV for 6 months or more (DUMMY)
HIV_DIAGNOSED = "HIV_diagnosed"                 # bool: True if individual had a positive HIV test
HIV_DIAGNOSIS_DATE = "HIV_Diagnosis_Date"       # None | datetime.date: date of HIV diagnosis (to nearest timestep) if HIV+, o/w None
UNDER_CARE = "under_care"                       # bool: True if under care after a positive HIV diagnosis
VIRAL_LOAD_GROUP = "viral_load_group"           # int: value 0-5 placing bounds on viral load for an HIV positive person
VIRAL_LOAD = "viral_load"                       # float: viral load for HIV+ person
CD4 = "cd4"                                     # None | float: CD4 count per cubic millimeter; set to None for people w/o HIV
MAX_CD4 = "max_cd4"                             # float: maximum CD4 count to which a person can return when on ART
ART_NAIVE = "art_naive"                         # bool: True if person has never been on antiretroviral therapy
X4_VIRUS = "x4_virus"                           # bool: True if X4 virus is present in person, False otherwise

WHO3_EVENT = "who3_event"                       # Bool: True if who3 disease occurs this timestep in HIV positive person
NON_TB_WHO3 = "non_tb_who3"                     # Bool: True if non-tb who3 disease occurs this timestep in HIV positive person
TB = "tb"                                       # Bool: True if tb occurs this timestep in HIV positive person
TB_DIAGNOSED = "tb_diagnosed"                   # Bool: True if TB is diagnosed this timestep
TB_INFECTION_DATE = "tb_infection_date"         # date: Date of start of most recent TB infection
C_MENINGITIS = "c_meningitis"                       # Bool: True if cryptococcal meningitis occurs this timestep
C_MENINGITIS_DIAGNOSED = "c_meningitis_diagnosed"   # Bool: True if cryptococcal meningitis diagnosed this timestep
SBI = "serious_bacterial_infection"             # Bool: True if serious bacterial infection this time step
SBI_DIAGNOSED = "sbi_diagnosed"                 # Bool: True if SBI diagnosed this time step
WHO4_OTHER = "who4_other"                       # Bool: True if other WHO4 disease occurs this timestep
WHO4_OTHER_DIAGNOSED = "who4_other_diagnosed"   # Bool: True if other WHO4 disease diagnosed this timestep
ADC = "AIDS_defining_condition"                 # Bool: presence of AIDS defining condition (any WHO4)
PREP_TYPE = "prep_type"                         # None | prep.PrEPType(enum): Oral, Cabotegravir, Lenacapavir, or VaginalRing if PrEP is being used, o/w None (DUMMY)
PREP_JUST_STARTED = "prep_just_started"         # Bool: True if PrEP usage began this time step (DUMMY)

ART_ADHERENCE = "art_adherence"                 # DUMMY

TA_MUTATION = "tam"                             # X_MUTATION: drug resistance cols (TODO: all 24 currently DUMMIED)
M184_MUTATION = "m184m"
K65_MUTATION = "k65m"
Q151_MUTATION = "q151m"
K103_MUTATION = "k103m"
Y181_MUTATION = "y181m"
G190_MUTATION = "g190m"
P32_MUTATION = "p32m"
P33_MUTATION = "p33m"
P46_MUTATION = "p46m"
P47_MUTATION = "p47m"
P50L_MUTATION = "p50lm"
P50V_MUTATION = "p50vm"
P54_MUTATION = "p54m"
P76_MUTATION = "p76m"
P82_MUTATION = "p82m"
P84_MUTATION = "p84m"
P88_MUTATION = "p88m"
P90_MUTATION = "p90m"
IN118_MUTATION = "in118m"
IN140_MUTATION = "in140m"
IN148_MUTATION = "in148m"
IN155_MUTATION = "in155m"
IN263_MUTATION = "in263m"
