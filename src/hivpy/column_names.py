"""
Column names in the Population Data-Frame.
"""

SEX = "sex"                                     # common.SexType: SexType.Male for male and SexType.Female for female
AGE = "age"                                     # float: age at the current date
AGE_GROUP = "age_group"                         # int: discrete age grouping based on age
LTP_AGE_GROUP = "ltp_age_group"                 # int: discrete age group for starting / longevity of ltp

RRED = "rred"                                   # float: overall rred value from combined factors
RRED_AGE = "rred_age"                           # float: risk reduction factor based on age
RRED_ADC = "rred_adc"                           # float: risk reduction for AIDS defining condition
RRED_BALANCE = "rred_balance"                   # float: risk reduction factor to re-balance male & female partner numbers
RRED_DIAGNOSIS = "rred_diagnosis"               # float: risk reduction associated with recent HIV diagnosis
RRED_PERSONAL = "rred_personal"                 # float: individual risk reduction applied with a certain probability
RRED_LTP = "rred_long_term_partnered"           # float: risk reduction for people in long term partnerships
RRED_ART_ADHERENCE = "rred_art_adherence"       # float: risk reduction associated with low ART adherence
RRED_INTIIAL = "rred_initial"                   # float: initial risk reduction factor
CIRCUMCISED = "circumcised"                     # bool: True if a man is circumcised
CIRCUMCISION_DATE = "circumcision_date"         # None | datetime.date: date of circumcision if circumcised, o/w None
VMMC = "vmmc"                                   # bool: True if voluntary medical male circumcision was applied
NUM_PARTNERS = "num_partners"                   # float: number of short term condomless sex partners during the current time step
SEX_MIX_AGE_GROUP = "sex_mix_age_group"         # int: Discrete age group for sexual mixing
STP_AGE_GROUPS = "stp_age_groups"               # int array: ages groups of short term partners
SEX_BEHAVIOUR = "sex_behaviour"                 # int: sexual behaviour grouping
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

HARD_REACH = "hard_reach"                       # bool: True if person is reluctant to test for HIV (also affects PrEP and VMMC), but will still test if symptomatic or in antenatal care
HIV_STATUS = "HIV_status"                       # bool: True if person is HIV positive, o/w False
HIV_DIAGNOSIS_DATE = "HIV_Diagnosis_Date"       # None | datetime.date: date of HIV diagnosis (to nearest timestep) if HIV+, o/w None
VIRAL_LOAD_GROUP = "viral_load_group"           # int: value 1-6 placing bounds on viral load for an HIV positive person
ART_NAIVE = "art_naive"                         # bool: True if person has never been on antiretroviral therapy

ADC = "AIDS_defining_condition"                 # Bool: presence of AIDS defining condition

DATE_OF_DEATH = "date_of_death"                 # None | datetime.date: date of death if dead, o/w None

ART_ADHERENCE = "art_adherence"                 # DUMMY
