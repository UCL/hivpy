"""Column names in the Population Data-Frame."""

SEX = "sex"                                 # common.SexType: SexType.Male for male and SexType.Female for female
AGE = "age"                                 # float: age at the current date
AGE_GROUP = "age_group"                     # int: discrete age grouping based on age

RRED = "rred"                               # float: overall rred value from combined factors
RRED_AGE = "rred_age"                       # float: risk reduction factor based on age
RRED_ADC = "rred_adc"                       # float: risk reduction for AIDS defining condition
RRED_BALANCE = "rred_balance"               # float: risk reduction factor to re-balance male & female partner numbers
RRED_DIAGNOSIS = "rred_diagnosis"           # float: risk reduction associated with recent HIV diagnosis
RRED_PERSONAL = "rred_personal"             # float: individual risk reduction applied with a certain probability
RRED_LTP = "rred_long_term_partnered"       # float: risk reduction for people in long term partnerships
RRED_ART_ADHERENCE = "rred_art_adherence"   # float: risk reduction associated with low ART adherence
RRED_INTIIAL = "rred_initial"               # float: initial risk reduction factor
NUM_PARTNERS = "num_partners"               # float: number of short term condomless sex partners during the current time step
SEX_MIX_AGE_GROUP = "sex_mix_age_group"     # int: Discrete age group for sexual mixing
STP_AGE_GROUPS = "stp_age_groups"           # int array: ages groups of short term partners
SEX_BEHAVIOUR = "sex_behaviour"             # int: sexual behaviour grouping
LONG_TERM_PARTNER = "long_term_partner"     # bool: True if the subject has a long term condomless partner
LTP_LONGEVITY = "ltp_longevity"             # int: categorises longevity of long term partnerships (higher => more stable)

HIV_STATUS = "HIV_status"                   # bool: true if person if HIV positive, o/w false
HIV_DIAGNOSIS_DATE = "HIV_Diagnosis_Date"   # None | datetime.date: date of HIV diagnosis (to nearest timestep) if HIV+, o/w None
VIRAL_LOAD_GROUP = "viral_load_group"       # int: value 1-6 placing bounds on viral load for an HIV positive person

DATE_OF_DEATH = "date_of_death"             # None | datetime.date: date of death if dead, o/w None
