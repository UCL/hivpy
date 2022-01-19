import numpy as np
import scipy.stats as stat

# Baseline sexual risk for age and sex
# age groups 15 <= age < 20
#            20 <= age < 25
#            ...
#            60 <= age
baseline_risk = np.array([[0.3,   1.8],
                          [0.4,   1.8],
                          [0.85,  1.0],
                          [1.0,   0.8],
                          [0.85,  0.5],
                          [0.5,   0.35],
                          [0.4,   0.3],
                          [0.35,  0.1],
                          [0.2,   0.03],
                          [0.15,  0.02]])

# Sexual behaviour transition matrices

# Male
# Probability of transitions between male sexual behaviour groups
# 0 = Zero, 1 = Low, 2 = Medium, 3 = High
sex_behaviour_trans_male_options = np.array([
    [[0.995, 0.005, 0.005, 0.00005],
     [0.95,  0.03,  0.02,  0.00005],
     [0.03,  0.07,  0.90,  0.00025],
     [0.0,   0.0,   0.05,  0.95]],

    [[0.98, 0.01, 0.01, 0.00025],
     [0.98, 0.01, 0.01, 0.00025],
     [0.05, 0.15, 0.80, 0.00125],
     [0.0,  0.0,  0.2,  0.8]],

    [[0.95, 0.03, 0.02, 0.0005],
     [0.93, 0.05, 0.02, 0.0005],
     [0.2,  0.2,  0.6,  0.0025],
     [0.0,  0.0,  0.4,  0.6]],

    [[0.995, 0.005, 0.005, 0.0001],
     [0.95,  0.03,  0.02,  0.0001],
     [0.03,  0.07,  0.9,   0.0005],
     [0.04,  0.04,  0.09,  0.83]],

    [[0.98,  0.01, 0.01, 0.005],
     [0.98,  0.01, 0.01, 0.005],
     [0.05,  0.15, 0.8,  0.0025],
     [0.025, 0.06, 0.17, 0.75]],

    [[0.95, 0.03, 0.02, 0.001],
     [0.93, 0.05, 0.02, 0.001],
     [0.2,  0.2,  0.6,  0.005],
     [0.04, 0.08, 0.21, 0.67]],

    [[0.995, 0.005, 0.005, 0.000025],
     [0.95,  0.03,  0.02,  0.000025],
     [0.03,  0.07,  0.90,  0.000125],
     [0.0,   0.0,   0.05,  0.95]],

    [[0.98, 0.01, 0.01, 0.000125],
     [0.98, 0.01, 0.01, 0.000125],
     [0.05, 0.15, 0.8,  0.000625],
     [0.0,  0.0,  0.2,  0.8]],

    [[0.95, 0.03, 0.02, 0.00025],
     [0.93, 0.05, 0.02, 0.00025],
     [0.2,  0.2,  0.6,  0.00125],
     [0.0,  0.0,  0.4,  0.6]],

    [[0.9,  0.06,  0.04,  0.0005],
     [0.99, 0.005, 0.005, 0.0005],
     [0.2,  0.2,   0.6,   0.0025],
     [0.0,  0.0,   0.4,   0.6]],

    [[0.9,  0.06,  0.04,  0.001],
     [0.99, 0.005, 0.005, 0.001],
     [0.2,  0.2,   0.6,   0.005],
     [0.04, 0.08,  0.21,  0.67]],

    [[0.9,  0.06,  0.04,  0.00025],
     [0.99, 0.005, 0.005, 0.00025],
     [0.2,  0.2,   0.6,   0.00125],
     [0.0,  0.0,   0.0,   1.0]],

    [[0.75, 0.15,  0.10,  0.0005],
     [0.99, 0.005, 0.005, 0.0005],
     [0.9,  0.05,  0.03,  0.02],
     [0.9,  0.05,  0.03,  0.02]],

    [[0.75, 0.15, 0.1,  0.001],
     [0.99, 0.05, 0.02, 0.001],
     [0.95, 0.03, 0.01, 0.01],
     [0.95, 0.03, 0.01, 0.01]],

    [[0.75, 0.15, 0.1,  0.00025],
     [0.93, 0.05, 0.02, 0.00025],
     [0.8,  0.1,  0.05, 0.05],
     [0.8,  0.1,  0.05, 0.05]]
])

# Female (not sex worker)
# Probability of transitions between female sexual behaviour groups
# 0 = Zero, 1= Any
sex_behaviour_trans_female_options = np.array([
    [[0.995, 0.005],
     [0.99,  0.01]],

    [[0.995, 0.005],
     [0.98,  0.02]],

    [[0.995, 0.005],
     [0.95,  0.05]],

    [[0.995, 0.005],
     [0.85,  0.15]],

    [[0.995, 0.005],
     [0.75,  0.25]],

    [[0.99, 0.01],
     [0.99, 0.01]],

    [[0.99, 0.01],
     [0.98, 0.02]],

    [[0.99, 0.01],
     [0.95, 0.05]],

    [[0.99, 0.01],
     [0.85, 0.15]],

    [[0.99, 0.01],
     [0.75, 0.25]],

    [[0.98, 0.02],
     [0.99, 0.01]],

    [[0.98, 0.02],
     [0.98, 0.02]],

    [[0.98, 0.02],
     [0.95, 0.05]],

    [[0.98, 0.02],
     [0.85, 0.15]],

    [[0.98, 0.02],
     [0.75, 0.25]]
])

# Sexual age mixing matrix
# groups 15-24, 35-34, 45-54, 55-65
# presumably no partners over 65?
sex_mixing_matrix_male_options = np.array([[[0.865, 0.11, 0.025, 0.0,  0.0],
                                            [0.47,  0.43, 0.10,  0.0,  0.0],
                                            [0.3,   0.5,  0.2,   0.0,  0.0],
                                            [0.43,  0.3,  0.23,  0.03, 0.01],
                                            [0.18,  0.18, 0.27,  0.27, 0.1]]])

sex_mixing_matrix_female_options = np.array([[[0.43, 0.34, 0.12, 0.1,  0.01],
                                              [0.09, 0.49, 0.3,  0.1,  0.02],
                                              [0.03, 0.25, 0.34, 0.25, 0.13],
                                              [0.0,  0.0,  0.05, 0.7,  0.25],
                                              [0.0,  0.0,  0.0,  0.1,  0.9]]])

# Doesn't seem to be interpolation between 10,15,20,25,30,35 on high?
short_term_partners_male_options = np.array([[stat.rv_discrete(values=([0], [1.0])),
                                              stat.rv_discrete(values=([1, 2, 3], [0.5, 0.3, 0.2])),
                                              stat.rv_discrete(values=([4, 5, 6, 7, 8, 9], [
                                                  0.35, 0.21, 0.17, 0.13, 0.09, 0.05])),
                                              stat.rv_discrete(values=([10, 15, 20, 25, 30, 35],
                                                                       [0.6, 0.2, 0.1, 0.05, 0.04, 0.01]))]])

# need to add another distribution to handle over 25s as well
short_term_partners_female_options = np.array([[stat.rv_discrete(values=([0], [1.0])),
                                                stat.rv_discrete(values=([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                         [0.3, 0.2, 0.15, 0.12, 0.09,
                                                                 0.06, 0.04, 0.02, 0.02]))]])
