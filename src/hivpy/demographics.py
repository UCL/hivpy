import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.interpolate import interp1d

SexType = CategoricalDtype(["female", "male"])


# Default values
# Can wrap those in something that ensures they have a description?
FEMALE_RATIO = 0.52
USE_STEPWISE_AGES = False
INC_CAT = 1

class StepwiseAgeDistribution:
    #stepwise distributions
    stepwise_model1 = np.array([0.18,0.165,0.144,0.114,0.09,0.08,0.068,0.047,0.036,0.027,0.021,0.016,0.012])
    stepwise_model2 = np.array([0.15,0.13,0.12,0.11,0.1,0.09,0.08,0.065,0.048,0.04,0.03,0.021,0.016])
    stepwise_model3 = np.array([0.128,0.119,0.113,0.104,0.097,0.09,0.081,0.074,0.06,0.05,0.038,0.026,0.02])
    stepwise_boundaries = np.array([-68,-55,-45,-35,-25,-15,-5,5,15,25,35,45,55,65])

    def __init__(self, ages, probabilities):
        self.probability_list = probabilities
        self.age_boundaries = ages

    @classmethod
    def selectModel(cls, inc_cat):
        if(inc_cat == 1):
            return cls(cls.stepwise_boundaries, cls.stepwise_model1)
        elif(inc_cat == 2):
            return cls(cls.stepwise_boundaries, cls.stepwise_model1)
        else:
            return cls(cls.stepwise_boundaries, cls.stepwise_model1)

    #Generate cumulative probabilites from stepwise distribution
    def _genCumulativeProb(self):
        N = self.probability_list.size
        CP = np.array([0.0]*N)
        CP[0] = self.probability_list[0]
        for i in range(1, N):
            CP[i] = CP[i-1] + self.probability_list[i]
        return CP

    #Generate Age With Stepwise Distribution 
    #the size of ages should be one larger than the cumulative probabilty 
    #so that each bucket has an upper and lower bound.
    def genAges(self, N):
        cpd = self._genCumulativeProb()
        ageDist = zip(self.age_boundaries, cpd)
        p0 = 0.0
        rands = np.random.rand(N)
        ages_out = np.array([0.0]*N)
        for i in range(0,cpd.size):
            #in order to vectorise this method we create a mask of values
            #that need to change for each age group
            prob_mask = (p0 < rands) & (rands <= cpd[i])
            ages_out += (self.age_boundaries[i] + 
                        (rands - p0)/(cpd[i]-p0)*(self.age_boundaries[i+1]-self.age_boundaries[i]))*prob_mask
            p0 = cpd[i]

        return ages_out
    

class ContinuousAgeDistribution:
    #example distribution: linexp
    # y = (mx + x)*exp(A(x-B))
    # This fits all three example cases quite neatly
    # We only need the cumulative distribution i.e. the integral of the function
    def integratedLinexp(x, m, c, A, B):
        return np.exp(A*(x-B))*(m*x+c - m/A)/A

    #Example parameters
    modelParams1 = [-7.19e-4,5.39e-2,-8.10e-3,2.12e1]
    modelParams2 = [-1.03e-3,7.45e-2,-1.12e-3,8.47]
    modelParams3 = [-1.15e-3,8.47e-2,2.24e-3,2.49e1]

    def __init__(self, min_age, max_age, cpd):
        self.min_age = min_age
        self.max_age = max_age
        self.cpd = cpd

    @classmethod
    def selectModel(cls, inc_cat):
        if(inc_cat == 1):
            return cls(-68,65,lambda x: cls.integratedLinexp(x, *cls.modelParams1))
        elif(inc_cat == 2):
            return cls(-68,65,lambda x: cls.integratedLinexp(x, *cls.modelParams2))
        else:
            return cls(-68,65,lambda x: cls.integratedLinexp(x, *cls.modelParams3))

    #Generate ages using a (non-normalised) continuous cumulative probability distribution
    #Given an analytic PD, this should also be analytically defined
    def genAges(self, N):
        #Normalise distribution over given range 
        C = self.cpd(self.min_age)
        M = 1/(self.cpd(self.max_age)-self.cpd(self.min_age))
        NormDist = lambda x: M*(self.cpd(x) - C)
        
        #sample and invert the normalised distribution (in case analytic inverse in impractical)
        NormX = np.linspace(self.min_age,self.max_age,11)
        NormY = NormDist(NormX)
        NormY[0] = 0.0
        NormY[10] = 1.0
        NormInv = interp1d(NormY,NormX,kind='cubic')

        #generate N random numbers in (0,1) and convert to ages
        R = np.random.rand(N)
        Ages = NormInv(R)
        return Ages


class DemographicsModule:

    def __init__(self, **kwargs):
        params = {
            "female_ratio": FEMALE_RATIO
        }
        # allow setting some parameters explicitly
        # could be useful if we have another method for more complex initialization,
        # e.g. from a config file
        for param,  value in kwargs.items():
            assert param in params, f"{param} is not related to this module."
            params[param] = value
        self.params = params

    def initialize_sex(self, count):
        sex_distribution = (
            self.params['female_ratio'], 1 - self.params['female_ratio'])
        return np.random.choice(SexType.categories, count, sex_distribution)

    def initialise_age(self, count):
        if(self.params['use_stepwise_ages'] == True):
            age_distribution = StepwiseAgeDistribution.selectModel(self.params['inc_cat'])
        else:
            age_distribution = ContinuousAgeDistribution.selectModel(self.params['inc_cat'])
        
        return age_distribution.genAges(count)