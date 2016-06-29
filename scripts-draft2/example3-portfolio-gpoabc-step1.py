import numpy   as np
from   state   import smc
from   para    import gpo_gpy
from   models  import hwsvalpha_4parameters

import os
nStart = int( os.sys.argv[1] );

def estimateLogVolatility ( data ):
    # Arrange the data structures
    sm                = smc.smcSampler();
    gpo               = gpo_gpy.stGPO();
    
    # Setup the system
    sys               = hwsvalpha_4parameters.ssm()
    sys.par           = np.zeros((sys.nPar,1))
    sys.xo            = 0.0
    sys.T             = len(data)
    sys.version       = "standard"
    sys.transformY    = "none"
    
    # Load data
    sys.generateData()
    sys.y          = np.array( data ).reshape((sys.T,1))
    sys.ynoiseless = np.array( data ).reshape((sys.T,1))
   
    # Setup the parameters
    th               = hwsvalpha_4parameters.ssm()
    th.nParInference = 4
    th.copyData(sys)
    th.version       = "standard"
    th.transformY    = "arctan"
    
    # Setup the GPO algorithm
    gpo.verbose                         = True
    
    gpo.initPar                         = np.array([ 0.20, 0.95, 0.14,  1.90 ])
    gpo.upperBounds                     = np.array([ 2.00, 1.00, 1.00,  2.00 ])
    gpo.lowerBounds                     = np.array([-2.00, 0.80, 0.05,  1.00 ])
    
    gpo.preIter                         = 50    
    gpo.maxIter                         = 150    
    
    gpo.jitteringCovariance             = 0.01 * np.diag(np.ones(th.nParInference))
    gpo.preSamplingMethod               = "latinHyperCube"
    
    gpo.EstimateHyperparametersInterval = 25
    gpo.EstimateThHatEveryIteration     = False
    gpo.EstimateHessianEveryIteration   = False
    
    # Setup the SMC algorithm
    sm.filter                           = sm.bPFabc
    
    sm.nPart                            = 5000
    sm.genInitialState                  = True
    sm.weightdist                       = "gaussian"
    sm.tolLevel                         = 0.10
    
    # Add noise to data for noisy ABC
    th.makeNoisy(sm)
    
    # Estimate parameters
    gpo.bayes(sm, sys, th)
    gpo.estimateHessian()
    
    # Estimate the log-volatility
    th.storeParameters( gpo.thhat, sys)
    sm.filter(th)    
    
    return sm.xhatf[:,0], gpo.thhat, np.diag(-gpo.invHessianEstimate)

##############################################################################
##############################################################################
##############################################################################

# Get the log-returns
log_returns    = np.loadtxt('data/30_industry_portfolios_marketweighted.txt',skiprows=1)[:,1:]
T              = 805 #log_returns.shape[0]
nAssets        = log_returns.shape[1]

# Estimate the log-volatility
log_volatility          = np.zeros((T,nAssets))
models                  = np.zeros((4,nAssets))
modelsVar               = np.zeros((4,nAssets))

for ii in range(nStart*5,(nStart+1)*5):
    log_volatility[:,ii], models[:,ii], modelsVar[:,ii] = estimateLogVolatility( log_returns[0:T,ii] )

import pandas
fileOut = pandas.DataFrame(log_volatility)
fileOut.to_csv('results/example3-portfolio-gpoabc-step1-volatility' + str(nStart) + '.csv')

fileOut = pandas.DataFrame(models)
fileOut.to_csv('results/example3-portfolio-gpoabc-step1-models' + str(nStart) + '.csv')

fileOut = pandas.DataFrame(modelsVar)
fileOut.to_csv('results/example3-portfolio-gpoabc-step1-modelsVar' + str(nStart) + '.csv')
