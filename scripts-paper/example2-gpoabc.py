##############################################################################
##############################################################################
# Estimating the volatility of coffee futures
# using a stochastic volatility (SV) model with alpha-stable log-returns.
#
# The SV model is inferred using the GPO-SMC-ABC algorithm.
#
# For more details, see https://github.com/compops/gpo-abc2015
#
# (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################


import sys
sys.path.insert(0, '/media/sf_home/src/gpo-abc2015')

# Setup files
output_file = 'results/example2/example2-gpoabc'

# Load packages and helpers
import quandl
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from state import smc
from para import gpo_gpy
from models import hwsvalpha_4parameters

from misc.portfolio import ensure_dir

# Set the seed for re-producibility
np.random.seed(87655678)


##############################################################################
# Arrange the data structures
##############################################################################
sm = smc.smcSampler()
gpo = gpo_gpy.stGPO()

##############################################################################
# Setup the system
##############################################################################
sys = hwsvalpha_4parameters.ssm()
sys.par = np.zeros((sys.nPar, 1))

sys.par[0] = 0.20
sys.par[1] = 0.98
sys.par[2] = 0.15
sys.par[3] = 1.80

sys.T = 399
sys.xo = 0.0
sys.version = "standard"
sys.transformY = "arctan"

##############################################################################
# Generate data
##############################################################################
sys.generateData()

d = quandl.get("CHRIS/ICE_KC2", trim_start="2013-06-01", trim_end="2015-01-01")
logReturns = 100 * np.diff(np.log(d['Settle']))
logReturns = logReturns[~np.isnan(logReturns)]

sys.y = np.matrix(logReturns).reshape((sys.T, 1))
sys.ynoiseless = np.matrix(logReturns).reshape((sys.T, 1))

##############################################################################
# Setup the parameters
##############################################################################
th = hwsvalpha_4parameters.ssm()
th.version = "standard"
th.transformY = "arctan"
th.nParInference = 4
th.copyData(sys)

##############################################################################
# Setup the GPO algorithm
##############################################################################

settings = {'gpo_initPar':     np.array([0.00, 0.95, 0.50, 1.80]),
            'gpo_upperBounds': np.array([5.00, 0.99, 1.00, 2.00]),
            'gpo_lowerBounds': np.array([0.00, 0.00, 0.10, 1.20]),
            'gpo_estHypParInterval': 25,
            'gpo_preIter': 50,
            'gpo_maxIter': 150,
            'smc_weightdist': "gaussian",
            'smc_tolLevel': 0.10,
            'smc_nPart': 2000
            }

gpo.initPar = settings['gpo_initPar'][0:th.nParInference]
gpo.upperBounds = settings['gpo_upperBounds'][0:th.nParInference]
gpo.lowerBounds = settings['gpo_lowerBounds'][0:th.nParInference]
gpo.maxIter = settings['gpo_maxIter']
gpo.preIter = settings['gpo_preIter']
gpo.EstimateHyperparametersInterval = settings['gpo_estHypParInterval']

gpo.verbose = True
gpo.jitteringCovariance = 0.01 * np.diag(np.ones(th.nParInference))
gpo.preSamplingMethod = "latinHyperCube"

gpo.EstimateThHatEveryIteration = False
gpo.EstimateHessianEveryIteration = False


##############################################################################
# Setup the SMC algorithm
##############################################################################

sm.filter = sm.bPFabc
sm.nPart = settings['smc_nPart']
sm.weightdist = settings['smc_weightdist']

sm.rejectionSMC = False
sm.adaptTolLevel = False
sm.propAlive = 0.00
sm.tolLevel = settings['smc_tolLevel']

sm.genInitialState = True
sm.xo = sys.xo
th.xo = sys.xo

##############################################################################
# GPO-ABC using the Particle filter
##############################################################################

nRuns = 10

Txhats = np.zeros((nRuns, sys.T))
Tthhat = np.zeros((nRuns, th.nParInference))
Thessian = np.zeros((nRuns, th.nParInference, th.nParInference))

for ii in range(nRuns):

    # Add noise to data for noisy ABC
    th.makeNoisy(sm)

    # Run the GPO routine
    gpo.bayes(sm, sys, th)

    # Estimate inverse Hessian and print it to screen
    gpo.estimateHessian()
    Tthhat[ii, :] = gpo.thhat
    Thessian[ii, :, :] = gpo.invHessianEstimate

    # Run state estimation using particle smoother
    th.storeParameters(gpo.thhat, sys)
    sm.calcGradientFlag = False
    sm.calcHessianFlag = False

    sm.nPaths = 50
    sm.nPathsLimit = 10
    sm.ffbsiPS(th)
    Txhats[ii, :] = sm.xhats


#############################################################################
# Write results to file
##############################################################################

ensure_dir(output_file + '-thhat.csv')

# Model parameters
fileOut = pd.DataFrame(gpo.thhat)
fileOut.to_csv(output_file + '-model.csv')

# Inverse Hessian estimate
for ii in range(nRuns):
    fileOut = pd.DataFrame(gpo.invHessianEstimate[ii, :, :])
    fileOut.to_csv(output_file + '-modelvar-' + str(ii) + '.csv')

# Log-volatility
for ii in range(nRuns):
    fileOut = pd.DataFrame(Txhats[ii, :])
    fileOut.to_csv(output_file + '-volatility-' + str(ii) + '.csv')

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
