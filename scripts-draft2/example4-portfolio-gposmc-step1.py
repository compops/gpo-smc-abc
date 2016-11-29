import pandas as pd
import numpy as np

from state import smc
from para import gpo_gpy
from models import hwsv_4parameters

import os
nStart = int(os.sys.argv[1])


def estimateLogVolatility(data):
    # Arrange the data structures
    sm = smc.smcSampler()
    gpo = gpo_gpy.stGPO()

    # Setup the system
    sys = hwsv_4parameters.ssm()
    sys.par = np.zeros((sys.nPar, 1))
    sys.xo = 0.0
    sys.T = len(data)
    sys.version = "standard"
    sys.transformY = "none"

    # Load data
    sys.generateData()
    sys.y = np.array(data).reshape((sys.T, 1))

    # Setup the parameters
    th = hwsv_4parameters.ssm()
    th.nParInference = 3
    th.copyData(sys)
    th.version = "standard"
    th.transformY = "none"

    # Setup the GPO algorithm
    gpo.verbose = True

    gpo.initPar = np.array([0.20, 0.95, 0.14])
    gpo.upperBounds = np.array([2.00, 1.00, 1.00])
    gpo.lowerBounds = np.array([-2.00, 0.80, 0.05])

    gpo.preIter = 50
    gpo.maxIter = 150

    gpo.jitteringCovariance = 0.01 * np.diag(np.ones(th.nParInference))
    gpo.preSamplingMethod = "latinHyperCube"

    gpo.EstimateHyperparametersInterval = 25
    gpo.EstimateThHatEveryIteration = False
    gpo.EstimateHessianEveryIteration = False

    # Setup the SMC algorithm
    sm.filter = sm.bPF
    sm.nPart = 1000
    sm.genInitialState = True

    # Estimate parameters
    gpo.bayes(sm, sys, th)
    gpo.estimateHessian()

    # Estimate the log-volatility
    th.storeParameters(gpo.thhat, sys)
    sm.filter(th)

    return sm.xhatf[:, 0], gpo.thhat, np.diag(-gpo.invHessianEstimate)

##############################################################################
##############################################################################
##############################################################################

# Get the log-returns
log_returns = np.loadtxt(
    'data/30_industry_portfolios_marketweighted.txt', skiprows=1)[:, 1:]
T = 805  # log_returns.shape[0]
nAssets = log_returns.shape[1]

# Estimate the log-volatility
log_volatility = np.zeros((T, nAssets))
models = np.zeros((3, nAssets))
modelsVar = np.zeros((3, nAssets))

for ii in range(nStart * 5, (nStart + 1) * 5):
    log_volatility[:, ii], models[:, ii], modelsVar[
        :, ii] = estimateLogVolatility(log_returns[0:T, ii])

fileOut = pd.DataFrame(log_volatility)
fileOut.to_csv(
    'results/example4-portfolio-gposmc-step1-volatility' + str(nStart) + '.csv')

fileOut = pd.DataFrame(models)
fileOut.to_csv(
    'results/example4-portfolio-gposmc-step1-models' + str(nStart) + '.csv')

fileOut = pd.DataFrame(modelsVar)
fileOut.to_csv(
    'results/example4-portfolio-gposmc-step1-modelsVar' + str(nStart) + '.csv')
