##############################################################################
##############################################################################
# Estimating the volatility of coffee futures
# using a stochastic volatility (SV) model with alpha-stable log-returns.
#
# The SV model is inferred using the PMH algorithm.
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
output_file = 'results/example2/example2-pmhabc'

# Load packages and helpers
import quandl
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from state import smc
from para import pmh
from models import hwsvalpha_4parameters

from misc.portfolio import ensure_dir

# Set the seed for re-producibility
np.random.seed(87655678)


##############################################################################
# Arrange the data structures
##############################################################################
sm = smc.smcSampler()
pmh = pmh.stPMH()


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
# Setup the SMC algorithm
##############################################################################

# Use the particle filter to estimate the log-likelihood
sm.filter = sm.bPFabc

sm.nPart = 5000
sm.genInitialState = True

sm.weightdist = "gaussian"
sm.rejectionSMC = False
sm.adaptTolLevel = False
sm.propAlive = 0.00
sm.tolLevel = 0.10


##############################################################################
# Setup the PMH algorithm
##############################################################################

pmh.dataset = 0
pmh.nIter = 15000
pmh.nBurnIn = 5000

# Set initial parameters
pmh.initPar = (0.22,  0.93,  0.25,  1.55)

# Set the pre-conditioning matrix from initial run
pmh.invHessian = np.matrix([[0.0256918580, -0.0018746969,  0.001512602,  0.0004085695],
                            [-0.0018746969, 0.0014677388, -0.002594432, -0.0001351334],
                            [0.0015126023, -0.0025944317,  0.009365981,  0.0022698498],
                            [0.0004085695, -0.0001351334,  0.002269850,  0.0106446958]])

# Use only the diagonal, off-diagonal elements can be poorly estimated by GPO
pmh.invHessian = np.diag(np.diag(pmh.invHessian)[0:th.nParInference])


##############################################################################
# Run the PMH sampler
##############################################################################

# Add noise to data for noisy ABC
th.makeNoisy(sm)

# Settings for the PMH routine
pmh.stepSize = 2.562 / np.sqrt(th.nParInference)

# Run the pre-conditioned PMH0 sampler
pmh.runSampler(sm, sys, th, "pPMH0")

# Write the results to file
pmh.writeToFile(fileOutName=output_file + '-run.csv')
    

#############################################################################
# Write results to file
##############################################################################

ensure_dir(output_file + '.csv')

# Model parameters
fileOut = pd.DataFrame(np.mean(pmh.th[pmh.nBurnIn:pmh.nIter, :], axis=0))
fileOut.to_csv(output_file + '-model.csv')

# Inverse Hessian estimate
fileOut = pd.DataFrame(np.cov(pmh.th[pmh.nBurnIn:pmh.nIter, :]), rowvar=False)
fileOut.to_csv(output_file + '-modelvar.csv')


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
