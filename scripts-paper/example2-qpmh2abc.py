##############################################################################
##############################################################################
# Estimating the volatility of coffee futures
# using a stochastic volatility (SV) model with alpha-stable log-returns.
#
# The SV model is inferred using the qPMH2 algorithm.
#
# For more details, see https://github.com/compops/gpo-abc2015
#
# (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

import Quandl
import numpy as np
import matplotlib.pylab as plt

from state import smc
from para import pmh
from models import hwsvalpha_4parameters


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

d = Quandl.get("CHRIS/ICE_KC2", trim_start="2013-06-01", trim_end="2015-01-01")
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

sm.filter = sm.bPFabc
sm.smoother = sm.flPS
sm.fixedLag = 12
sm.genInitialState = True
sm.resampFactor = 2.0

sm.weightdist = "gaussian"
sm.rejectionSMC = False
sm.adaptTolLevel = False
sm.propAlive = 0.00
sm.tolLevel = 0.10

sm.nPart = 5000
sm.xo = sys.xo
th.xo = sys.xo


##############################################################################
# Setup the PMH algorithm
##############################################################################

pmh.dataset = 0

pmh.nIter = 30000
pmh.nBurnIn = 5000

pmh.initPar = (0.27777778,  0.91975309,  0.27222222,  1.45061728)
pmh.stepSize = 1.0

pmh.epsilon = 400
pmh.memoryLength = 40
pmh.makeHessianPSDmethod = "hybrid"
pmh.PSDmethodhybridSamps = 2500


##############################################################################
# Run the qPMH2 sampler
##############################################################################

# Set the seed for re-producibility
np.random.seed(87655678)

# Add noise to data for noisy ABC
th.makeNoisy(sm)

# Run the quasi-Newton PMH2 sampler
pmh.runSampler(sm, sys, th, "qPMH2")

# Write the results to file
pmh.writeToFile()


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
