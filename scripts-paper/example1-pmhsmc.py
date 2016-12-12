##############################################################################
##############################################################################
# Estimating the volatility of synthetic data
# using a stochastic volatility (SV) model with Gaussian log-returns.
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
output_file = 'results/example1/example1-pmhsmc'

# Load packages and helpers
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from state import smc
from para import pmh
from models import hwsv_4parameters
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
sys = hwsv_4parameters.ssm()
sys.par = np.zeros((sys.nPar, 1))

sys.par[0] = 0.20
sys.par[1] = 0.96
sys.par[2] = 0.15
sys.par[3] = 0.00

sys.T = 500
sys.xo = 0.0
sys.version = "standard"


##############################################################################
# Generate data
##############################################################################
sys.generateData(
    fileName='data/hwsv_4parameters_syntheticT500.csv', order="xy")


##############################################################################
# Setup the parameters
##############################################################################
th = hwsv_4parameters.ssm()
th.nParInference = 3
th.version = "standard"
th.copyData(sys)


##############################################################################
# Setup the SMC algorithm
##############################################################################
sm.filter = sm.bPF
sm.genInitialState = True

sm.nPart = 2000
sm.xo = sys.xo
th.xo = sys.xo


##############################################################################
# Setup the PMH algorithm
##############################################################################
pmh.dataset = 0
pmh.nIter = 15000
pmh.nBurnIn = 5000

# Set initial parameters
pmh.initPar = (0.10, 0.95, 0.12)

# Set the pre-conditioning matrix from initial run
pmh.invHessian = np.matrix([[ 0.0137448825, -0.0011175262,  0.0006854814],
                            [-0.0011175262,  0.0007471863, -0.0011258477],
                            [ 0.0006854814,  0.0011258477,  0.0037545209]])

# Use only the diagonal, off-diagonal elements can be poorly estimated by GPO
pmh.invHessian = np.diag(np.diag(pmh.invHessian)[0:th.nParInference])


##############################################################################
# Run the PMH sampler
##############################################################################

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
# End of file
##############################################################################
