##############################################################################
##############################################################################
# Estimating the volatility of synthetic data
# using a stochastic volatility (SV) model with Gaussian log-returns.
#
# The SV model is inferred using the SPSA algorithm.
#
# For more details, see https://github.com/compops/gpo-abc2015
#
# (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################


import numpy as np
import matplotlib.pylab as plt

from state import smc
from para import ml_spsa
from models import hwsv_4parameters

from misc.portfolio import ensure_dir

##############################################################################
# Arrange the data structures
##############################################################################
sm = smc.smcSampler()
ml = ml_spsa.stMLspsa()


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
th.copyData(sys)
th.version = "standard"


##############################################################################
# Setup the SMC algorithm
##############################################################################

sm.filter = sm.bPF
sm.nPart = 1000
sm.resampFactor = 2.0
sm.genInitialState = True
sm.xo = sys.xo
th.xo = sys.xo


##############################################################################
# Setup the SPSA algorithm
##############################################################################
ml.a = 0.001
ml.c = 0.30
ml.maxIter = 350
ml.initPar = np.array([0.50, 0.95, 0.50])


##############################################################################
# SPSA using the Particle filter
##############################################################################

# Set the seed for re-producibility
np.random.seed(87655678)

# Run the SPSA routine
ml.bayes(sm, sys, th)

# Write output for plotting
import pandas
out = np.hstack((ml.th, ml.ll))
out = out.transpose()

output_file = "results/example1/spsa_map_bPF_N1000_3par.csv"
ensure_dir(output_file)
pandas.DataFrame(out).to_csv(output_file)


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
