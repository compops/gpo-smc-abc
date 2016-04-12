##############################################################################
##############################################################################
# Parameter inference in stochastic volatility model
# using GPO
#
# (c) 2015 Johan Dahlin
# johan.dahlin (at) liu.se
#
##############################################################################
##############################################################################

import numpy            as np
import matplotlib.pylab as plt

from   state   import smc
from   para    import gpo_gpy
from   models  import hwsv_4parameters


##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
gpo              = gpo_gpy.stGPO();


##############################################################################
# Setup the system
##############################################################################
sys               = hwsv_4parameters.ssm()
sys.par           = np.zeros((sys.nPar,1))
sys.par[0]        =  0.20;
sys.par[1]        =  0.96;
sys.par[2]        =  0.15;
sys.par[3]        =  0.00;
sys.T             =  500;
sys.xo            =  0.0;
sys.version       = "standard"


##############################################################################
# Generate data
##############################################################################
sys.generateData(fileName='data/hwsv_4parameters_syntheticT500.csv',order="xy");


##############################################################################
# Setup the parameters
##############################################################################
th               = hwsv_4parameters.ssm()
th.nParInference = 3;
th.copyData(sys);
th.version       = "standard"


##############################################################################
# Setup the GPO algorithm
##############################################################################

gpo.verbose                         = True;

gpo.initPar                         = np.array([ 0.20, 0.95, 0.14 ])
gpo.upperBounds                     = np.array([ 1.00, 1.00, 1.00 ]);
gpo.lowerBounds                     = np.array([-1.00, 0.00, 0.00 ]);

#gpo.maxIter                        = 650;
gpo.maxIter                         = 250;
gpo.preIter                         = 50;

gpo.jitteringCovariance             = 0.01 * np.diag(np.ones(th.nParInference));
gpo.preSamplingMethod               = "latinHyperCube";

gpo.EstimateHyperparametersInterval = 50;
gpo.EstimateThHatEveryIteration     = True;
gpo.EstimateHessianEveryIteration   = True;


##############################################################################
# Setup the SMC algorithm
##############################################################################

sm.filter          = sm.bPF;
sm.nPart           = 1000;
sm.resampFactor    = 0.5;
sm.genInitialState = True;
sm.xo              = sys.xo;
th.xo              = sys.xo;


##############################################################################
# GPO using the Particle filter
##############################################################################

# Set the seed for re-producibility
np.random.seed( 87655678 );

# Run the GPO routine
gpo.bayes(sm, sys, th);

# Write output
gpo.writeToFile(sm,fileOutName='results/synthetic-svmodel/gposmc_map_bPF_N1000_3par.csv')

# Estimate inverse Hessian and print it to screen
gpo.estimateHessian()
gpo.thhat
gpo.invHessianEstimate
#
# array([ 0.23045267,  0.87037037,  0.24897119])
#
# array([[  8.52169932e-03,   5.12490723e-05,  -1.68267781e-03],
#        [  5.12490723e-05,   1.61496991e-03,  -1.20550267e-03],
#        [ -1.68267781e-03,  -1.20550267e-03,   3.59943216e-03]])

# Plot marginals to check for convergence
gpo.plotPredictiveMarginals(matrixPlotSide=(1,3))


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
