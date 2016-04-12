##############################################################################
# Minimal working example
# Parameter inference in a stochastic volatility model
# using Gaussian process optimisation (GPO) with
# sequential Monte Carlo (SMC)
#
# (c) Johan Dahlin 2016 ( johan.dahlin (at) liu.se )
##############################################################################

import numpy   as np
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
sys.par[1]        =  0.90;
sys.par[2]        =  0.15;
sys.par[3]        =  -0.70;
sys.T             =  1000;
sys.xo            =  0.0;
sys.version       = "standard"


##############################################################################
# Generate data
##############################################################################
sys.generateData(u=np.zeros(sys.T));


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
gpo.upperBounds                     = np.array([ 0.50, 1.00, 0.50 ]);
gpo.lowerBounds                     = np.array([-0.50, 0.80, 0.05 ]);

gpo.maxIter                         = 150;
gpo.preIter                         = 50;

gpo.jitteringCovariance             = 0.01 * np.diag(np.ones(th.nParInference));
gpo.preSamplingMethod               = "latinHyperCube";

gpo.EstimateHyperparametersInterval = 50;
gpo.EstimateThHatEveryIteration     = False;
gpo.EstimateHessianEveryIteration   = False;


##############################################################################
# Setup the SMC algorithm
##############################################################################

sm.filter          = sm.bPF;
sm.nPart           = 100;
sm.resampFactor    = 2.0;
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

# Parameter estimates
gpo.thhat
# array([ 0.2345679 ,  0.91975309,  0.14166667])

# Estimate of inverse Hessian
gpo.estimateHessian()

# Compute half-length of 95% confidence intervals
1.96*np.sqrt(np.diag(gpo.invHessianEstimate))
# array([ 0.12445807,  0.04638067,  0.05792237])

# Plot marginals to check for convergence
gpo.plotPredictiveMarginals(matrixPlotSide=(1,3))

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
