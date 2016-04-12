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

import Quandl
import numpy            as np
import matplotlib.pylab as plt

from   state   import smc
from   para    import gpo_gpy
from   models  import hwsvalpha_4parameters


##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
gpo              = gpo_gpy.stGPO();


##############################################################################
# Setup the system
##############################################################################
sys               = hwsvalpha_4parameters.ssm()
sys.par           = np.zeros((sys.nPar,1))
sys.par[0]        =  0.20;
sys.par[1]        =  0.90;
sys.par[2]        =  0.15;
sys.par[3]        =  1.80;
sys.T             =  357;
sys.xo            =  0.0;
sys.version       = "standard"
sys.transformY    = "arctan"


##############################################################################
# Download the data
##############################################################################
sys.generateData();

d     = Quandl.get("BAVERAGE/EUR", trim_start="2013-09-03", trim_end="2014-09-03")
sys.y = np.matrix(100 * np.diff(np.log(d['24h Average']))).reshape((sys.T,1))


##############################################################################
# Setup the parameters
##############################################################################
th               = hwsvalpha_4parameters.ssm()
th.nParInference = 4;
th.copyData(sys);

th.version       = "standard"
th.transformY    = "arctan"
th.ynoiseless    = np.array(sys.y,copy=True)


##############################################################################
# Setup the GPO algorithm
##############################################################################

gpo.verbose                         = True;

gpo.initPar                         = np.array([ 0.20, 0.95, 0.14, 1.8 ])
gpo.upperBounds                     = np.array([ 0.50, 1.00, 0.90, 2.0 ]);
gpo.lowerBounds                     = np.array([-0.50, 0.80, 0.05, 1.2 ]);

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

sm.filter          = sm.bPFabc;
sm.nPart           = 2000;
sm.resampFactor    = 2.0;
sm.weightdist      = "gaussian"

sm.rejectionSMC    = False;
sm.adaptTolLevel   = False;
sm.propAlive       = 0.00;
sm.tolLevel        = 0.10;

sm.genInitialState = True;
sm.xo              = sys.xo;
th.xo              = sys.xo;

##############################################################################
# GPO using the Particle filter with ABC
##############################################################################

# Set the seed for re-producibility
np.random.seed( 87655678 );

# Add noise to data for noisy ABC
th.makeNoisy(sm);

# Run the GPO routine
gpo.bayes(sm, sys, th);

# Parameter estimates
gpo.thhat
# array([ 0.14814815,  0.94938272,  0.62191358,  1.54074074])

# Estimate of inverse Hessian
gpo.estimateHessian()

# Compute half-length of 95% confidence intervals
1.96*np.sqrt(np.diag(gpo.invHessianEstimate))
# array([ 0.37539896,  0.03176862,  0.16280434,  0.1632567 ])

# Estimate the log-volatility and plot it
th.storeParameters( gpo.thhat, sys )
sm.bPFabc(th)

plt.figure(1);
plt.subplot(2,1,1)
plt.plot(sys.y); 
plt.xlabel("time"); 
plt.ylabel("Bitcoin log-returns")

plt.subplot(2,1,2)
plt.plot(sm.xhatf,'g'); 
plt.xlabel("time");
plt.ylabel("estimated log-volatility")

# Plot marginals to check for convergence
gpo.plotPredictiveMarginals(matrixPlotSide=(2,2))

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################