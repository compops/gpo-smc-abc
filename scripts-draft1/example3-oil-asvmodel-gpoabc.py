##############################################################################
##############################################################################
# Parameter inference in stochastic volatility model
# using GPO-ABC
#
# (c) 2015 Johan Dahlin
# johan.dahlin (at) liu.se
#
##############################################################################
##############################################################################


import numpy   as np
import pandas
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
sys.par[1]        =  0.98;
sys.par[2]        =  0.15;
sys.par[3]        =  1.80;
sys.T             =  698;
sys.xo            =  0.0;
sys.version       = "standard"
sys.transformY    = "arctan"

##############################################################################
# Generate data
##############################################################################

gpo.dataset = 'brent';
#gpo.dataset = 'dubai';
#gpo.dataset = 'maya';
#gpo.dataset = 'wti';

sys.generateData(fileName="data/gpo_joe2015/"+str(gpo.dataset)+"_19970110_20100604.csv",order="y");
sys.y          = sys.y;
sys.ynoiseless = sys.ynoiseless;

##############################################################################
# Setup the parameters
##############################################################################
th               = hwsvalpha_4parameters.ssm()
th.version       = "standard"
th.transformY    = "arctan"
th.nParInference = 4;
th.copyData(sys);

##############################################################################
# Setup the GPO algorithm
##############################################################################

gpo.verbose                         = True;

gpo.initPar                         = np.array([ 0.50, 0.95, 0.20, 1.90 ])
gpo.upperBounds                     = np.array([ 5.00, 1.00, 1.00, 2.00 ]);
gpo.lowerBounds                     = np.array([ 0.00, 0.00, 0.10, 1.00 ]);

gpo.maxIter                         = 250;
gpo.preIter                         = 100;

gpo.jitteringCovariance             = 0.01 * np.diag(np.ones(th.nParInference));
gpo.preSamplingMethod               = "latinHyperCube";

gpo.EstimateHyperparametersInterval = 50;
gpo.EstimateThHatEveryIteration     = False;
gpo.EstimateHessianEveryIteration   = False;


##############################################################################
# Setup the SMC algorithm
##############################################################################

sm.filter          = sm.bPFabc;
sm.nPart           = 2500;
sm.resampFactor    = 2.0;

sm.weightdist      = "gaussian"
sm.rejectionSMC    = False;
sm.adaptTolLevel   = False;
sm.propAlive       = 0.00;
sm.tolLevel        = 0.10;

sm.genInitialState = True;

##############################################################################
# GPO-ABC using the Particle filter
##############################################################################

# Set the seed for re-producibility
np.random.seed( 87655678 );

# Add noise to data for noisy ABC
th.makeNoisy(sm);

# Run the GPO routine
gpo.bayes(sm, sys, th);

# Write output
gpo.writeToFile(sm,fileOutName='results/oil-asvmodel/gpoabc_map_bPF_N2000_3par_dataset' + str(gpo.dataset) + '.csv')

# Estimate inverse Hessian and print it to screen
gpo.estimateHessian()

# Write to file
fileOut = pandas.DataFrame(gpo.thhat);
fileOut.to_csv('results/oil-asvmodel/thhat-' + str(gpo.dataset) + '.csv');

fileOut = pandas.DataFrame(gpo.invHessianEstimate);
fileOut.to_csv('results/oil-asvmodel/h-' + str(gpo.dataset) + '.csv');

# Plot marginals to check for convergence
gpo.plotPredictiveMarginals(matrixPlotSide=(2,2))

##############################################################################
# # Run state estimation using particle smoother
##############################################################################

th.storeParameters( gpo.thhat, sys)
sm.calcGradientFlag = False
sm.calcHessianFlag  = False
sm.nPaths           = 50;
sm.nPathsLimit      = 10;
sm.ffbsiPS(th)

# Plot the state estimate
figure(1); plot(sys.y); plot(sm.xhats)

# Write state estimate to file
sm.writeToFile(fileOutName='results/oil-asvmodel/stateest-' + str(gpo.dataset) + '.csv')

np.round( gpo.thhat, 2 )
np.round( np.sqrt(np.abs(np.diag(gpo.invHessianEstimate))), 2)

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################


import numdifftools     as nd
hes  = nd.Hessian( gpo.evaluateSurrogate )
foo  = -hes( gpo.thhat )
invHessianEstimate = np.linalg.pinv( foo )
