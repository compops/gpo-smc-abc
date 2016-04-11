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
sys.par[0]        =  1.00;
sys.par[1]        =  0.98;
sys.par[2]        =  0.15;
sys.par[3]        =  0.00;
sys.T             =  698;
sys.xo            =  0.0;
sys.version       = "standard"
#sys.transformY    = "arctan"

##############################################################################
# Generate data
##############################################################################

gpo.dataset = 'brent';
#gpo.dataset = 'dubai';
#gpo.dataset = 'maya';

sys.generateData(fileName="data/gpo_joe2015/"+str(gpo.dataset)+"_19970110_20100604.csv",order="y");
sys.y          = sys.y;

##############################################################################
# Setup the parameters
##############################################################################
th               = hwsv_4parameters.ssm()
th.version       = "standard"
#th.transformY    = "arctan"
th.nParInference = 3;
th.copyData(sys);

##############################################################################
# Setup the GPO algorithm
##############################################################################

gpo.verbose                         = True;

gpo.initPar                         = np.array([ 0.20, 0.95, 0.14 ])
gpo.upperBounds                     = np.array([ 2.00, 0.99, 0.50 ]);
gpo.lowerBounds                     = np.array([ 0.00, 0.80, 0.10 ]);

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

sm.filter          = sm.bPF;
sm.nPart           = 1000;
sm.resampFactor    = 2.0;

sm.genInitialState = True;


##############################################################################
# GPO-SMC using the Particle filter
##############################################################################

# Set the seed for re-producibility
np.random.seed( 87655678 );

# Run the GPO routine
gpo.bayes(sm, sys, th);

# Write output
#gpo.writeToFile(sm,fileOutName='results/oil-svmodel/gpo_map_bPF_N2000_3par_dataset' + str(gpo.dataset) + '.csv')

# Estimate inverse Hessian and print it to screen
gpo.estimateHessian()

# Write to file
fileOut = pandas.DataFrame(gpo.thhat);
fileOut.to_csv('results/oil-svmodel/thhat-' + str(gpo.dataset) + '.csv');

fileOut = pandas.DataFrame(gpo.invHessianEstimate);
fileOut.to_csv('results/oil-svmodel/h-' + str(gpo.dataset) + '.csv');

# Plot marginals to check for convergence
#gpo.plotPredictiveMarginals(matrixPlotSide=(1,3))

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

figure(2);
subplot(2,1,1); plot(sys.y * np.exp(-0.5*sm.xhats), 'k.' )
subplot(2,1,2); hist(sys.y * np.exp(-0.5*sm.xhats), bins=40 )

figure(3); plot(sys.y,'k.'); plot(1.96 * np.exp(0.5*sm.xhats),'r'); plot(-1.96 * np.exp(0.5*sm.xhats),'r')

# Write state estimate to file
sm.writeToFile(fileOutName='results/oil-svmodel/stateest-' + str(gpo.dataset) + '.csv')

np.round( gpo.thhat, 2 )
np.round( np.sqrt(np.abs(np.diag(gpo.invHessianEstimate))), 2)

## Brent
#>>> >>> >>> >>> np.round( gpo.thhat, 2 )
#array([ 1.78,  0.99,  0.13])
#>>> np.round( np.sqrt(np.abs(np.diag(gpo.invHessianEstimate))), 2)
#array([ 0.13,  0.  ,  0.02])
#>>> gpo.invHessianEstimate
#array([[  1.63751240e-02,   6.58704551e-04,  -2.55263863e-03],
#       [  6.58704551e-04,  -1.31663662e-05,  -6.07390454e-06],
#       [ -2.55263863e-03,  -6.07390454e-06,   2.98021775e-04]])

## Dubai
#>>> np.round( gpo.thhat, 2 )
#array([ 1.63,  0.99,  0.15])
#>>> np.round( np.sqrt(np.abs(np.diag(gpo.invHessianEstimate))), 2)
#array([ 0.07,  0.  ,  0.01])
#>>> gpo.invHessianEstimate
#array([[  5.38859571e-03,   7.50447724e-04,  -1.73607832e-03],
#       [  7.50447724e-04,  -1.70056249e-05,   7.43068858e-06],
#       [ -1.73607832e-03,   7.43068858e-06,   2.19701235e-04]])

## Maya
#>>> np.round( gpo.thhat, 2 )
#array([ 1.86,  0.99,  0.14])
#>>> np.round( np.sqrt(np.abs(np.diag(gpo.invHessianEstimate))), 2)
#array([ 0.17,  0.  ,  0.03])
#>>> gpo.invHessianEstimate
#array([[  2.95511015e-02,   1.89422866e-03,  -5.34426253e-03],
#       [  1.89422866e-03,   1.42600215e-05,  -1.45457491e-04],
#       [ -5.34426253e-03,  -1.45457491e-04,   7.03077315e-04]])

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################