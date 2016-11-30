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
sys.par[1]        =  0.98;
sys.par[2]        =  0.15;
sys.par[3]        =  1.80;
sys.T             =  399;
sys.xo            =  0.0;
sys.version       = "standard"
sys.transformY    = "arctan"

##############################################################################
# Generate data
##############################################################################
sys.generateData();

d              = Quandl.get("CHRIS/ICE_KC2", trim_start="2013-06-01", trim_end="2015-01-01")
logReturns     = 100 * np.diff(np.log(d['Settle']));
logReturns     = logReturns[~np.isnan(logReturns)];
sys.y          = np.matrix(logReturns).reshape((sys.T,1))
sys.ynoiseless = np.matrix(logReturns).reshape((sys.T,1))

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

gpo.initPar                         = np.array([ 0.50, 0.95, 0.14, 1.90 ])
gpo.upperBounds                     = np.array([ 1.00, 1.00, 0.70, 2.00 ]);
gpo.lowerBounds                     = np.array([ 0.00, 0.00, 0.00, 1.00 ]);

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
sm.nPart           = 5000;
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
# GPO-ABC using the Particle filter
##############################################################################

nRuns    = 10;

Tthhat   = np.zeros( (nRuns,th.nParInference) )
Thessian = np.zeros( (nRuns,th.nParInference,th.nParInference) )

for ii in range(nRuns):

    # Set the seed for re-producibility
    np.random.seed( 87655678 + ii);

    # Add noise to data for noisy ABC
    th.makeNoisy(sm);

    # Run the GPO routine
    gpo.bayes(sm, sys, th);

    # Write output
    gpo.writeToFile(sm,fileOutName='results/coffee-asvmodel/gpoabc_map_bPF_N2000_3par_run' + str(ii) + '.csv')

    # Estimate inverse Hessian and print it to screen
    gpo.estimateHessian()
    Tthhat[ii,:]     = gpo.thhat;
    Thessian[ii,:,:] = gpo.invHessianEstimate


import pandas
columnlabels = [None]*(th.nParInference)
columnlabels[0] = 'th0';
columnlabels[1] = 'th1';
columnlabels[2] = 'th2';
columnlabels[3] = 'th3';

fileOut = pandas.DataFrame(Thessian,columns=columnlabels);
fileOut.to_csv('example2-coffee-asvmodel-gpoabc-tthat.csv');

for ii in range(nRuns):
    fileOut = pandas.DataFrame(Thessian[ii,:,:]);
    fileOut.to_csv('example2-coffee-asvmodel-gpoabc-thessian-' + str(ii) + '.csv');


#
#>>> gpo.thhat
#array([ 0.27777778,  0.91975309,  0.27222222,  1.45061728])
#>>> gpo.invHessianEstimate
#array([[ 0.01459183, -0.00029639,  0.00088382,  0.00427282],
#       [-0.00029639,  0.00150004, -0.00198543,  0.00022116],
#       [ 0.00088382, -0.00198543,  0.00857372,  0.00161204],
#       [ 0.00427282,  0.00022116,  0.00161204,  0.00925242]])

# Plot marginals to check for convergence
# gpo.plotPredictiveMarginals(matrixPlotSide=(2,2))

##############################################################################
# # Run state estimation using particle smoother
##############################################################################

th.storeParameters( (0.27777778,  0.91975309,  0.27222222,  1.45061728),sys)
sm.calcGradientFlag = False
sm.calcHessianFlag  = False
sm.nPaths           = 50;
sm.nPathsLimit      = 10;
sm.ffbsiPS(th)

# Plot the state estimate
plt.figure(1); 
plt.plot(sys.y); 
plt.plot(sm.xhats)

# Write state estimate to file
sm.writeToFile()


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################