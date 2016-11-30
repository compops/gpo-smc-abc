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
sys.transformY    = "none"

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
th.transformY    = "none"
th.ynoiseless    = np.array(sys.y,copy=True)


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

sm.filter          = sm.bPFabc;
sm.nPart           = 2000;
sm.resampFactor    = 0.5;
sm.weightdist      = "gaussian"

sm.rejectionSMC    = False;
sm.adaptTolLevel   = False;
sm.propAlive       = 0.00;

#sm.tolLevel        = 0.10;
#sm.tolLevel        = 0.20;
#sm.tolLevel        = 0.30;
#sm.tolLevel        = 0.40;
sm.tolLevel        = 0.50;

sm.genInitialState = True;
sm.xo              = sys.xo;
th.xo              = sys.xo;

##############################################################################
# GPO using the Particle filter
##############################################################################

# Set the seed for re-producibility
np.random.seed( 87655678 );

# Add noise to data for noisy ABC
th.makeNoisy(sm);

# Run the GPO routine
gpo.bayes(sm, sys, th);

# Write output
gpo.writeToFile(sm,fileOutName='results/synthetic-svmodel/gpoabc_map_bPF_N1000_3par.csv')

# Estimate inverse Hessian and print it to screen
gpo.estimateHessian()
sm.tolLevel
gpo.thhat
gpo.invHessianEstimate
#
#>>> >>> 0.1
#>>> array([ 0.28806584,  0.86625514,  0.3600823 ])
#>>> array([[ 0.0145148 , -0.00110559, -0.00010567],
#       [-0.00110559,  0.0006584 , -0.00012965],
#       [-0.00010567, -0.00012965,  0.0005114 ]])
#
#>>> >>> 0.2
#>>> array([ 0.14814815,  0.89917695,  0.26131687])
#>>> array([[ 0.01002031,  0.00048493, -0.00201742],
#       [ 0.00048493,  0.00142153, -0.00147207],
#       [-0.00201742, -0.00147207,  0.00292853]])
#
#>>> >>> 0.3
#>>> array([ 0.18930041,  0.88271605,  0.25720165])
#>>> array([[ 0.0099959 ,  0.00011847, -0.00280843],
#       [ 0.00011847,  0.00207768, -0.00238605],
#       [-0.00280843, -0.00238605,  0.00574637]])
#
#>>> >>> 0.4
#>>> array([ 0.22222222,  0.88271605,  0.25308642])
#>>> array([[  1.17579934e-02,  -5.79787130e-05,  -2.25528565e-03],
#       [ -5.79787130e-05,   1.42609302e-03,  -1.42829007e-03],
#       [ -2.25528565e-03,  -1.42829007e-03,   4.48085561e-03]])
#
#>>> >>> 0.5
#>>> array([ 0.18930041,  0.88271605,  0.2654321 ])
#>>> array([[ 0.01077807, -0.00012377, -0.00203055],
#       [-0.00012377,  0.00144916, -0.00136923],
#       [-0.00203055, -0.00136923,  0.00441841]])


# Plot marginals to check for convergence
gpo.plotPredictiveMarginals(matrixPlotSide=(2,2))


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
