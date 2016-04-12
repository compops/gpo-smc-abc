##############################################################################
##############################################################################
# Parameter inference in stochastic volatility model
# using qPMH2
#
# (c) 2015 Johan Dahlin
# johan.dahlin (at) liu.se
#
##############################################################################
##############################################################################

import numpy            as np
import matplotlib.pylab as plt

from   state   import smc
from   para    import pmh
from   models  import hwsv_4parameters


##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
pmh              = pmh.stPMH();


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
sys.version       = "tanhexp"


##############################################################################
# Generate data
##############################################################################
sys.generateData(fileName='data/hwsv_4parameters_syntheticT500.csv',order="xy");


##############################################################################
##############################################################################
# Setup the parameters
th               = hwsv_4parameters.ssm()
th.nParInference = 3
th.version       = "tanhexp"
th.copyData(sys);


##############################################################################
# Setup the SMC algorithm
##############################################################################

sm.filter          = sm.bPF;
sm.smoother        = sm.flPS;
sm.fixedLag        = 12;
sm.genInitialState = True;

sm.nPart           = 2000;
sm.resampFactor    = 2.5;
sm.xo              = sys.xo;
th.xo              = sys.xo;

##############################################################################
# Setup the PMH algorithm
##############################################################################

pmh.dataset              = 0;

pmh.nIter                = 30000;
pmh.nBurnIn              = 5000;
pmh.initPar              = ( 0.10, 0.95, 0.12);

pmh.stepSize             = 1.0;
pmh.epsilon              = 400;
pmh.memoryLength         = 20;
pmh.makeHessianPSDmethod = "hybrid"
pmh.PSDmethodhybridSamps = 2500;

##############################################################################
# Run the qPMH2 sampler
##############################################################################

# Set seed for re-prodcibility
np.random.seed( 87655678 );

# Run the quasi-Newton PMH2 sampler
pmh.runSampler(sm, sys, th, "qPMH2");

# Write the results to file
pmh.writeToFile();

# np.mean( pmh.tho[pmh.nBurnIn:pmh.nIter,:], axis=0 )
# array([ 0.21207816,  0.87635889,  0.25195264])

##############################################################################
# End of file
##############################################################################
