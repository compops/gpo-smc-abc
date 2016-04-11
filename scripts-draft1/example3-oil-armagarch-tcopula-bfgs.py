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

import pandas         as pd
import numpy          as np
import numdifftools   as nd

from   misc.evtCDF    import *
from   state          import smc
from   para           import gpo_gpy
from   models.copula  import studentt
from   models.dists   import genPareto
from   models.models_dists import multiTSimulate
from scipy            import stats
from scipy            import optimize

##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
gpo              = gpo_gpy.stGPO();
gpa              = genPareto.dist();

##############################################################################
# Setup the system
##############################################################################
th                = studentt.copula()

th.nPar           = 4
th.par            = np.zeros((th.nPar,1))
th.T              = 698;
th.xo             = 0.0;
th.nParInference  = th.nPar;

##############################################################################
# Load data
##############################################################################

ehat = pd.read_csv('results/oil-armagarch/example3-oil-armagarch-residuals.csv', sep="," , header = False)

ehat1 = np.array( ehat['Brent'] )
ehat2 = np.array( ehat['Dubai'] )
ehat3 = np.array( ehat['Maya'] )

d1 = np.loadtxt('data/gpo_joe2015/brent_19970110_20100604.csv')
d2 = np.loadtxt('data/gpo_joe2015/dubai_19970110_20100604.csv')
d3 = np.loadtxt('data/gpo_joe2015/maya_19970110_20100604.csv')

##############################################################################
# Compute the smoothed residuals and approximate the inverse CDF
##############################################################################

( uhat1, thU1, thL1, sdU1, sdL1 ) = evtECDF( ehat1, gpa, ehat1, 0.10 )
( uhat2, thU2, thL2, sdU2, sdL2 ) = evtECDF( ehat2, gpa, ehat2, 0.10 )
( uhat3, thU3, thL3, sdU3, sdL3 ) = evtECDF( ehat3, gpa, ehat3, 0.10 )

#np.round( np.hstack( (thL1[0], sdL1[0], thL1[1], sdL1[1], thU1[0], sdU1[0], thU1[1], sdU1[1]) ), 2 )
#np.round( np.hstack( (thL2[0], sdL2[0], thL2[1], sdL2[1], thU2[0], sdU2[0], thU2[1], sdU2[1]) ), 2 )
#np.round( np.hstack( (thL3[0], sdL3[0], thL3[1], sdL3[1], thU3[0], sdU3[0], thU3[1], sdU3[1]) ), 2 )


#figure(1)
#subplot(2,3,1); plot(ehat1,uhat1,'k.'); xlabel("Brent"); ylabel("hybrid EDF");
#subplot(2,3,2); plot(ehat2,uhat2,'k.'); xlabel("Dubai"); ylabel("hybrid EDF");
#subplot(2,3,3); plot(ehat3,uhat3,'k.'); xlabel("Maya");  ylabel("hybrid EDF");
#
#subplot(2,3,4); plot(uhat1,uhat2,'k.'); xlabel("Brent"); ylabel("Dubai");
#subplot(2,3,5); plot(uhat1,uhat3,'k.'); xlabel("Brent"); ylabel("Maya");
#subplot(2,3,6); plot(uhat2,uhat3,'k.'); xlabel("Dubai"); ylabel("Maya");

# Write empricial DF to file
#fileOut = pd.DataFrame(uhat1); fileOut.to_csv('results/oil-asvmodel/uhat-brent.csv');
#fileOut = pd.DataFrame(uhat2); fileOut.to_csv('results/oil-asvmodel/uhat-dubai.csv');
#fileOut = pd.DataFrame(uhat3); fileOut.to_csv('results/oil-asvmodel/uhat-maya.csv');

# Write residuals to file
#fileOut = pd.DataFrame(ehat1); fileOut.to_csv('results/oil-asvmodel/ehat-brent.csv');
#fileOut = pd.DataFrame(ehat2); fileOut.to_csv('results/oil-asvmodel/ehat-dubai.csv');
#fileOut = pd.DataFrame(ehat3); fileOut.to_csv('results/oil-asvmodel/ehat-maya.csv');

th.uhat = np.vstack((uhat1,uhat2,uhat3)).transpose()

p = (th.uhat).shape[1];
n = (th.uhat).shape[0];

##############################################################################
# Kendall's tau for Student t copula
##############################################################################

rho10Est = 2.0/np.pi * np.arcsin( stats.kendalltau(th.uhat[:,1],th.uhat[:,0]) )[0]
rho20Est = 2.0/np.pi * np.arcsin( stats.kendalltau(th.uhat[:,2],th.uhat[:,0]) )[0]
rho21Est = 2.0/np.pi * np.arcsin( stats.kendalltau(th.uhat[:,2],th.uhat[:,1]) )[0]

##############################################################################
# Use BFGS to optimize the log-posterior
##############################################################################

res = optimize.fmin_l_bfgs_b(th.evaluateLogPosteriorBFGS, np.array([1.0,rho10Est,rho20Est,rho21Est]), approx_grad=1, bounds=( (0.1,50.0),(-0.90,0.90),(-0.90,0.90),(-0.90,0.90) ) )

# Estimate standard deviations by numerical differencation to obtain Hessian
hes  = nd.Hessian( th.evaluateLogPosteriorBFGS, delta=.00001 )
foo  = np.linalg.pinv( hes( res[0] ) )

# Store parameters and construct correlation matrix
th.storeParameters(res[0],th)
th.constructCorrelationMatrix( 3 )

# Estimate the Spearman correlation using Monte Carlo
bar = multiTSimulate(10000,np.zeros(p),th.P,th.par[0])
corrSpearman = ( stats.spearmanr(bar[:,0],bar[:,1])[0], stats.spearmanr(bar[:,0],bar[:,2])[0], stats.spearmanr(bar[:,1],bar[:,2])[0] )

print( np.round( res[0] ,2) )
print( np.round( np.sqrt( np.diag( np.abs( foo) ) ) ,2) )
print( np.round( th.P ,2) )
print( np.round(corrSpearman,2) )

#>>> print( np.round( res[0] ,2) )
#[ 4.29 -0.12  0.38  0.29]
#>>> print( np.round( np.sqrt( np.diag( np.abs( foo) ) ) ,2) )
#[ 0.65  0.03  0.02  0.02]
#>>> print( np.round( th.P ,2) )
#[[ 1.    0.76  0.87]
# [ 0.76  1.    0.79]
# [ 0.87  0.79  1.  ]]
#>>> print( np.round(corrSpearman,2) )
#[ 0.73  0.85  0.76]

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################