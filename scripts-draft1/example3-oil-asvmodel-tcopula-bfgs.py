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
import scipy          as sp

from   misc.evtCDF    import *
from   state          import smc
from   para           import gpo_gpy

from   models.copula  import studentt
from   models.dists   import genPareto
from   models.models_dists import multiTSimulate

from   scipy          import stats
from   scipy          import optimize

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

x1 = pd.read_csv('results/oil-asvmodel/stateest-brent.csv', sep="," , header = False)
x2 = pd.read_csv('results/oil-asvmodel/stateest-dubai.csv', sep="," , header = False)
x3 = pd.read_csv('results/oil-asvmodel/stateest-maya.csv', sep="," , header = False)

m1 = pd.read_csv('results/oil-asvmodel/thhat-brent.csv', sep="," , header = False)
m2 = pd.read_csv('results/oil-asvmodel/thhat-dubai.csv', sep="," , header = False)
m3 = pd.read_csv('results/oil-asvmodel/thhat-maya.csv', sep="," , header = False)

d1 = np.loadtxt('data/gpo_joe2015/brent_19970110_20100604.csv')
d2 = np.loadtxt('data/gpo_joe2015/dubai_19970110_20100604.csv')
d3 = np.loadtxt('data/gpo_joe2015/maya_19970110_20100604.csv')

##############################################################################
# Compute the smoothed residuals and approximate the inverse CDF
##############################################################################

ehat1 = np.exp( -0.5 * x1['xhats'] ) * d1
ehat2 = np.exp( -0.5 * x2['xhats'] ) * d2
ehat3 = np.exp( -0.5 * x3['xhats'] ) * d3

( uhat1, thU1, thL1, varU1, varL1, ecdf1, ppf1 ) = evtECDF_kernel( ehat1, gpa, ehat1, 0.10 )
( uhat2, thU2, thL2, varU2, varL2, ecdf2, ppf2 ) = evtECDF_kernel( ehat2, gpa, ehat2, 0.10 )
( uhat3, thU3, thL3, varU3, varL3, ecdf3, ppf3 ) = evtECDF_kernel( ehat3, gpa, ehat3, 0.10 )

#np.round( np.hstack(( thL1, thU1 )), 2)
#np.round( np.sqrt( np.abs( np.hstack(( varL1, varU1 )) ) ), 2)
#
#np.round( np.hstack(( thL2, thU2 )), 2)
#np.round( np.sqrt( np.abs( np.hstack(( varL2, varU2 )) ) ), 2)
#
#np.round( np.hstack(( thL3, thU3 )), 2)
#np.round( np.sqrt( np.abs( np.hstack(( varL3, varU3 )) ) ), 2)


#figure(2)
#subplot(2,3,1); plot(ehat1,uhat1,'k.'); xlabel("Brent"); ylabel("hybrid EDF");
#subplot(2,3,2); plot(ehat2,uhat2,'k.'); xlabel("Dubai"); ylabel("hybrid EDF");
#subplot(2,3,3); plot(ehat3,uhat3,'k.'); xlabel("Maya");  ylabel("hybrid EDF");
#
#subplot(2,3,4); plot(uhat1,uhat2,'k.'); xlabel("Brent"); ylabel("Dubai");
#subplot(2,3,5); plot(uhat1,uhat3,'k.'); xlabel("Brent"); ylabel("Maya");
#subplot(2,3,6); plot(uhat2,uhat3,'k.'); xlabel("Dubai"); ylabel("Maya");

# Write empricial DF to file
fileOut = pd.DataFrame(uhat1); fileOut.to_csv('results/oil-asvmodel/uhat-brent.csv');
fileOut = pd.DataFrame(uhat2); fileOut.to_csv('results/oil-asvmodel/uhat-dubai.csv');
fileOut = pd.DataFrame(uhat3); fileOut.to_csv('results/oil-asvmodel/uhat-maya.csv');

# Write residuals to file
fileOut = pd.DataFrame(ehat1); fileOut.to_csv('results/oil-asvmodel/ehat-brent.csv');
fileOut = pd.DataFrame(ehat2); fileOut.to_csv('results/oil-asvmodel/ehat-dubai.csv');
fileOut = pd.DataFrame(ehat3); fileOut.to_csv('results/oil-asvmodel/ehat-maya.csv');

th.uhat = np.vstack((uhat1,uhat2,uhat3)).transpose()
p       = (th.uhat).shape[1];
n       = (th.uhat).shape[0];

##############################################################################
# Find initalisation by Kendall's tau for Student t copula
##############################################################################

rho10Est = 2.0/np.pi * np.arcsin( stats.kendalltau(th.uhat[:,1],th.uhat[:,0]) )[0]
rho20Est = 2.0/np.pi * np.arcsin( stats.kendalltau(th.uhat[:,2],th.uhat[:,0]) )[0]
rho21Est = 2.0/np.pi * np.arcsin( stats.kendalltau(th.uhat[:,2],th.uhat[:,1]) )[0]

foo = np.diag( np.ones(3) );
foo[1,0]=rho10Est; foo[2,0]=rho20Est;  foo[2,1]=rho21Est;
foo[0,1]=rho10Est; foo[0,2]=rho20Est;  foo[1,2]=rho21Est;
foo = sp.linalg.sqrtm(foo)

th.par[1] = foo[1,0]
th.par[2] = foo[2,0]
th.par[3] = foo[2,1]

##############################################################################
# Use BFGS to optimize the log-posterior to fit the copula
##############################################################################

res = optimize.fmin_l_bfgs_b(th.evaluateLogPosteriorBFGS, np.array([1.0,foo[1,0],foo[2,0],foo[2,1]]), approx_grad=1, bounds=( (0.1,50.0),(-0.90,0.90),(-0.90,0.90),(-0.90,0.90) ) )

# Estimate standard deviations by numerical differencation to obtain Hessian
hes  = nd.Hessian( th.evaluateLogPosteriorBFGS, delta=.001 )
hinv  = np.linalg.pinv( hes( res[0] ) )

##############################################################################
# Estimate correlation structure
##############################################################################

# Store parameters and construct correlation matrix
th.storeParameters(res[0],th)
th.constructCorrelationMatrix( 3 )

# Estimate the Spearman correlation using Monte Carlo
nSims = 100000;
foo   = multiTSimulate(nSims,np.zeros(p),th.P,th.par[0])
corrSpearman = ( stats.spearmanr(foo[:,0],foo[:,1])[0], stats.spearmanr(foo[:,0],foo[:,2])[0], stats.spearmanr(foo[:,1],foo[:,2])[0] )

##############################################################################
# Estimate Var
##############################################################################

# Quantile level
alpha = 0.01;

# Simulate from the copula
usim = (stats.t).cdf( foo, th.par[0] )

# Compute the quantile transformation for each type of oil
esim1 = ppf1.fit( usim[:,0] )[0]
esim2 = ppf2.fit( usim[:,1] )[0]
esim3 = ppf3.fit( usim[:,2] )[0]

# Compute the resulting log-return for each time step
varEst1 = np.zeros( th.T )
varEst2 = np.zeros( th.T )
varEst3 = np.zeros( th.T )

for tt in range(th.T):
    varEst1[tt] = np.percentile( esim1 * np.exp( ( x1['xhats'] )[tt] * 0.5 ), 100.0 * alpha )
    varEst2[tt] = np.percentile( esim2 * np.exp( ( x2['xhats'] )[tt] * 0.5 ), 100.0 * alpha )
    varEst3[tt] = np.percentile( esim3 * np.exp( ( x3['xhats'] )[tt] * 0.5 ), 100.0 * alpha )

# Write var to file
fileOut = pd.DataFrame(varEst1); fileOut.to_csv('results/oil-asvmodel/var-brent.csv');
fileOut = pd.DataFrame(varEst2); fileOut.to_csv('results/oil-asvmodel/var-dubai.csv');
fileOut = pd.DataFrame(varEst3); fileOut.to_csv('results/oil-asvmodel/var-maya.csv');

##############################################################################
# Print to screen
##############################################################################

print( np.round( res[0] ,2) )
print( np.round( np.sqrt( np.diag( np.abs( hinv) ) ) ,2) )
print( np.round( th.P ,2) )
print( np.round(corrSpearman,2) )

#>>> >>> >>> print( np.round( res[0] ,2) )
#[ 9.38 -0.1   0.33  0.25]
#>>> print( np.round( np.sqrt( np.diag( np.abs( hinv) ) ) ,2) )
#[ 1.97  0.03  0.02  0.02]
#>>> print( np.round( th.P ,2) )
#[[ 1.    0.77  0.84]
# [ 0.77  1.    0.79]
# [ 0.84  0.79  1.  ]]
#>>> print( np.round(corrSpearman,2) )
#[ 0.75  0.82  0.77]

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################