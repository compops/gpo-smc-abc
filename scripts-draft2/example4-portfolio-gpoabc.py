import numpy   as np
import scipy   as sp

import matplotlib.pylab as plt

from   state   import smc
from   para    import gpo_gpy
from   models  import hwsvalpha_4parameters

from models.copula       import studentt
from models.models_dists import multiTSimulate
from scipy               import stats
from scipy               import optimize


def estimateLogVolatility ( data ):
    # Arrange the data structures
    sm                = smc.smcSampler();
    gpo               = gpo_gpy.stGPO();
    
    # Setup the system
    sys               = hwsvalpha_4parameters.ssm()
    sys.par           = np.zeros((sys.nPar,1))
    sys.xo            = 0.0
    sys.T             = len(data)
    sys.version       = "standard"
    sys.transformY    = "none"
    
    # Load data
    sys.generateData()
    sys.y          = np.array( data ).reshape((sys.T,1))
    sys.ynoiseless = np.array( data ).reshape((sys.T,1))
   
    # Setup the parameters
    th               = hwsvalpha_4parameters.ssm()
    th.nParInference = 4
    th.copyData(sys)
    th.version       = "standard"
    th.transformY    = "arctan"
    
    # Setup the GPO algorithm
    gpo.verbose                         = True
    
    gpo.initPar                         = np.array([ 0.20, 0.95, 0.14,  1.90 ])
    gpo.upperBounds                     = np.array([ 2.00, 1.00, 1.00,  2.00 ])
    gpo.lowerBounds                     = np.array([-2.00, 0.80, 0.05,  1.00 ])
    
    gpo.preIter                         = 50    
    gpo.maxIter                         = 100    
    
    gpo.jitteringCovariance             = 0.01 * np.diag(np.ones(th.nParInference))
    gpo.preSamplingMethod               = "latinHyperCube"
    
    gpo.EstimateHyperparametersInterval = 50
    gpo.EstimateThHatEveryIteration     = False
    gpo.EstimateHessianEveryIteration   = False
    
    # Setup the SMC algorithm
    sm.filter                           = sm.bPFabc
    
    sm.nPart                            = 5000
    sm.genInitialState                  = True
    sm.weightdist                       = "gaussian"
    sm.tolLevel                         = 0.10
    
    # Add noise to data for noisy ABC
    th.makeNoisy(sm)
    
    # Estimate parameters
    gpo.bayes(sm, sys, th)
    
    # Estimate the log-volatility
    th.storeParameters( gpo.thhat, sys)
    sm.calcGradientFlag = False
    sm.calcHessianFlag  = False
    sm.nPaths           = 50
    sm.rho              = 5.0
    sm.nPathsLimit      = 10
    sm.ffbsiPS(th)    
    
    return sm.xhats[:,0], gpo.thhat


def computeValueAtRisk( x, d, alpha ):
    
    nAssets           = x.shape[1]
    T                 = x.shape[0]
    
    # Setup the system
    th                = studentt.copula()
    th.nPar           = int( 0.5 * nAssets * ( nAssets - 1.0 ) + 1.0 )
    th.par            = np.ones((th.nPar,1))
    th.T              = T
    th.xo             = 0.0
    th.nParInference  = th.nPar
       
    # Compute the smoothed residuals and approximate the inverse CDF
    ehat  = np.zeros((nAssets,T))     
    uhat  = np.zeros((nAssets,T))     
    
    for ii in range(nAssets):
        ehat[ii,:] = np.exp( -0.5 * x[:,ii] ) * d[:,ii]
        uhat[ii,:] = stats.norm.cdf( ehat[ii,:] )
    
    th.uhat = uhat.transpose()
    
    # Find initalisation by Kendall's tau for Student t copula
    rhoHat   = np.ones((nAssets,nAssets))
    
    for ii in range(nAssets):
        for jj in range(ii):
            rhoHat[ii,jj] = 2.0 / np.pi * np.arcsin( stats.kendalltau( th.uhat[:,ii], th.uhat[:,jj] ) )[0]
            rhoHat[jj,ii] = rhoHat[ii,jj]
    
    foo = sp.linalg.sqrtm(rhoHat)
    
    kk = 1
    for ii in range(nAssets):
        for jj in range(ii+1,nAssets):
            th.par[kk] = foo[ii,jj]
            kk += 1
    
    # Use BFGS to optimize the log-posterior to fit the copula   
    #b   = [(0.1,50.0)] + [(-0.90,0.90)]*(len(th.par)-1)
    #res = optimize.fmin_l_bfgs_b(th.evaluateLogPosteriorBFGS, (th.par).transpose(), approx_grad=1, bounds=b )
    b   = [(0.1,50.0)]
    res = optimize.fmin_l_bfgs_b(th.evaluateLogPosteriorBFGS, 1.0, approx_grad=1, bounds=b )

    
    # Store parameters and construct correlation matrix
    th.nParInference = 1
    th.storeParameters(res[0],th)
    th.constructCorrelationMatrix( nAssets )
           
    # Simulate from the copula
    nSims        = 100000
    foo          = multiTSimulate(nSims,np.zeros(nAssets),th.P,th.par[0])
    usim         = (stats.t).cdf( foo, th.par[0] )
    
    # Compute spearman correlation    
    corrSpearman = np.ones((int(0.5*(nAssets**2-nAssets)),1))    
    kk = 0
    for ii in range(nAssets):
        for jj in range(ii+1,nAssets):
            corrSpearman[kk] = stats.spearmanr(foo[:,ii],foo[:,jj])[0]
            kk = kk +1

    ##########################################################################    
    # Estimate Var
    ##########################################################################
       
    # Compute the quantile transformation for each asset and then Var
    esim   = np.zeros((nAssets,nSims))
    varEst = np.zeros((th.T,nAssets))
    
    for ii in range( nAssets ):
        esim[ii,:] = (stats.norm).ppf ( usim[:,ii] )
        for tt in range(th.T):
            varEst[tt,ii] = np.percentile( esim[ii,:] * np.exp( ( x[:,ii] )[tt] * 0.5 ), 100.0 * alpha )
    
    # Return VAR estimates
    return corrSpearman, varEst

##############################################################################
##############################################################################
##############################################################################

# Get the log-returns
log_returns    = np.loadtxt('data/30_industry_portfolios_marketweighted.txt',skiprows=1)[:,1:]
T              = log_returns.shape[0]
nAssets        = log_returns.shape[1]

# Estimate the log-volatility
log_volatility          = np.zeros((T,nAssets))
models                  = np.zeros((4,nAssets))

##############################################################################
# Load file
##############################################################################

#for ii in range(nAssets):
#    log_volatility[:,ii], models[:,ii] = estimateLogVolatility( log_returns[:,ii] )

# Compute the VAR
correlation, value_at_risk = computeValueAtRisk(log_volatility, log_returns[:,0:nAssets], 0.01)

plt.plot(np.mean(-value_at_risk,axis=1))
plt.plot(np.mean(log_returns[:,0:nAssets],axis=1),'k.')
plt.plot(np.mean(value_at_risk,axis=1))

ab1 = np.mean( np.mean(-value_at_risk,axis=1) < np.mean(log_returns[:,0:nAssets],axis=1) )
ab2 = np.mean( np.mean(value_at_risk,axis=1) > np.mean(log_returns[:,0:nAssets],axis=1) )

ab1 + ab2