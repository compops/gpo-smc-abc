import numpy             as np
import scipy             as sp

from   state             import smc
from   models            import hwsvalpha_4parameters

from   models.dists      import genPareto
from   misc.ecdf_tools   import ecdf, inv_ecdf

from models.copula       import studentt
from models.models_dists import multiTSimulate
from scipy               import stats
from scipy               import optimize

def estimateLogVolatility ( data, theta ):
    # Arrange the data structures
    sm                = smc.smcSampler();
    
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
    th.storeParameters( theta, sys)
    
    # Setup the SMC algorithm
    sm.filter                           = sm.bPFabc
    sm.nPart                            = 5000
    sm.genInitialState                  = True
    sm.weightdist                       = "gaussian"
    sm.tolLevel                         = 0.10
    
    # Add noise to data for noisy ABC
    th.makeNoisy(sm)
    
    # Estimate the log-volatility
    sm.filter(th)    
    
    return sm.xhatf[:,0]

def computeValueAtRisk( x, d, Test, alpha ):
    
    nAssets           = x.shape[1]
    T                 = x.shape[0]
    
    # Setup the system
    th                = studentt.copula()
    th.nPar           = int( 0.5 * nAssets * ( nAssets - 1.0 ) + 1.0 )
    th.par            = np.ones((th.nPar,1))
    th.T              = Test
    th.xo             = 0.0
    th.nParInference  = th.nPar
       
    # Compute the smoothed residuals and approximate the inverse CDF
    ehat   = np.zeros((nAssets,Test))     
    uhat   = np.zeros((nAssets,Test))      
    
    for ii in range(nAssets):
        ehat[ii,:] = np.exp( -0.5 * x[0:Test,ii] ) * d[0:Test,ii]
        uhat[ii,:] = pobs( ehat[ii,:] )
    
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
            kk += 1;
    
    # Use BFGS to optimize the log-posterior to fit the copula   
    #b   = [(0.1,50.0)] + [(-0.90,0.90)]*(len(th.par)-1)
    #res = optimize.fmin_l_bfgs_b(th.evaluateLogPosteriorBFGS, (th.par).transpose(), approx_grad=1, bounds=b )
    b   = [(0.1,50.0)]
    res = optimize.fmin_l_bfgs_b(th.evaluateLogPosteriorBFGS, 1.0, approx_grad=1, bounds=b )

    # Store parameters and construct correlation matrix
    th.nParInference = 1
    th.storeParameters(res[0],th)
    th.constructCorrelationMatrix( nAssets )
    
    th.T = T
           
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
        esim[ii,:] = inv_pobs( ehat[ii,:], usim[:,ii], 100 )
        
        for tt in range(th.T):
            varEst[tt,ii] = np.percentile( esim[ii,:] * np.exp( ( x[:,ii] )[tt] * 0.5 ), 100.0 * alpha )
    
    # Return VAR estimates
    return corrSpearman, varEst

##############################################################################
##############################################################################
##############################################################################

import pandas

# Get the log-returns
log_returns    = np.loadtxt('data/30_industry_portfolios_marketweighted.txt',skiprows=1)[:,1:]
T              = log_returns.shape[0]
nAssets        = log_returns.shape[1]

Test           = 805 
log_volatility = np.zeros((T,nAssets))
models         = np.zeros((4,nAssets))

for ii in range(nAssets):
    models[:,ii]         = np.matrix( pandas.read_csv('results/example3-portfolio-gpoabc-step1-models' + str(ii//5) + '.csv') )[:,ii+1].reshape(4)
    log_volatility[:,ii] = estimateLogVolatility( log_returns[0:T,ii], models[:,ii] )

# Compute the VAR
correlation, value_at_risk = computeValueAtRisk(log_volatility, log_returns[:,0:nAssets], Test, 0.01)

# Plot
plot(np.mean(value_at_risk[10:],axis=1))
plot(np.mean(log_returns[10:,0:nAssets],axis=1),'k.')

# Count number of violations on validation data
np.sum( np.mean(value_at_risk[Test:],axis=1) > np.mean(log_returns[Test:,0:nAssets],axis=1) )

# Export
import pandas
fileOut = pandas.DataFrame(value_at_risk)
fileOut.to_csv('results/example3-portfolio-gpoabc-var.csv')
