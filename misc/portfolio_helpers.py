import urllib2

import pandas as pd
import numpy as np
import scipy as sp

from state import smc
from para import gpo_gpy
from models import hwsvalpha_4parameters, hwsv_4parameters

from misc.ecdf_tools import pobs, inv_pobs

from models.copula import studentt
from models.models_dists import multiTSimulate
from scipy import stats
from scipy import optimize


##############################################################################
# Get the oil future data
##############################################################################

def getOilData():
    
    # Get the Excel file and parse the first sheet
    url = 'http://freakonometrics.free.fr/oil.xls'
    socket = urllib2.urlopen(url)
    xd = pd.ExcelFile(socket)
    df=xd.parse(xd.sheet_names[0])
    
    # Select the data for brent, Dubai and Maya oil
    datasets = ['brent','Dubai','Maya']
    
    # Get the no. observations and the size of the estimation set (2/3)    
    T = df[datasets[0]].shape[0]
    Test = int(np.floor(2*T/3))
    nAssets = len(datasets)
    
    # Extract the log-returns for each asset    
    log_returns = np.zeros((T,nAssets))
    
    for ii in range(len(datasets)):
        log_returns[:,ii] = np.array(df[datasets[ii]])
        
    return log_returns, T, Test, nAssets


##############################################################################
# Get the stock data
##############################################################################

def getStockData(Test=805):
        
    f = 'data/30_industry_portfolios_marketweighted.txt'
    
    # Get the log-returns
    log_returns = np.loadtxt(f, skiprows=1)[:, 1:]
    T = log_returns.shape[0]
    nAssets = log_returns.shape[1]
        
    return log_returns, T, Test, nAssets
    

##############################################################################
# Estimating the SV model using GPO-SMC-(ABC)
##############################################################################

def estModel(version, data, settings):
    
    # Arrange the data structures
    gpo = gpo_gpy.stGPO()
    sm = smc.smcSampler()

    #=========================================================================
    # Setup the system model and load data
    #=========================================================================
    if (version=='GSV'):
        sys = hwsv_4parameters.ssm()
    else:
        sys = hwsvalpha_4parameters.ssm()
    
    sys.par = np.zeros((sys.nPar, 1))
    sys.xo = 0.0
    sys.T = len(data)
    
    sys.version = "standard"
    sys.transformY = "none"
    
    # Load data
    if (version=='GSV'):
        sys.generateData()
    else:
        sys.par[3] = 2.0
        sys.generateData()
        sys.ynoiseless = np.array(data).reshape((sys.T, 1))
    
    sys.y = np.array(data).reshape((sys.T, 1))

    #=========================================================================
    # Setup the model for inference
    #=========================================================================
    if (version=='GSV'):
        th = hwsv_4parameters.ssm()
        th.transformY = "none"
        th.nParInference = 3
    else:
        th = hwsvalpha_4parameters.ssm()
        th.transformY = "arctan"
        th.nParInference = 4
    
    th.version = "standard"    
    th.copyData(sys)

    #=========================================================================
    # Setup the SMC algorithm
    #=========================================================================

    if (version=='GSV'):
        sm.filter = sm.bPF
    else:
        sm.filter = sm.bPFabc
        sm.weightdist = settings['smc_weightdist']
        sm.tolLevel = settings['smc_tolLevel']
        
        # Add noise to data for noisy ABC
        th.makeNoisy(sm)

    sm.nPart = settings['smc_nPart']
    sm.genInitialState = True

    #=========================================================================
    # Setup the GPO algorithm
    #=========================================================================
    gpo.verbose = True

    gpo.initPar = settings['gpo_initPar']
    gpo.upperBounds = settings['gpo_upperBounds']
    gpo.lowerBounds = settings['gpo_lowerBounds']

    gpo.preIter = settings['gpo_preIter']
    gpo.maxIter = settings['gpo_maxIter']

    gpo.jitteringCovariance = 0.01 * np.diag(np.ones(th.nParInference))
    gpo.preSamplingMethod = "latinHyperCube"

    gpo.EstimateHyperparametersInterval = settings['gpo_estHypParInterval']
    gpo.EstimateThHatEveryIteration = False
    gpo.EstimateHessianEveryIteration = False
    
    # Estimate parameters
    gpo.bayes(sm, sys, th)
    gpo.estimateHessian()

    # Estimate the log-volatility
    th.storeParameters(gpo.thhat, sys)
    sm.filter(th)

    return sm.xhatf[:, 0], gpo.thhat, np.diag(-gpo.invHessianEstimate)


##############################################################################
# Estimating the log-volatility from parameters and data
##############################################################################

def estVol(version, data, theta, settings):
    
    # Arrange the data structures
    sm = smc.smcSampler()

    #=========================================================================
    # Setup the system model and load data
    #=========================================================================
    if (version=='GSV'):
        sys = hwsv_4parameters.ssm()
    else:
        sys = hwsvalpha_4parameters.ssm()
    
    sys.par = np.zeros((sys.nPar, 1))
    sys.xo = 0.0
    sys.T = len(data)
    sys.version = "standard"
    sys.transformY = "none"
    
    # Load data
    if (version=='GSV'):
        sys = hwsv_4parameters.ssm()
        sys.generateData()
    else:
        sys.par[3] = 2.0
        sys.generateData()
        sys.ynoiseless = np.array(data).reshape((sys.T, 1))
    
    sys.y = np.array(data).reshape((sys.T, 1))

    #=========================================================================
    # Setup the model for inference
    #=========================================================================
    if (version=='GSV'):
        th = hwsv_4parameters.ssm()
        th.transformY = "none"
    else:
        th = hwsvalpha_4parameters.ssm()
        th.transformY = "arctan"
    
    th.version = "standard"    
    th.nParInference = sys.nPar
    th.copyData(sys)

    #=========================================================================
    # Setup the SMC algorithm
    #=========================================================================

    if (version=='GSV'):
        sm.filter = sm.bPF
    else:
        sm.filter = sm.bPFabc
        sm.weightdist = settings['smc_weightdist']
        sm.tolLevel = settings['smc_tolLevel']
        
        # Add noise to data for noisy ABC
        th.makeNoisy(sm)

    sm.nPart = settings['smc_nPart']
    sm.genInitialState = True
    
    # Estimate the log-volatility
    sm.filter(th)
    
    # Return the log-volatility estimate
    return sm.xhatf[:, 0]


##############################################################################
# Estimate the Value-At-Risk from log-returns and log-volatility
# Fits a Student-t copula to the filtered residuals and simulates from the
# model to obtain the VaR-estimate
##############################################################################

def estVaR(x, d, Test, alpha):
    
    # Get the no. assets and no. time steps from the data
    Test = int(Test)
    nAssets = int(x.shape[1])
    T = int(x.shape[0])

    # Setup the copula model
    th = studentt.copula()
    th.nPar = int(0.5 * nAssets * (nAssets - 1.0) + 1.0)
    th.par = np.ones((th.nPar, 1))
    th.nParInference = th.nPar
    th.xo = 0.0

    
    #=========================================================================
    # Fit the copula model
    # Use only the estimation data (Test first observations)
    #=========================================================================
    th.T = Test

    # Compute the smoothed residuals and approximate the inverse CDF
    ehat = np.zeros((nAssets, Test))
    uhat = np.zeros((nAssets, Test))

    for ii in range(nAssets):
        ehat[ii, :] = np.exp(-0.5 * x[0:Test, ii]) * d[0:Test, ii]
        uhat[ii, :] = pobs(ehat[ii, :])

    th.uhat = uhat.transpose()

    # Find initalisation by Kendall's tau for Student t copula
    rhoHat = np.ones((nAssets, nAssets))

    for ii in range(nAssets):
        for jj in range(ii):
            rhoHat[ii, jj] = 2.0 / np.pi * \
                np.arcsin(stats.kendalltau(th.uhat[:, ii], th.uhat[:, jj]))[0]
            rhoHat[jj, ii] = rhoHat[ii, jj]

    foo = sp.linalg.sqrtm(rhoHat)

    kk = 1
    for ii in range(nAssets):
        for jj in range(ii + 1, nAssets):
            th.par[kk] = foo[ii, jj]
            kk += 1

    # Use BFGS to optimize the log-posterior to fit the copula
    b = [(0.1, 50.0)]
    res = optimize.fmin_l_bfgs_b(
        th.evaluateLogPosteriorBFGS, 2.0, approx_grad=1, bounds=b)
    
    # Store parameters and construct correlation matrix
    th.nParInference = 1
    th.storeParameters(res[0], th)
    th.constructCorrelationMatrix(nAssets)
    
    
    #=========================================================================
    # Simulate from the copula model
    # Use all the data (estimation and validation)
    #=========================================================================
    th.T = T

    # Simulate from the copula
    nSims = 100000
    foo = multiTSimulate(nSims, np.zeros(nAssets), th.P, th.par[0])
    usim = (stats.t).cdf(foo, th.par[0])

    # Compute spearman correlation
    corrSpearman = np.ones((int(0.5 * (nAssets**2 - nAssets)), 1))
    kk = 0
    for ii in range(nAssets):
        for jj in range(ii + 1, nAssets):
            corrSpearman[kk] = stats.spearmanr(foo[:, ii], foo[:, jj])[0]
            kk = kk + 1

    ##########################################################################
    # Estimate Var
    ##########################################################################

    # Compute the quantile transformation for each asset and then Var
    esim = np.zeros((nAssets, nSims))
    varEst = np.zeros((th.T, nAssets))

    for ii in range(nAssets):
        esim[ii, :] = inv_pobs(ehat[ii, :], usim[:, ii], 100)

        for tt in range(th.T):
            varEst[tt, ii] = np.percentile(
                esim[ii, :] * np.exp((x[:, ii])[tt] * 0.5), 100.0 * alpha)

    # Return VAR estimates
    return res[0], corrSpearman, varEst