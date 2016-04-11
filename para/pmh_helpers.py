##############################################################################
##############################################################################
# Default settings and helpers for
# Particle Metrpolis-Hastings for Bayesian parameter inference
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy as np
import os

from scipy.special import gammaln

##############################################################################
# Set default settings if needed
##############################################################################
def setSettings(pmh,vers):

    #=========================================================================
    # Settings for all PMH algorithms
    #=========================================================================
    if ( pmh.nIter == None ):
        pmh.nIter = 1000;
        print("pmh (nIter): no number of iterations given, defaulting to " + str(pmh.nIter) + ".");

    if ( pmh.nBurnIn == None ):
        pmh.nBurnIn = 250;
        print("pmh (nBurnIn): no length of burn-in given, defaulting to " + str(pmh.nBurnIn) + ".");

    if ( pmh.nProgressReport == None ):
        pmh.nProgressReport = 100;
        print("pmh (nProgressReport): period for progress reports given, defaulting to " + str(pmh.nProgressReport) + ".");

    if ( pmh.dataset == None ):
        pmh.dataset = 0;
        print("pmh (dataset): no number of data set given, defaulting to " + str(pmh.dataset) + ".");

    if ( pmh.filePrefix == None ):
        pmh.filePrefix = "model";
        print("pmh (filePrefix): no short name for model given, defaulting to " + str(pmh.filePrefix) + ".");

    if ( pmh.adaptHessianAfterBurnIn == None ):
        pmh.adaptHessianAfterBurnIn = False;
        print("pmh (adaptHessianAfterBurnIn): defaulting to not adapt the Hessian using the burn-in.");

    if ( ( pmh.adaptHessianNoSamples == None ) & (pmh.adaptHessianAfterBurnIn == True ) ):
        pmh.adaptHessianNoSamples = np.floor( pmh.nBurnIn / 2.0 );
        print("pmh (adaptHessianNoSamples): defaulting to using half the burn-in, i.e. " + str(pmh.adaptHessianNoSamples) + " samples to adapt the Hessian.");

    if ( pmh.writeOutPriorWarnings == None ):
        pmh.writeOutPriorWarnings = False;
        print("pmh (writeOutPriorWarnings): defaulting to not writing out violations of hard prior.");

    if ( ( pmh.writeOutProgressToFileInterval == None ) & ( pmh.writeOutProgressToFile == None ) ):
        pmh.writeOutProgressToFileInterval  = 1000;
        pmh.writeOutProgressToFile          = True;
        print("pmh (writeOutProgressToFileInterval,writeOutProgressToFile): defaulting write out current progress to file for each: " + str(pmh.writeOutProgressToFileInterval) + " step of the algorithm.");

    #=========================================================================
    # Settings for PMH0 and PMH1 algorithms
    #=========================================================================
    if ( ( vers == "PMH0" ) | ( vers == "PMH1" ) ):

        if ( pmh.stepSize == None ):
            pmh.stepSize = 0.10;
            print("pmh (stepSize): no step size given, defaulting to " + str(pmh.stepSize) + " for all parameters.");

    #=========================================================================
    # Settings for all preconditioned PMH algorithms
    #=========================================================================
    if ( ( vers == "pPMH1" ) | ( vers == "pPMH0" ) | ( vers == "bpPMH1" ) ) :

        if ( ( vers == "pPMH0" ) & ( pmh.stepSize == None ) ):
            pmh.stepSize = 2.562 / np.sqrt( pmh.nPars );
            print("pmh0 (stepSize): no step size given, defaulting to optimal " + str(pmh.stepSize) + " for all parameters.");

        if ( ( vers == "pPMH1" ) & ( pmh.stepSize == None ) ):
            pmh.stepSize = 1.125 / np.sqrt( pmh.nPars**(1.0/3.0) );
            print("pmh1 (stepSize): no step size given, defaulting to optimal " + str(pmh.stepSize) + " for all parameters.");

        if ( pmh.invHessian == None ):
            print("pmh (invHessian): no inverse Hessian given, defaulting to unit matrix.");
            pmh.invHessian = np.diag( np.ones( pmh.nPars ) );

    #=========================================================================
    # Settings for all PMH2 algorithms
    #=========================================================================
    if ( ( vers == "PMH2" ) | ( vers == "qPMH2" ) ):
        if ( pmh.stepSize == None ):
            pmh.stepSize = 0.80;
            print("pmh (stepSize): no step size given, defaulting to " + str(pmh.stepSize) + " for all parameters.");

        if ( pmh.makeHessianDiagonal == None ):
            pmh.makeHessianDiagonal = False;
            print("pmh2 (makeHessianDiagonal): defaulting to use full Hessian (not only a diagonal Hessian).");

        # Fix for old run scripts
        if ( pmh.makeHessianPSDreject == True ):
            print("pmh2: using makeHessianPSDreject, change to pmh.makeHessianPSDmethod = reject." )
            pmh.makeHessianPSDmethod = "reject"

        # Fix for old run scripts
        if ( pmh.makeHessianPSDregularise == True ):
            print("pmh2: using makeHessianPSDregularise, change to pmh.makeHessianPSDmethod = regularise." )
            pmh.makeHessianPSDmethod = "regularise"

        # Fix for old run scripts
        if ( pmh.makeHessianPSDhybrid == True ):
            print("pmh2: using makeHessianPSDhybrid, change to pmh.makeHessianPSDmethod = hybrid." )
            pmh.makeHessianPSDmethod = "hybrid"

        if ( pmh.makeHessianPSDmethod == None ):
            pmh.makeHessianPSDmethod = "regularise";
            print("pmh2 (makeHessianPSDmethod): defaulting to make the Hessian PSD by regularisation.");

        if ( ( pmh.makeHessianPSDmethod == "hybrid" ) & ( pmh.PSDmethodhybridSamps == None ) ):
            pmh.PSDmethodhybridSamps = 1500;
            print("pmh (PSDmethodhybridSamps): hybrid method selected by no sample to use for covariance estimation not given, defaulting to " + str(pmh.makeHessianPSDhybridNoSamples) + ".");

    if ( vers == "qPMH2" ):

        if ( pmh.qPMHadaptInitialHessian == None ):
            pmh.qPMHadaptInitialHessian = False;
            print("qpmh2 (qPMHadaptInitialHessian): defaulting to not adapting initial Hessian.");

        if ( pmh.memoryLength == None ):
            pmh.memoryLength = 20;
            print("qpmh2 (memoryLength): defaulting to use memory length " + str(pmh.memoryLength) + " to estimate the Hessian.");

        if ( pmh.epsilon == None ):
            pmh.epsilon = 400;
            print("qpmh (epsilon): defaulting to use a diagonal matrix with" + str(pmh.epsilon) + " on the diagonal as initial Hessian.");

    #=========================================================================
    # Settings for fixing random variables in particle filter
    #=========================================================================
    if ( pmh.proposeRVs == True ):

        if ( pmh.rvnSamples == None ):
            pmh.rvnSamples = pmh.T * ( 1.0 + pmh.nPart );
            print("cpmh (rvnSamples): defaulting to have " + str( pmh.rvnSamples) + " random variables in the Markov chain.");

        if ( pmh.alpha == None ):
            pmh.alpha = 0.0;
            print("cpmh (alpha): defaulting to have " + str( pmh.rvnSamples) + " as probability for global move.");

        if ( pmh.sigmaU == None ):
            pmh.sigmaU = 0.80;
            print("pmh (sigmaU): defaulting to have " + str( pmh.sigmaU) + " as scaling in local random walk move.");

        if ( pmh.adaptHessianRecursively == None ):
            pmh.adaptHessianRecursively = False;
            print("pmh (adaptHessianRecursively): defaulting to not adapt Hessian.");

        if ( ( pmh.adaptHessianRecursively == True ) & ( pmh.adaptHessianRecursivelyIterLimitFrom == None ) ):
            pmh.adaptHessianRecursivelyIterLimitFrom = np.floor( pmh.nBurnIn * 0.50 );
            print("pmh (adaptHessianRecursivelyIterLimitFrom): defaulting to adapt Hessian during second half of burn-in.");

        if ( ( pmh.adaptHessianRecursively == True ) & ( pmh.adaptHessianRecursivelyIterLimitTo == None ) ):
            pmh.adaptHessianRecursivelyIterLimitTo = pmh.nBurnIn;
            print("pmh (adaptHessianRecursivelyIterLimit): defaulting to adapt Hessian up tp burn-in.");

        if ( ( pmh.adaptHessianRecursively == True ) ):
            pmh.adaptiveU_sigmau       = pmh.sigmaU * np.ones( pmh.nIter );
            pmh.adaptiveU_mean         = 0.0;
            pmh.adaptiveU_counter      = 0;

##############################################################################
# Print small progress reports
##############################################################################
def progressPrint(pmh):
    print("################################################################################################ ");
    print(" Iteration: " + str(pmh.iter+1) + " of : " + str(pmh.nIter) + " completed.")
    print("");
    print(" Current state of the Markov chain:               ")
    print(["%.4f" % v for v in pmh.th[pmh.iter,:]])
    print("");
    print(" Proposed next state of the Markov chain:         ")
    print(["%.4f" % v for v in pmh.thp[pmh.iter,:]])
    print("");
    print(" Current posterior mean estimate (untransformed): ")
    print(["%.4f" % v for v in np.mean(pmh.tho[range(pmh.iter),:], axis=0)])
    print("");
    print(" Current acceptance rate:                         ")
    print("%.4f" % np.mean(pmh.accept[range(pmh.iter)]) )
    if ( pmh.iter > (pmh.nBurnIn*1.5) ):
        print("");
        print(" Current IACT values:                         ")
        print(["%.2f" % v for v in pmh.calcIACT() ])
        print("");
        print(" Current log-SJD value:                          ")
        print( str( np.log( pmh.calcSJD() ) ) )
    if ( ( pmh.PMHtype == "qPMH2" ) & ( pmh.iter > pmh.memoryLength ) ):
        print("");
        print(" (qpmh2): mean no. samples for Hessian estimate:           ")
        print("%.4f" % np.mean(pmh.nHessianSamples[range(pmh.memoryLength,pmh.iter)]) )
    print("################################################################################################ ");

##############################################################################
# Check if dirs for outputs exists, otherwise create them
##############################################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

##########################################################################
# Compute the IACT
##########################################################################

def calcIACT_prototype( pmh, thhat=False, maxlag=False ):

    # Internal subroutine for computing IACT for a given parameter
    def estimateIACT( x, m, maxlag ):
        # estimate the acf coefficient
        # code from http://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
        n    = len(x)
        d    = np.asarray(x);
        nmax = int(np.floor(n/10.0));
        c0   = np.sum((x - m) ** 2) / float(n)

        def r(h):
            acf_lag = ((d[:n - h] - m) * (d[h:] - m)).sum() / float(n) / c0
            return round(acf_lag, 3)
        x = np.arange(n) # Avoiding lag 0 calculation
        acf_coeffs = map(r, x)

        if ( maxlag == False ):
            try:
                cutoff = np.where( np.abs( acf_coeffs[0:int(nmax)] ) < 2.0 / np.sqrt(n) )[0][0];
            except:
                cutoff = nmax;
        else:
            cutoff = maxlag;

        # estimate the maximum number of acf coefficients to use to calculate IACT
        tmp  = int( min(cutoff,nmax) );

        # estimate the IACT
        return 1.0 + 2.0 * np.sum( acf_coeffs[0:tmp] );

    # Estimate IACT
    IACT = np.zeros( pmh.nPars );
    MCtrace = pmh.th[ range( int(pmh.nBurnIn) , int(pmh.iter) ) , : ];

    if ( thhat == False ):
        thhat = np.mean( MCtrace, axis = 0 );

    for ii in range( pmh.nPars ):
        IACT[ii] = estimateIACT( MCtrace[:,ii], thhat[ii], maxlag )

    return IACT

##############################################################################
# Calculate the effective sample size
##############################################################################
def calculateESS_prototype( pmh, thhat=False, maxlag=False ):
    return( ( pmh.nIter - pmh.nBurnIn ) / calcIACT_prototype( pmh, thhat, maxlag ) );

##############################################################################
# Calculate the Squared Jump distance
##############################################################################
def calcSJD_prototype( pmh ):

    MCtrace = pmh.th[ range( int(pmh.nBurnIn) , int(pmh.iter) ) , : ];
    SJD     = np.sum( np.linalg.norm( np.diff(MCtrace,axis=0), 2, axis=1)**2 ) / ( MCtrace.shape[0] - 1.0 );

    return SJD;

##############################################################################
# Calculate the log-pdf of a univariate Gaussian
##############################################################################
def loguninormpdf(x,mu,sigma):
    return -0.5 * np.log( 2.0 * np.pi * sigma**2) - 0.5 * (x-mu)**2 * sigma**(-2);

##############################################################################
# Calculate the log-pdf of a multivariate Gaussian with mean vector mu and covariance matrix S
##############################################################################
def lognormpdf(x,mu,S):
    nx = len(S)
    norm_coeff = nx * np.log( 2.0 * np.pi ) + np.linalg.slogdet(S)[1]
    err = x-mu

    numerator = np.dot( np.dot(err,np.linalg.pinv(S)),err.transpose())
    return -0.5*(norm_coeff+numerator)

##############################################################################
# Check if a matrix is positive semi-definite but checking for negative eigenvalues
##############################################################################
def isPSD(x):
    return np.all(np.linalg.eigvals(x) > 0)

##############################################################################
# Zero-variance post processing with linear correction
##############################################################################
def zvpost_linear_prototype(pmh):

    ahat = np.zeros((pmh.nPars,pmh.nPars))

    for ii in range(pmh.nPars):
        z = -0.5 * pmh.gradient[pmh.nBurnIn:pmh.nIter,:]
        g = pmh.th[pmh.nBurnIn:pmh.nIter,ii]

        covAll = np.cov( np.vstack((z.transpose(),g.transpose())) )
        Sigma  = np.linalg.inv( covAll[0:3,0:3] )
        sigma  = covAll[0:3,3]
        ahat[:,ii] = - np.dot(Sigma, sigma)

    pmh.thzv = pmh.th[pmh.nBurnIn:pmh.nIter,:] + np.dot(z,ahat);

##############################################################################
# Logit and inverse-Logit transformations
##############################################################################

def logit( x ):
    return np.log ( x / ( 1.0 - x ) );

def invlogit( x ):
    return np.exp( x ) / ( 1.0 + np.exp( x ) );

##############################################################################
# Generate multivariate t random variables
#
# Code from
# http://kennychowdhary.me/2013/03/python-code-to-generate-samples-from-multivariate-t/
#
##############################################################################

def rmvt(mu,Sigma,N):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    N = degrees of freedom
    '''
    d = len(mu)

    g = np.tile( np.random.gamma( N/2.0, 2.0/N, 1 ),(d,1) ).T
    Z = np.random.multivariate_normal( np.zeros(d), Sigma.astype(float) )

    return ( mu + Z / np.sqrt(g) )[0];

##############################################################################
# Evaluate multivariate t log-density
##############################################################################

def logdmvt(x,mu,Sigma,df):
    p = len(x);

    part1 = gammaln( 0.5 * (df + p) );
    part2 = -gammaln( 0.5 * df ) - 0.5 * p * np.log( df ) - 0.5 * p * np.log( np.pi )
    part3 = - 0.5 * np.linalg.det( Sigma.astype(float) );
    part4 = - 0.5 * ( df + p ) * np.log( 1.0 + np.dot( np.dot( (x - mu), np.linalg.inv( Sigma.astype(float) ) ), (x - mu) ) / df );

    return part1 + part2 + part3 + part4;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################