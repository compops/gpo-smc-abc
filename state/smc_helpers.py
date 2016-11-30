##############################################################################
##############################################################################
# Default settings and helpers for
# Particle filtering and smoothing
#
# Copyright (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

import numpy as np
import pandas
import os

##############################################################################
# Set default settings if needed
##############################################################################


def setSettings(sm, vers):

    #=====================================================================
    # Settings for the ABC particle filter
    #=====================================================================
    if (vers == "abcfilter"):
        vers = "filter"

        if (sm.rejectionSMC == None):
            print("abc-pf (rejectionSMC): defaulting to not using rejection SMC.")
            sm.rejectionSMC = False

        if (sm.adaptTolLevel == None):
            print(
                "abc-pf (adaptTolLevel,propAlive): defaulting to not adapting the tolerance level.")
            sm.adaptTolLevel = False
            sm.propAlive = 0

        if (sm.propAlive == None):
            sm.propAlive = 0.10
            print("abc-pf (propAlive): defaulting to setting alpha to " +
                  str(sm.propAlive) + ".")

        if ((sm.adaptTolLevel != True) & (sm.tolLevel == None)):
            sm.tolLevel = 0.10
            print("abc-pf (tolLevel) : defaulting using " +
                  str(sm.tolLevel) + " as the tolerance level (epsilon).")

        if (sm.weightdist == None):
            sm.weightdist = "gaussian"
            print(
                "abc-pf (weightdist): defaulting using noisy ABC with the Gaussian kernel.")

    #=====================================================================
    # Settings for the filter
    #=====================================================================
    if (vers == "filter"):
        if (sm.xo != None):
            sm.genInitialState = False

        if (sm.genInitialState == None):
            print(
                "pf (genInitialState): No initial state given, so assuming known zero state.")
            sm.genInitialState = False
            sm.xo = 0.0

        if (sm.nPart == None):
            print(
                "pf (nPart): No of particles not given, so defaulting to using N=T=" + str(sm.T) + ".")
            sm.nPart = sm.T

        if (sm.resamplingType == None):
            print(
                "pf (resamplingType): No resampling scheme given, so defauling to systematic resampling.")
            sm.resamplingType = "systematic"

        if (sm.resampFactor == None):
            print("pf (resampFactor): No limit of effective particles given for resampling, so resampling at every iteration.")
            sm.resampFactor = 2.0

    #=====================================================================
    # Specific settings for the FL smoother
    #=====================================================================
    if (vers == "flsmoother"):
        vers = "smoother"

        if (sm.fixedLag == None):
            sm.fixedLag = np.int(np.floor(np.log(sm.T)))
            print("ps (fixedLag): no fixed-lag given, so running with rule-of-thumb and Delta: " +
                  str(sm.fixedLag) + ".")

    #=====================================================================
    # Settings for the general smoother
    #=====================================================================
    if (vers == "smoother"):
        if (sm.calcHessianFlag == None):
            print("ps (calcHessianFlag): defaulting to not computing Hessians.")
            sm.calcHessianFlag = False
        elif (sm.calcHessianFlag == True):
            sm.calcGradientFlag = True

        if (sm.calcGradientFlag == None):
            print("ps (calcGradientFlag): defaulting to not computing gradients.")
            sm.calcGradientFlag = False

        if (sm.calcQFlag == None):
            print("ps (calcQFlag): defaulting to not computing Q function.")
            sm.calcQFlag = False

##############################################################################
# Calculate the pdf of a univariate Gaussian
##############################################################################


def uninormpdf(x, mu, sigma):
    return 1.0 / np.sqrt(2.0 * np.pi * sigma**2) * np.exp(- 0.5 * (x - mu)**2 * sigma**(-2))

##############################################################################
# Calculate the log-pdf of a univariate Gaussian
##############################################################################


def loguninormpdf(x, mu, sigma):
    return -0.5 * np.log(2.0 * np.pi * sigma**2) - 0.5 * (x - mu)**2 * sigma**(-2)

##############################################################################
# Calculate the log-pdf of a univariate quasi-Cauchy distribution
##############################################################################


def loguniqcauchypdf(x, mu, h):
    u = (x - mu) / h
    K = -2.0 * np.log(1.0 + 0.25 * np.pi**2 * u**2)

    return K / h

##############################################################################
# Calculate the log-pdf of a multivariate Gaussian with mean vector mu and covariance matrix S
##############################################################################


def lognormpdf(x, mu, S):
    nx = len(S)
    norm_coeff = nx * np.log(2.0 * np.pi) + np.linalg.slogdet(S)[1]
    err = x - mu

    numerator = np.dot(np.dot(err, np.linalg.pinv(S)), err.transpose())
    return -0.5 * (norm_coeff + numerator)

##############################################################################
# Check if a matrix is positive semi-definite but checking for negative eigenvalues
##############################################################################


def isPSD(x):
    return np.all(np.linalg.eigvals(x) > 0)

##########################################################################
# Helper: Reconstruct the particle trajectories
##########################################################################


def reconstructTrajectories_helper(sm, sys):
    xtraj = np.zeros((sm.nPart, sys.T))
    xtraj[:, sys.T - 1] = sm.p[:, sys.T - 1]

    # Plot all the particles and their resampled ancestors
    for ii in range(0, sm.nPart):
        att = ii
        for tt in np.arange(sys.T - 2, 0, -1):

            at = sm.a[att, tt + 1]
            at = at.astype(int)
            xtraj[ii, tt] = sm.p[at, tt]

            att = at
            att = att.astype(int)

    sm.x = xtraj

##########################################################################
# Helper: compile the results and write to file
##########################################################################


def writeToFile_helper(sm, fileOutName=None, noLLests=False):

    # Compile the results for output for smoother and filter
    if hasattr(sm, 'xhats'):
        # Smoother
        columnlabels = [None] * 3
        columnlabels[0] = "xhats"
        columnlabels[1] = "xhatf"
        columnlabels[2] = "llt"
        out = np.hstack((sm.xhats, sm.xhatf, (sm.llt).reshape((sm.T, 1))))
    else:
        # Filter
        columnlabels = [None] * 2
        columnlabels[0] = "xhatf"
        columnlabels[1] = "llt"
        out = np.hstack((sm.xhatf, (sm.llt).reshape((sm.T, 1))))

    # Write out the results to file
    fileOut = pandas.DataFrame(out, columns=columnlabels)

    if (fileOutName == None):
        if hasattr(sm, 'xhats'):
            fileOutName = 'results/' + \
                str(sm.filePrefix) + '/state_' + sm.filterType + '_' + \
                sm.smootherType + '_N' + str(sm.nPart) + '.csv'
        else:
            fileOutName = 'results/' + \
                str(sm.filePrefix) + '/state_' + \
                sm.filterType + '_N' + str(sm.nPart) + '.csv'

    ensure_dir(fileOutName)
    fileOut.to_csv(fileOutName)

    print("writeToFile_helper: wrote results to file: " + fileOutName)

##############################################################################
# Check if dirs for outputs exists, otherwise create them
##############################################################################


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
