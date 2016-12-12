##############################################################################
##############################################################################
# Model specification
# HW model with leverage
#
# Copyright (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

#=============================================================================
# Model structure
#=============================================================================
#
# xtt = par[0] + par[1] * ( xt - par[0] ) + par[2] * vt,
# yt  = exp( 0.5 * xt) * et,
#
# ( v(t), e(t) ) ~ N(0,E), E = ( 1, par[3]; par[3], 1)

import numpy as np
from scipy.stats import norm
from models_helpers import *
from models_dists import *


class ssm(object):
    #=========================================================================
    # Define model settings
    #=========================================================================
    nPar = 4
    par = np.zeros(nPar)
    modelName = "Hull-White stochastic volatility model with leverage"
    filePrefix = "hwsv"
    supportsFA = False
    nParInfernce = None
    nQInference = None
    transformY = "none"
    version = "standard"

    #=========================================================================
    # Define Jacobian and parameter transforms
    #=========================================================================
    def Jacobian(self):
        if (self.version == "tanhexp"):
            return np.log(1.0 - self.par[1]**2) + np.log(self.par[2])
        else:
            return 0.0

    def transform(self):
        if (self.version == "tanhexp"):
            self.par[1] = np.tanh(self.par[1])
            self.par[2] = np.exp(self.par[2])

        return None

    def invTransform(self):
        if (self.version == "tanhexp"):
            self.par[1] = np.arctanh(self.par[1])
            self.par[2] = np.log(self.par[2])

        return None

    #=========================================================================
    # Define the model
    #=========================================================================
    def generateInitialState(self, nPart):
        return self.par[0] + np.random.normal(size=(1, nPart)) * self.par[2] / np.sqrt(1.0 - self.par[1]**2)

    def evaluateState(self, xtt, xt, tt):
        return norm.pdf(xtt, self.par[0] + self.par[1] * (xt - self.par[0]), self.par[2])

    def generateState(self, xt, tt):

        # Deterministic part
        dpart = self.par[0] + self.par[1] * (xt - self.par[0])

        # Stochastic shock (noise) uncorrelated with returns
        spart = self.par[
            2] * np.sqrt(1.0 - self.par[3]**2) * np.random.randn(1, len(xt))

        # Stochastic shock (noise) correlated with returns
        et = self.y[tt] * np.exp(-0.5 * xt)
        lpart = self.par[2] * self.par[3] * et

        # Calculate output
        return dpart + spart + lpart

    def generateObservation(self, xt, tt):

        w = np.random.randn(1, len(xt)) * np.exp(0.5 * xt)

        if (self.transformY == "arctan"):
            w = np.arctan(w)

        return (w, np.zeros(len(xt)), np.zeros(len(xt)))

    def evaluateObservation(self, xt, tt):

        if (self.transformY == "arctan"):
            return norm.logpdf(np.arctan(self.y[tt]), 0, np.exp(xt / 2))
        if (self.transformY == "none"):
            return norm.logpdf(self.y[tt], 0, np.exp(xt / 2))

    def generateInitialStateRV(self, nPart, u):
        return self.par[0] + u[range(1, nPart + 1)] * self.par[2] / np.sqrt(1.0 - self.par[1]**2)

    def generateStateRV(self, xt, tt, u):
        # Deterministic part
        dpart = self.par[0] + self.par[1] * (xt - self.par[0])

        # Stochastic shock (noise) uncorrelated with returns
        spart = self.par[2] * \
            np.sqrt(1.0 - self.par[3]**2) * u[range(1, len(xt) + 1)]

        # Stochastic shock (noise) correlated with returns
        et = self.y[tt] * np.exp(-0.5 * xt)
        lpart = self.par[2] * self.par[3] * et

        # Calculate output
        return dpart + spart + lpart

    #=========================================================================
    # Define gradients of logarithm of complete data-likelihood
    #=========================================================================
    def Dparm(self, xtt, xt, st, at, tt):

        nOut = len(xtt)
        gradient = np.zeros((nOut, self.nParInference))
        Q1 = self.par[2]**(-1)
        Q2 = self.par[2]**(-2)
        Q3 = self.par[2]**(-3)
        px = xtt - self.par[0] - self.par[1] * (xt - self.par[0])

        if (self.version == "tanhexp"):
            for v1 in range(0, self.nParInference):
                if v1 == 0:
                    gradient[:, v1] = Q2 * px * (1.0 - self.par[1])
                elif v1 == 1:
                    gradient[:, v1] = Q2 * px * \
                        (xt - self.par[0]) * (1.0 - self.par[1]**2)
                elif v1 == 2:
                    gradient[:, v1] = (Q2 * px**2 - 1.0)
                elif v1 == 3:
                    raise NameError("Gradients of rho, not implemented")
                else:
                    gradient[:, v1] = 0.0
        else:
            for v1 in range(0, self.nParInference):
                if v1 == 0:
                    gradient[:, v1] = Q2 * px * (1.0 - self.par[1])
                elif v1 == 1:
                    gradient[:, v1] = Q2 * px * (xt - self.par[0])
                elif v1 == 2:
                    gradient[:, v1] = (Q3 * px**2 - Q1)
                elif v1 == 3:
                    raise NameError("Gradients of rho, not implemented")
                else:
                    gradient[:, v1] = 0.0
        return(gradient)

    #=========================================================================
    # Define Hessians of logarithm of complete data-likelihood
    #=========================================================================
    def DDparm(self, xtt, xt, st, at, tt):
        nOut = len(xtt)
        hessian = np.zeros((nOut, self.nParInference, self.nParInference))
        return(hessian)

    #=========================================================================
    # Define data generation mechanisms
    #=========================================================================
    def generateData(self, u=None, fileName=None, order=None):
        # Set input to zero if not given
        if (u == None):
            u = np.zeros(self.T)

        self.u = u

        x = np.zeros((self.T + 1, 1))
        y = np.zeros((self.T, 1))
        x[0] = self.xo

        if (fileName == None):
            # Generate w, e and J
            v = np.random.randn(self.T + 1)
            e = np.random.randn(self.T)

            # Generate observations and states
            for tt in range(0, self.T):
                # Generate observation
                y[tt] = np.exp(0.5 * x[tt]) * e[tt]

                # Generate state
                x[tt + 1] = self.par[0] + self.par[1] * (x[tt] - self.par[0]) + self.par[
                    2] * self.par[3] * e[tt] + self.par[2] * np.sqrt(1.0 - self.par[3]) * v[tt]

            self.x = x[0:self.T]
            self.y = y
            self.v = v
            self.e = e
        else:
            # Try to import data
            tmp = np.loadtxt(fileName, delimiter=",")

            if (order == None):
                self.y = np.array(
                    tmp[0:self.T], copy=True).reshape((self.T, 1))
                self.u = u
            elif (order == "y"):
                self.y = np.array(
                    tmp[0:self.T], copy=True).reshape((self.T, 1))
                self.u = u
            elif (order == "xy"):
                self.x = np.array(tmp[0:self.T, 0],
                                  copy=True).reshape((self.T, 1))
                self.y = np.array(tmp[0:self.T, 1],
                                  copy=True).reshape((self.T, 1))
                self.u = u
            elif (order == "xuy"):
                self.x = np.array(tmp[0:self.T, 0],
                                  copy=True).reshape((self.T, 1))
                self.u = np.array(tmp[0:self.T, 1],
                                  copy=True).reshape((self.T, 1))
                self.y = np.array(tmp[0:self.T, 2],
                                  copy=True).reshape((self.T, 1))
            else:
                raise NameError(
                    "generateData, import data: cannot import that order.")

    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================
    def priorUniform(self):
        out = 1.0

        if(np.abs(self.par[1]) > 1.0):
            out = 0.0

        if(self.par[2] < 0.0):
            out = 0.0

        if(np.abs(self.par[3]) > 1.0):
            out = 0.0

        return(out)

    #=========================================================================
    # Define log-priors for the PMH sampler
    #=========================================================================
    def prior(self):
        out = 0.0

        # Normal prior for mu
        if (self.nParInference >= 1):
            out += normalLogPDF(self.par[0], 0, 0.2)

        # Truncated normal prior for phi
        if (self.nParInference >= 2):
            out += normalLogPDF(self.par[1], 0.9, 0.05)

        # Gamma prior for sigma
        if (self.nParInference >= 3):
            out += gammaLogPDF(self.par[2], a=2.0, b=1.0 / 20.0)

        # Truncated normal prior for rho (truncation by hard prior)
        if (self.nParInference >= 4):
            out += normalLogPDF(self.par[3], -0.5, 0.2)

        return out

    #=========================================================================
    # Define gradients of log-priors for the PMH sampler
    #=========================================================================
    def dprior1(self, v1):
        if (self.version == "tanhexp"):
            if (v1 == 0):
                # Normal prior for mu
                return normalLogPDFgradient(self.par[0], 0, 0.2)
            elif (v1 == 1):
                # Truncated normal prior for phi
                return normalLogPDFgradient(self.par[1], 0.9, 0.05) * (1.0 - self.par[1]**2)
            elif (v1 == 2):
                # Gamma prior for sigma
                return gammaLogPDFgradient(self.par[2], a=2.0, b=1.0 / 20.0) * self.par[2]
            elif (v1 == 3):
                # Truncated normal prior for rho (truncation by hard prior)
                return normalLogPDFgradient(self.par[3], -0.5, 0.2)
            else:
                return 0.0

        else:
            if (v1 == 0):
                # Normal prior for mu
                return normalLogPDFgradient(self.par[0], 0, 0.2)
            elif (v1 == 1):
                # Truncated normal prior for phi
                return normalLogPDFgradient(self.par[1], 0.9, 0.05)
            elif (v1 == 2):
                # Gamma prior for sigma
                return gammaLogPDFgradient(self.par[2], a=2.0, b=1.0 / 20.0)
            elif (v1 == 3):
                # Truncated normal prior for rho (truncation by hard prior)
                return normalLogPDFgradient(self.par[3], -0.5, 0.2)
            else:
                return 0.0

    #=========================================================================
    # Define hessians of log-priors for the PMH sampler
    #=========================================================================
    def ddprior1(self, v1, v2):

        if ((v1 == 0) & (v1 == 0)):
            # Normal prior for mu
            return normalLogPDFhessian(self.par[0], 0, 0.2)
        elif ((v1 == 1) & (v1 == 1)):
            # Truncated normal prior for phi
            return normalLogPDFhessian(self.par[1], 0.9, 0.05)
        elif ((v1 == 2) & (v1 == 2)):
            # Gamma prior for sigma
            return gammaLogPDFhessian(self.par[2], a=2.0, b=1.0 / 20.0)
        elif ((v1 == 3) & (v1 == 3)):
            # Truncated normal prior for rho (truncation by hard prior)
            return normalLogPDFhessian(self.par[3], -0.5, 0.2)
        else:
            return 0.0

    #=========================================================================
    # Define hessians of log-priors for the PMH sampler
    #=========================================================================
    def samplePrior(self):

        out = np.zeros(self.nParInference)

        # Normal prior for mu
        if (self.nParInference >= 1):
            out[0] = np.random.normal(0.0, 0.20)

        # Truncated normal prior for phi (truncation by hard prior)
        if (self.nParInference >= 2):
            uu = 1.2
            while (uu > 1.0):
                uu = np.random.normal(0.9, 0.05)

            out[1] = uu

        # Gamma prior for sigma
        if (self.nParInference >= 3):
            out[2] = np.random.gamma(shape=2.0, scale=1.0 / 20.0)

        # Truncated normal prior for rho (truncation by hard prior)
        if (self.nParInference >= 4):
            uu = -1.2
            while (np.abs(uu) > 1.0):
                uu = np.random.normal(-0.5, 0.2)

            out[3] = uu

        return out

    #=========================================================================
    # Define standard methods for the model struct
    #=========================================================================

    # Make data noisy for the ABC procedur
    makeNoisy = template_makeNoisy

    # Standard operations on struct
    copyData = template_copyData
    storeParameters = template_storeParameters
    returnParameters = template_returnParameters

    # No faPF available for this model
    generateStateFA = empty_generateStateFA
    evaluateObservationFA = empty_evaluateObservationFA
    generateObservationFA = empty_generateObservationFA

    # No EM algorithm available for this model
    Qfunc = empty_Qfunc
    Mstep = empty_Mstep
