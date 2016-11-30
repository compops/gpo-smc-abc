##############################################################################
##############################################################################
# Model specification
# Linear Gaussian state space model
#
# Copyright (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

#=============================================================================
# Model structure
#=============================================================================
# xtt = par[0] * xt  + par[1] * vt,
# yt  = xt           + par[2] * et.
#
# vt  ~ N(0,1)
# et  ~ N(0,1)

import numpy as np
from scipy.stats import norm
from models_helpers import *


class ssm(object):

    #=========================================================================
    # Define model settings
    #=========================================================================
    nPar = 3
    par = np.zeros(nPar)
    modelName = "Linear Gaussian system with three parameters"
    filePrefix = "lgss"
    supportsFA = True
    nParInfernce = None
    nQInference = None
    scale = 1.0
    version = "standard"

    #=========================================================================
    # Define the model
    #=========================================================================
    def generateInitialState(self, nPart):
        return np.random.normal(size=(1, nPart)) * self.par[1] / np.sqrt(1 - self.par[0]**2)

    def generateState(self, xt, tt):
        return self.par[0] * xt + self.u[tt] + self.par[1] * np.random.randn(1, len(xt))

    def evaluateState(self, xtt, xt, tt):
        return norm.pdf(xtt, self.par[0] * xt + self.u[tt], self.par[1])

    def generateObservation(self, xt, tt):
        return xt + self.par[2] * np.random.randn(1, len(xt))

    def evaluateObservation(self,  xt, tt, condPath=None):
        if (condPath == None):
            return norm.logpdf(self.y[tt], xt, np.sqrt(self.par[2]))
        else:
            return norm.logpdf(condPath, xt, np.sqrt(self.par[2]))

    def generateStateFA(self, xt, tt):
        delta = self.par[1]**(-2) + self.par[2]**(-2)
        delta = 1.0 / delta
        part1 = delta * (self.y[tt + 1] * self.par[2]**(-2) +
                         self.par[1]**(-2) * (self.par[0] * xt + self.u[tt]))
        part2 = np.sqrt(delta) * np.random.randn(1, len(xt))
        return part1 + part2

    def evaluateStateFA(self, condPath, xt, tt):
        delta = self.par[1]**(-2) + self.par[2]**(-2)
        delta = 1.0 / delta
        fa = delta * (self.y[tt] * self.par[2]**(-2) +
                      self.par[1]**(-2) * (self.par[0] * xt + self.u[tt]))
        return norm.logpdf(condPath, fa, np.sqrt(delta))

    def generateObservationFA(self, xt, tt):
        return self.par[0] * xt + self.u[tt] + np.sqrt(self.par[1]**2 + self.par[2]**2) * np.random.randn(1, len(xt))

    def evaluateObservationFA(self, xt, tt, condPath=None):
        return norm.logpdf(self.y[tt + 1], self.par[0] * xt + self.u[tt], np.sqrt(self.par[1]**2 + self.par[2]**2))

    def generateInitialStateRV(self, nPart, u):
        return norm.ppf(u[range(1, nPart + 1)]) * self.par[1] / np.sqrt(1.0 - self.par[0]**2)

    def generateStateRV(self, xt, tt, u):
        return self.par[0] * xt + self.par[1] * norm.ppf(u[range(1, len(xt) + 1)])

    def generateStateFARV(self, xt, tt, u):
        delta = self.par[1]**(-2) + self.par[2]**(-2)
        delta = 1.0 / delta
        part1 = delta * (self.y[tt + 1] * self.par[2]**(-2) +
                         self.par[1]**(-2) * self.par[0] * xt + self.u[tt])
        part2 = np.sqrt(delta) * norm.ppf(u[range(1, len(xt) + 1)])
        return part1 + part2

    #=========================================================================
    # Define Q-function for EM algorithm
    #=========================================================================
    def Qfunc(self, xtt, xt, st, at, tt):
        nOut = len(xtt)
        qq = np.zeros((nOut, self.nQInference))

        for v1 in range(self.nQInference):
            if v1 == 0:
                qq[:, v1] = xtt * xt
            elif v1 == 1:
                qq[:, v1] = self.u[tt] * xt
            elif v1 == 2:
                qq[:, v1] = xt * xt
            elif v1 == 3:
                qq[:, v1] = (xtt - self.par[0] * xt - self.u[tt])**2
            elif v1 == 4:
                qq[:, v1] = (self.y[tt] - xt)**2
            else:
                qq[:, v1] = 0.0

        return(qq)

    #=========================================================================
    # Define M-step for EM algorithm
    #=========================================================================
    def Mstep(self, sm, gamma=0.0):

        th = np.zeros(self.nParInference)

        if (gamma == 0.0):
            # No stochastic approximation
            for v1 in range(0, self.nParInference):
                if v1 == 0:
                    th[v1] = (sm.qfunc[0] - sm.qfunc[1]) / sm.qfunc[2]
                elif v1 == 1:
                    th[v1] = np.sqrt(sm.qfunc[3] / self.T)
                elif v1 == 2:
                    th[v1] = np.sqrt(sm.qfunc[4] / self.T)
                else:
                    th[v1] = 0.0
        else:
            # stochastic approximation
            self.suff = (1 - gamma) * self.suff + gamma * sm.qfunc

            for v1 in range(0, self.nParInference):
                if v1 == 0:
                    th[v1] = (self.suff[0] - self.suff[1]) / self.suff[2]
                elif v1 == 1:
                    th[v1] = np.sqrt(self.suff[3] / self.T)
                elif v1 == 2:
                    th[v1] = np.sqrt(self.suff[4] / self.T)
                else:
                    th[v1] = 0.0

        return th

    #=========================================================================
    # Define gradients of logarithm of complete data-likelihood
    #=========================================================================
    def Dparm(self, xtt, xt, st, at, tt):

        nOut = len(xtt)
        gradient = np.zeros((nOut, self.nParInference))
        Q1 = self.par[1]**(-1)
        Q2 = self.par[1]**(-2)
        Q3 = self.par[1]**(-3)
        R1 = self.par[2]**(-1)
        R3 = self.par[2]**(-3)
        px = xtt - self.par[0] * xt - self.u[tt]
        py = self.y[tt] - xt

        if (self.version == "tanhexp"):
            for v1 in range(0, self.nParInference):
                if v1 == 0:
                    gradient[:, v1] = xt * Q2 * px * (1.0 - self.par[0]**2)
                elif v1 == 1:
                    gradient[:, v1] = (Q2 * px**2 - 1.0) * self.scale
                elif v1 == 2:
                    gradient[:, v1] = R3 * py**2 - R1
                else:
                    gradient[:, v1] = 0.0
        else:
            for v1 in range(0, self.nParInference):
                if v1 == 0:
                    gradient[:, v1] = xt * Q2 * px
                elif v1 == 1:
                    gradient[:, v1] = (Q3 * px**2 - Q1) * self.scale
                elif v1 == 2:
                    gradient[:, v1] = R3 * py**2 - R1
                else:
                    gradient[:, v1] = 0.0
        return(gradient)

    #=========================================================================
    # Define Hessians of logarithm of complete data-likelihood
    #=========================================================================
    def DDparm(self, xtt, xt, st, at, tt):

        nOut = len(xtt)
        hessian = np.zeros((nOut, self.nParInference, self.nParInference))
        Q1 = self.par[1]**(-1)
        Q2 = self.par[1]**(-2)
        Q3 = self.par[1]**(-3)
        Q4 = self.par[1]**(-4)
        R2 = self.par[2]**(-2)
        R4 = self.par[2]**(-4)
        px = xtt - self.par[0] * xt - self.u[tt]
        py = self.y[tt] - xt

        if (self.version == "tanhexp"):
            for v1 in range(0, self.nParInference):
                for v2 in range(0, self.nParInference):
                    if ((v1 == 0) & (v2 == 0)):
                        hessian[:, v1, v2] = - xt**2 * Q2 * (1.0 - self.par[0]**2)**2 - 2.0 * self.par[
                            0] * Q2 * xt * px * (1.0 - self.par[0]**2)

                    elif ((v1 == 1) & (v2 == 1)):
                        hessian[:, v1, v2] = - 2.0 * Q2 * px**2 * self.scale**2

                    elif (((v1 == 1) & (v2 == 0)) | ((v1 == 0) & (v2 == 1))):
                        hessian[:, v1, v2] = - 2.0 * xt * Q2 * \
                            px * (1.0 - self.par[0]) * self.scale

                    elif ((v1 == 2) & (v2 == 2)):
                        hessian[:, v1, v2] = R2 - 3.0 * R4 * py**2

                    else:
                        hessian[:, v1, v2] = 0.0

        else:
            for v1 in range(0, self.nParInference):
                for v2 in range(0, self.nParInference):
                    if ((v1 == 0) & (v2 == 0)):
                        hessian[:, v1, v2] = - xt**2 * Q2

                    elif ((v1 == 1) & (v2 == 1)):
                        hessian[:, v1, v2] = (
                            Q2 - 3.0 * Q4 * px**2 - Q1) * self.scale**2

                    elif (((v1 == 1) & (v2 == 0)) | ((v1 == 0) & (v2 == 1))):
                        hessian[:, v1, v2] = - 2.0 * xt * Q3 * px * self.scale

                    elif ((v1 == 2) & (v2 == 2)):
                        hessian[:, v1, v2] = R2 - 3.0 * R4 * py**2

                    else:
                        hessian[:, v1, v2] = 0.0

        return(hessian)

    #=========================================================================
    # Define hard priors for the PMH sampler
    #=========================================================================
    def priorUniform(self):
        out = 1.0

        if(np.abs(self.par[0]) > 1.0):
            out = 0.0

        if(self.par[1] < 0.0):
            out = 0.0

        if(self.par[2] < 0.0):
            out = 0.0

        return(out)

    #=========================================================================
    # Define Jacobian and parameter transforms
    #=========================================================================
    def Jacobian(self):
        if (self.version == "tanhexp"):
            return np.log(1.0 - self.par[0]**2) + np.log(self.par[1])
        else:
            return 0.0

    def transform(self):
        if (self.version == "tanhexp"):
            self.par[0] = np.tanh(self.par[0])
            self.par[1] = np.exp(self.par[1]) * self.scale
        else:
            self.par[1] = self.par[1] * self.scale

    def invTransform(self):
        if (self.version == "tanhexp"):
            self.par[0] = np.arctanh(self.par[0])
            self.par[1] = np.log(self.par[1] / self.scale)
        else:
            self.par[1] = self.par[1] / self.scale

    #=========================================================================
    # Define standard methods for the model struct
    #=========================================================================

    # Standard operations on struct
    copyData = template_copyData
    storeParameters = template_storeParameters
    returnParameters = template_returnParameters

    # Standard data generation for this model
    generateData = template_generateData

    # Simple priors for this model
    prior = empty_prior
    dprior1 = empty_dprior1
    ddprior1 = empty_ddprior1
