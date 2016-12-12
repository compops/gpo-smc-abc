##############################################################################
##############################################################################
# Routines for
# Parameter inference using GPO
#
# Copyright (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

import numpy as np
import scipy as sp
import numdifftools as nd
import matplotlib.pylab as plt

import GPy
import pandas

from DIRECT import solve
from pyDOE import lhs
from sobol import i4_sobol
from ml_helpers import *

##############################################################################
# Main class
##############################################################################


class stGPO(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # Should the algorithm write out its status during each iteration
    verbose = None
    EstimateHessianEveryIteration = False

    # Variables for constructing the file name when writing the output to file
    filePrefix = None
    dataset = None

    # Exit conditions (the diff in norm of theta or maximum no iterations)
    tolLevel = None
    maxIter = None

    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval = None
    writeOutProgressToFile = None
    fileOutName = None

    # GPO settings
    epsilon = None
    preIter = None
    upperBounds = None
    lowerBounds = None
    EstimateHyperparametersInterval = None
    AQfunction = None
    EstimateThHatEveryIteration = None
    preSamplingMethod = None

    # Give step size and inital parameters
    stepSize = None
    initPar = None

    # Input design
    inputDesignMethod = None

    # Jittering of parameters
    jitterParameters = None
    jitteringCovariance = None

    ##########################################################################
    # Wrappers for different versions of the algorithm
    ##########################################################################

    def ml(self, sm, sys, th):
        self.optType = "MLparameterEstimation"
        self.optMethod = "gpo_ml"
        self.gpo(sm, sys, th)

    def bayes(self, sm, sys, th):
        self.optType = "MAPparameterEstimation"
        self.optMethod = "gpo_map"
        self.gpo(sm, sys, th)

    ##########################################################################
    # Gaussian process optimisation (GPO)
    ##########################################################################
    def gpo(self, sm, sys, thSys):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set initial settings
        sm.calcGradientFlag = False
        sm.calcHessianFlag = False
        self.nPars = thSys.nParInference
        self.filePrefix = thSys.filePrefix
        runNextIter = True
        self.iter = 0

        # Check algorithm settings and set to default if needed
        setSettings(self, "gpo")

        # Make a grid to evaluate the EI on
        l = np.array(self.lowerBounds[0:thSys.nParInference], dtype=np.float64)
        u = np.array(self.upperBounds[0:thSys.nParInference], dtype=np.float64)

        # Allocate vectors
        AQ = np.zeros((self.maxIter + 1, 1))
        mumax = np.zeros((self.maxIter + 1))
        thp = np.zeros((self.maxIter + 1, self.nPars))
        obp = np.zeros((self.maxIter + 1, 1))
        thhat = np.zeros((self.maxIter, self.nPars))
        thhatHessian = np.zeros((self.maxIter, self.nPars, self.nPars))
        obmax = np.zeros((self.maxIter + 1, 1))
        hyperParams = np.zeros((self.maxIter, 3 + self.nPars))
        xhatf = np.zeros((self.maxIter + 1, sys.T))

        #=====================================================================
        # Pre-run using random sampling to estimate hyperparameters
        #=====================================================================

        # Pre allocate vectors
        thPre = np.zeros((self.preIter, self.nPars))
        obPre = np.zeros((self.preIter, 1))

        #=====================================================================
        # Main loop
        #=====================================================================

        # Pre-compute hypercube points if required
        if (self.preSamplingMethod == "latinHyperCube"):
            lhd = lhs(self.nPars, samples=self.preIter)

        for kk in range(0, self.preIter):

            # Sampling parameters using uniform sampling or Latin hypercubes
            if (self.preSamplingMethod == "latinHyperCube"):
                # Sample parameters using a Latin hypercube over the parameter
                # bounds
                thPre[kk, :] = l + (u - l) * lhd[kk, :]
            elif (self.preSamplingMethod == "sobol"):
                # Sample parameters using a Sobol sequence over the parameter
                # bounds
                thPre[kk, :] = l + (u - l) * i4_sobol(self.nPars, 100 + kk)[0]
            else:
                # Draw parameters uniform over the parameter bounds
                thPre[kk, :] = l + (u - l) * np.random.random(self.nPars)

            # Evaluate the objective function in the parameters
            thSys.storeParameters(thPre[kk, :], sys)
            obPre[kk], tmp1 = self.evaluateObjectiveFunction(sm, sys, thSys)

            # Transform and save the parameters
            thSys.transform()
            thPre[kk, :] = thSys.returnParameters()[0:thSys.nParInference]

            # Write out progress if requested
            if (self.verbose):
                print("gpo: Pre-iteration: " + str(kk) + " of " + str(self.preIter) + " completed, sampled " +
                      str(np.round(thPre[kk, :], 3)) + " with " + str(np.round(obPre[kk], 2)) + ".")

        #=====================================================================
        # Fit the GP regression
        #=====================================================================

        # Remove nan values for the objective function
        idxNotNaN = ~np.isnan(obPre)
        thPre = thPre[(idxNotNaN).any(axis=1)]
        obPre = obPre[(idxNotNaN).any(axis=1)]

        # Specify the kernel ( Matern52 with ARD plus bias kernel to compensate
        # for non-zero mean )
        kernel = GPy.kern.Matern52(
            input_dim=self.nPars, ARD=True) + GPy.kern.Bias(input_dim=self.nPars)

        # Normalize the objective function evaluations
        ynorm = (obPre - np.mean(obPre)) / np.sqrt(np.var(obPre))

        # Create the model object
        m = GPy.models.GPRegression(thPre, ynorm, kernel, normalizer=False)

        #=====================================================================
        # Update hyperparameters
        #=====================================================================

        # Set constraints on hyperparameters
        m.Gaussian_noise.variance.constrain_bounded(0.01, 10.0)
        m.kern.Mat52.lengthscale.constrain_bounded(0.01, 10.0)
        m.kern.Mat52.variance.constrain_bounded(0.01, 25.0)

        # Run empirical Bayes to estimate the hyperparameters
        m.optimize('bfgs', max_iters=200)
        m.optimize_restarts(num_restarts=10, robust=True)
        self.GaussianNoiseVariance = np.array(
            m.Gaussian_noise.variance, copy=True)

        #=====================================================================
        # Write to output
        #=====================================================================

        self.thPre = thPre
        self.obPre = obPre
        self.m = m

        #=====================================================================
        # Main loop
        #=====================================================================

        # Save the initial parameters
        thSys.storeParameters(self.initPar, sys)
        thp[self.iter, :] = thSys.returnParameters()
        thSys.transform()

        while (runNextIter):

            # Store the parameter
            thSys.storeParameters(thp[self.iter, :], sys)
            thSys.transform()

            #------------------------------------------------------------------
            # Evalute the objective function
            #------------------------------------------------------------------
            obp[self.iter], xhatf[self.iter,
                                  :] = self.evaluateObjectiveFunction(sm, sys, thSys)

            # Collect the sampled data (if the objective is finite)
            idxNotNaN = ~np.isnan(obp[range(self.iter), :])
            x = np.vstack((thPre, thp[(idxNotNaN).any(axis=1)]))
            y = np.vstack((obPre, obp[(idxNotNaN).any(axis=1)]))

            #------------------------------------------------------------------
            # Fit the GP to the sampled data
            #------------------------------------------------------------------
            ynorm = (y - np.mean(y)) / np.sqrt(np.var(y))
            self.ynormMean = np.mean(y)
            self.ynormVar = np.var(y)

            m = GPy.models.GPRegression(x, ynorm, kernel, normalizer=False)

            #------------------------------------------------------------------
            # Re-estimate the hyperparameters
            #------------------------------------------------------------------
            if (np.remainder(self.iter + 1, self.EstimateHyperparametersInterval) == 0):

                # Set constraints on hyperparameters
                m.Gaussian_noise.variance.constrain_bounded(0.01, 10.0)
                m.kern.Mat52.lengthscale.constrain_bounded(0.01, 10.0)
                m.kern.Mat52.variance.constrain_bounded(0.01, 25.0)

                # Run empirical Bayes to estimate the hyperparameters
                m.optimize('bfgs', max_iters=200)
                m.optimize_restarts(num_restarts=10, robust=True)

                # Save the current noise variance
                self.GaussianNoiseVariance = np.array(
                    m.Gaussian_noise.variance, copy=True)

            else:

                # Overload current noise estimate (sets to 1.0 every time we
                # add data otherwise)
                m.Gaussian_noise.variance = self.GaussianNoiseVariance

            # Save all the hyperparameters
            hyperParams[self.iter, 0] = np.array(
                m.Gaussian_noise.variance, copy=True)
            hyperParams[self.iter, 1] = np.array(
                m.kern.bias.variance, copy=True)
            hyperParams[self.iter, 2] = np.array(
                m.kern.Mat52.variance, copy=True)
            hyperParams[self.iter, range(
                3, 3 + self.nPars)] = np.array(m.kern.Mat52.lengthscale, copy=True)

            #------------------------------------------------------------------
            # Find the maximum expected value of the GP over the sampled parameters
            #------------------------------------------------------------------
            Mup, ys2 = m.predict(x)
            mumax[self.iter] = np.max(Mup)

            #------------------------------------------------------------------
            # Compute the next point in which to sample the posterior
            #------------------------------------------------------------------

            # Optimize the AQ function
            aqThMax, aqMax, ierror = solve(self.AQfunction, l, u, user_data=(
                m, mumax[self.iter], self.epsilon), maxf=1000, maxT=1000)

            # Jitter the parameter estimates
            if (self.jitterParameters == True):
                flag = 0.0

                while (flag == 0.0):
                    z = np.random.multivariate_normal(np.zeros(self.nPars), self.jitteringCovariance[
                                                      range(self.nPars), :][:, range(self.nPars)])
                    flag = self.checkProposedParameters(aqThMax + z)

                thSys.storeParameters(aqThMax + z, sys)
                aqThMax += z

            # Set the new point and save the estimate of the AQ
            thp[self.iter + 1, :] = aqThMax
            AQ[self.iter + 1] = -aqMax

            # Update counter
            self.iter += 1

            #------------------------------------------------------------------
            # Check exit conditions
            #------------------------------------------------------------------

            # AQ function criteria
            if (AQ[self.iter] < self.tolLevel):
                print("GPO: reaches tolLevel, so exiting...")
                runNextIter = False

            # Max iteration criteria
            if (self.iter == self.maxIter):
                print("GPO: reaches maxIter, so exiting...")
                runNextIter = False

            #------------------------------------------------------------------
            # Estimate the current parameters by maximizing the GP
            #------------------------------------------------------------------
            if ((self.EstimateThHatEveryIteration == True) | (runNextIter == False)):
                thhatCurrent, obmaxCurrent, ierror = solve(
                    self.MUeval, l, u, user_data=m, algmethod=1, maxf=1000, maxT=1000)

                thhat[self.iter - 1, :] = thhatCurrent
                obmax[self.iter - 1, :] = obmaxCurrent

                print((thhatCurrent, obmaxCurrent))

                if (self.EstimateHessianEveryIteration == True):
                    self.estimateHessian(thhatCurrent)
                    thhatHessian[self.iter - 1, :, :] = self.invHessianEstimate

            #------------------------------------------------------------------
            # Print output to console
            #------------------------------------------------------------------
            if (self.verbose):
                if (self.EstimateThHatEveryIteration == True):
                    parm = ["%.4f" % v for v in thhat[self.iter - 1, :]]
                    print(
                        "##############################################################################################")
                    print("Iteration: " + str(self.iter) + " with current parameters: " +
                          str(parm) + " and AQ: " + str(np.round(AQ[self.iter], 2)))
                    print(
                        "##############################################################################################")
                else:
                    parm = ["%.4f" % v for v in thp[self.iter - 1, :]]
                    print(
                        "##############################################################################################")
                    print("Iteration: " + str(self.iter) + " sampled objective function at parameters: " +
                          str(parm) + " with value: " + str(np.round(obp[self.iter - 1], 2)))
                    print(
                        "##############################################################################################")

        #=====================================================================
        # Generate output
        #=====================================================================
        tmp = range(self.iter - 1)
        self.ob = obmax[tmp]
        self.th = thhat[tmp, :]
        self.thhat = thhat[self.iter - 1, :]
        self.thHessian = thhatHessian
        self.thhatHessian = thhatHessian[self.iter - 1, :, :]
        self.aq = AQ[range(self.iter)]
        self.obp = obp[tmp]
        self.thp = thp[range(self.iter), :]
        self.m = m
        self.x = x
        self.y = y
        self.xhatf = xhatf
        self.ynorm = ynorm
        self.hp = hyperParams

    ##########################################################################
    # Evaluate the surrogate function to find its maximum
    ##########################################################################

    def MUeval(self, x, m):
        # Predict over the surrogate
        Mup, Mus = m.predict(np.array(x).reshape((1, len(x))))

        # Compensate for normalization
        Mup = (Mup * np.sqrt(self.ynormVar)) + self.ynormMean

        return -Mup, 0

    ##########################################################################
    # Expected Improvement (EI) aquisition rule
    ##########################################################################

    def aq_ei(self, x, user_data):
        m = user_data[0]
        obmax = user_data[1]
        epsilon = user_data[2]

        # Find the predicted value in x
        #Mup, Mus = m.predict( np.matrix(x) )
        Mup, Mus = m.predict(np.array(x).reshape((1, len(x))))

        # Calculate auxillary quantites
        s = np.sqrt(Mus)
        yres = Mup - obmax - epsilon
        ynorm = (yres / s) * (s > 0)

        # Compute the EI and negate it
        ei = yres * sp.stats.norm.cdf(ynorm) + s * sp.stats.norm.pdf(ynorm)
        ei = np.max((ei, 0))
        return -ei, 0

    ##########################################################################
    # Probability of Improvement (PI) aquisition rule
    ##########################################################################

    def aq_pi(self, x, user_data):
        m = user_data[0]
        obmax = user_data[1]
        epsilon = user_data[2]

        # Find the predicted value in x
        #Mup, Mus = m.predict( np.matrix(x) )
        Mup, Mus = m.predict(np.array(x).reshape((1, len(x))))

        # Calculate auxillary quantites
        s = np.sqrt(Mus)
        yres = Mup - obmax - epsilon
        ynorm = (yres / s) * (s > 0)

        # Compute the EI and negate it
        pi = sp.stats.norm.cdf(ynorm)
        return -pi, 0

    ##########################################################################
    # Upper Confidence Bound (UCB) aquisition rule
    ##########################################################################

    def aq_ubc(self, x, user_data):
        m = user_data[0]
        epsilon = user_data[2]

        # Find the predicted value in x
        #Mup, Mus = m.predict( np.matrix(x) )
        Mup, Mus = m.predict(np.array(x).reshape((1, len(x))))

        # Calculate auxillary quantites
        s = np.sqrt(Mus)

        # Compute the EI and negate it
        ucb = Mup + epsilon * s
        return -ucb, 0

    def evaluateObjectiveFunction(self, sm, sys, thSys):

        # Evalute the objective function
        if (self.optType == "MLparameterEstimation"):

            # Sample the log-likelihood in the proposed parameters
            sm.filter(thSys)
            ob = sm.ll
            xhatf = sm.xhatf[:, 0]

        elif (self.optType == "MAPparameterEstimation"):

            # Sample the log-target in the proposed parameters
            sm.filter(thSys)
            ob = sm.ll + thSys.prior()
            xhatf = sm.xhatf[:, 0]

        elif(self.optType == "InputDesign"):

            # Sample the log-likelihood in the proposed parameters
            ob = self.inputDesignMethod(sm, thSys)
            xhatf = 0.0

        return ob, xhatf

    ##########################################################################
    # Helper: estimate the Hessian
    ##########################################################################
    def estimateHessian(self, x=None, version="standard"):

        # Use current estimate if parameters are not supplied
        if (x == None):
            x = self.thhat

        # Create the numdiff object
        hes = nd.Hessian(self.evaluateSurrogate)
        foo = -hes(x)

        # Regularise the Hessian if not PSD
#        if ~isPSD( foo ):
#            mineigv = np.min( np.linalg.eig( foo)[0] )
#            foo = foo - 2.0 * mineigv * np.eye( self.nPars )

        # Evaluate hessian and inverse hessian
        if (version == "standard"):
            self.hessianEstimate = foo
            self.invHessianEstimate = np.linalg.pinv(self.hessianEstimate)
        else:
            self.hessianEstimateTransformed = foo
            self.invHessianEstimateTransformed = np.linalg.pinv(
                self.hessianEstimateTransformed)

    ##########################################################################
    # Helper: estimate the Hessian in transformed domain
    ##########################################################################
    def estimateHessianTransformed(self, sys, th, transformation="standard"):

        # Collect the sampled data (if the objective is finite)
        idxNotNaN = ~np.isnan(self.obp[range(self.maxIter - 1), :])
        x = np.vstack((self.thPre, self.thp[(idxNotNaN).any(axis=1)]))
        y = np.vstack((self.obPre, self.obp[(idxNotNaN).any(axis=1)]))

        xt = np.zeros((x.shape[0], x.shape[1]))

        th.version = transformation

        for ii in range(x.shape[0]):
            th.storeParameters(x[ii, :], sys)
            th.invTransform()
            xt[ii, :] = th.returnParameters()

        #------------------------------------------------------------------
        # Fit the GP to the sampled data
        #------------------------------------------------------------------
        ynorm = (y - np.mean(y)) / np.sqrt(np.var(y))
        ynormMean = np.mean(y)
        ynormVar = np.var(y)

        kernel = GPy.kern.Matern52(
            input_dim=self.nPars, ARD=True) + GPy.kern.Bias(input_dim=self.nPars)
        m = GPy.models.GPRegression(xt, ynorm, kernel, normalizer=False)

        #------------------------------------------------------------------
        # Re-estimate the hyperparameters
        #------------------------------------------------------------------

        # Set constraints on hyperparameters
        m.Gaussian_noise.variance.constrain_bounded(0.01, 10.0)
        m.kern.Mat52.lengthscale.constrain_bounded(0.01, 10.0)
        m.kern.Mat52.variance.constrain_bounded(0.01, 25.0)

        # Run empirical Bayes to estimate the hyperparameters
        m.optimize('bfgs', max_iters=200)
        m.optimize_restarts(num_restarts=10, robust=True)

        self.m = m
        self.estimateHessian(version="transformed")

    ##########################################################################
    # Helper: calculate the log-likelihood
    ##########################################################################
    def evaluateSurrogate(self, x):
        model = self.m

        # Predict
        Mup, Mus = model.predict(np.array(x).reshape((1, len(x))))

        # Compensate for normalization
        Mup = (Mup * np.sqrt(self.ynormVar)) + self.ynormMean

        parm = ["%.4f" % v for v in x]
        print("Current parameters: " + str(parm) +
              " with objective: " + str(np.round(Mup, 3)))

        return(Mup)

    ##########################################################################
    # Helper: check that proposed parameters are within bounds
    ##########################################################################
    def checkProposedParameters(self, par):

        for ii in range(self.nPars):
            if (par[ii] < self.lowerBounds[ii]):
                return 0.0

            if (par[ii] > self.upperBounds[ii]):
                return 0.0

        return 1.0

    ##########################################################################
    # Helper: compile the results and write to file
    ##########################################################################
    def writeToFile(self, sm=None, fileOutName=None):

        # Construct the columns labels
        columnlabels = [None] * (2 * self.nPars + 3)

        for ii in range(0, self.nPars):
            columnlabels[ii] = "thhat" + str(ii)
            columnlabels[ii + self.nPars] = "thp" + str(ii)

        columnlabels[2 * self.nPars] = "ob"
        columnlabels[2 * self.nPars + 1] = "obp"
        columnlabels[2 * self.nPars + 2] = "aq"

        # Compile the results for output
        tmp = range(0, self.iter - 1)
        out = np.hstack((self.th[tmp, :], self.thp[tmp, :], self.ob[
                        tmp], self.obp[tmp], self.aq[tmp]))

        # Write out the results to file
        fileOut = pandas.DataFrame(out, columns=columnlabels)

        if (fileOutName == None):
            if (sm.filterType == "kf"):
                fileOutName = 'results/' + str(self.filePrefix) + '/' + str(
                    self.optMethod) + '_' + sm.filterType + '/' + str(self.dataset) + '.csv'
            elif hasattr(sm, 'filterType'):
                fileOutName = 'results/' + str(self.filePrefix) + '/' + str(
                    self.optMethod) + '_' + sm.filterType + '_N' + str(sm.nPart) + '/' + str(self.dataset) + '.csv'
            else:
                fileOutName = 'gpo-output.csv'

        ensure_dir(fileOutName)
        fileOut.to_csv(fileOutName)

        print("writeToFile_helper: wrote results to file: " + fileOutName)

    ##########################################################################
    # Helper: plot the marginal predictive distributions
    ##########################################################################
    def plotPredictiveMarginals(self, plotGridSize=0.001, matrixPlotSide=(2, 2)):

        plt.figure(177)

        # For each parameter
        for ii in range(self.nPars):
            plt.subplot(matrixPlotSide[0], matrixPlotSide[1], ii + 1)

            # Form grid for parameter
            grid1 = np.arange(self.lowerBounds[ii], self.upperBounds[
                              ii], plotGridSize)
            grid1 = grid1.reshape((len(grid1), 1))

            # Stack grids together, fix all other parameters to thhat
            grid2 = np.zeros((len(grid1), 1))

            for kk in range(self.nPars):
                if (kk == ii):
                    grid2 = np.hstack((grid2, grid1))
                else:
                    grid2 = np.hstack(
                        (grid2, np.ones((len(grid1), 1)) * self.thhat[kk]))

            # Remove the first column
            grid2 = grid2[:, range(1, self.nPars + 1)]

            # Make prediction
            ypred, s2 = self.m.predict(grid2)

            # Compensate for normalization
            ypred = (ypred * np.sqrt(self.ynormVar)) + self.ynormMean
            s2 = s2 * self.ynormVar

            # Plot 95% CIs
            plt.plot(grid1, ypred + 1.96 * np.sqrt(s2),
                     linewidth=0.5, color='k')
            plt.plot(grid1, ypred - 1.96 * np.sqrt(s2),
                     linewidth=0.5, color='k')

            # Plot mean function and data
            plt.plot(grid1, ypred)
            plt.plot(self.x[:, ii], self.y, 'k.')

            # Plot thhat
            plt.axvline(self.thhat[ii], linewidth=2, color='k')

########################################################################
# End of file
########################################################################
