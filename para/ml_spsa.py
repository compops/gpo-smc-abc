##############################################################################
##############################################################################
# Routines for
# Maximum-likelihood inference using SPSA
#
# Copyright (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

import numpy as np
from ml_helpers import *

##############################################################################
# Main class
##############################################################################


class stMLspsa(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # Should the algorithm write out its status during each iteration
    verbose = None

    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval = None
    writeOutProgressToFile = None
    fileOutName = None

    # Variables for constructing the file name when writing the output to file
    filePrefix = None
    dataset = None

    # Exit conditions (the diff in norm of theta or maximum no iterations)
    tolLevel = None
    noisyTolLevel = None
    maxIter = None

    # SPSA settings
    alpha = None
    A = None
    a = None
    c = None
    gamma = None

    # Give step size and inital parameters
    stepSize = None
    initPar = None

    ##########################################################################
    # Wrappers for different versions of the algorithm
    ##########################################################################

    def ml(self, sm, sys, th):
        self.optType = "MLparameterEstimation"
        self.optMethod = "spsa_ml"
        self.template_spsa(sm, sys, th)

    def bayes(self, sm, sys, th):
        self.optType = "MAPparameterEstimation"
        self.optMethod = "spsa_map"
        self.template_spsa(sm, sys, th)

    ##########################################################################
    # Simulanteous perturbation stochastic approxmation (SPSA)
    ##########################################################################

    def template_spsa(self, sm, sys, thSys):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set initial settings
        sm.calcGradientFlag = False
        sm.calcHessianFlag = False
        self.nPars = thSys.nParInference
        self.filePrefix = thSys.filePrefix
        runNextIter = True
        self.iter = 1

        # Check algorithm settings and set to default if needed
        setSettings(self, "spsa")

        # Allocate vectors
        aa = np.zeros((self.maxIter, 1))
        cc = np.zeros((self.maxIter, 1))
        delta = np.zeros((self.maxIter, self.nPars))
        gg = np.zeros((self.maxIter, self.nPars))
        ll = np.zeros((self.maxIter, 1))
        llp = np.zeros((self.maxIter, 1))
        llm = np.zeros((self.maxIter, 1))
        th = np.zeros((self.maxIter, self.nPars))
        thp = np.zeros((self.maxIter, self.nPars))
        thm = np.zeros((self.maxIter, self.nPars))
        llDiff = np.zeros((self.maxIter, 1))

        thSys.storeParameters(self.initPar, sys)
        th[0, :] = thSys.returnParameters()
        thSys.transform()

        #=====================================================================
        # Main loop
        #=====================================================================
        while (runNextIter):

            # Calculate a_k if self.a is given
            if (self.a != None):
                aa[self.iter] = self.a / (self.A + self.iter + 1)**self.alpha

            # Calculate c_k if self.c is given otherwise calculate SD of the
            # log-likelihood estimate to give suggestion
            if (self.c != None):
                cc[self.iter] = self.c / (self.iter + 1)**self.gamma
            else:
                tmp = np.zeros(50)
                for ii in range(50):
                    sm.filter(thSys)

                    if (self.optMethod == "spsa_map"):
                        tmp[ii] = sm.ll + thSys.prior()
                        # print((thSys.par,sm.ll,thSys.prior()))
                    else:
                        tmp[ii] = sm.ll

                chat = np.sqrt(np.var(tmp))
                print("ml-spsa: missing parameter c (should be selected to be around " +
                      str(chat) + " as the StDev of the LL estimate.")

                cc[self.iter] = chat / (self.iter + 1)**self.gamma

            # Calculate Delta_k
            delta[self.iter, :] = 2 * \
                np.round(np.random.uniform(size=self.nPars)) - 1.0

            # Perform update on evaluation variables
            thp[self.iter, :] = th[self.iter - 1, :] + \
                cc[self.iter] * delta[self.iter, :]
            thm[self.iter, :] = th[self.iter - 1, :] - \
                cc[self.iter] * delta[self.iter, :]

            # Compute the log-likelihood in thp
            thSys.storeParameters(thp[self.iter, :], sys)
            sm.filter(thSys)

            if (self.optMethod == "spsa_map"):
                llp[self.iter] = sm.ll + thSys.prior()
                # print((thSys.par,sm.ll,thSys.prior()))
            else:
                llp[self.iter] = sm.ll

            # Compute the log-likelihood in thm
            thSys.storeParameters(thm[self.iter, :], sys)
            sm.filter(thSys)

            if (self.optMethod == "spsa_map"):
                llm[self.iter] = sm.ll + thSys.prior()
                # print((thSys.par,sm.ll,thSys.prior()))
            else:
                llm[self.iter] = sm.ll

            # Estimate gradient
            gg[self.iter, :] = (llp[self.iter] - llm[self.iter]) / \
                (delta[self.iter, :] * 2.0 * cc[self.iter])

            # Check if aa is set or give suggestions
            if (self.a == None):
                ahat = (self.A + 1)**self.alpha * gg[self.iter, :]
                raise NameError("ml-spsa: missing parameter a (should be selected so that a times " +
                                str(ahat) + " is the smallest allowed step.")

            # Perform update
            th[self.iter] = th[self.iter - 1, :] + \
                aa[self.iter] * gg[self.iter, :]
            thSys.storeParameters(th[self.iter, :], sys)
            thSys.transform()
            sm.filter(thSys)

            if (self.optMethod == "spsa_map"):
                ll[self.iter] = sm.ll + thSys.prior()
            else:
                ll[self.iter] = sm.ll

            if (self.iter == 1):
                llDiff[self.iter] = 10000.0
            else:
                llDiff[self.iter] = np.abs(ll[self.iter] - ll[self.iter - 1])

            # Compute exit criteria
            if (llDiff[self.iter] < self.tolLevel):
                runNextIter = False
            elif (self.iter >= (self.maxIter - 1)):
                runNextIter = False

            # Print output to console
            if (self.verbose):
                parm = ["%.4f" % v for v in th[self.iter - 1]]
                print("Iteration: " + str(self.iter) + " with current parameters: " +
                      str(parm) + " and lldiff: " + str(llDiff[self.iter - 1]))

            # Update iteration number
            self.iter += 1

        #=====================================================================
        # Generate output
        #=====================================================================
        tmp = range(0, self.iter - 1)
        self.ll = ll[tmp]
        self.llDiff = llDiff[tmp]
        self.th = th[tmp, :]
        self.thhat = th[self.iter - 1, :]

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
