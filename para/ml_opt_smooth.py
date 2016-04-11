##############################################################################
##############################################################################
# Routines for
# Maximum-likelihood inference using optimisation and the smooth particle filter
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy        as     np
import scipy        as     sp
import numdifftools as     nd
from   ml_helpers   import *

##############################################################################
# Main class
##############################################################################

class stMLoptSmooth(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # Give inital parameters and bounds on parameters
    initPar           = None;
    parBounds         = None;

    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval  = None;
    writeOutProgressToFile          = None;
    fileOutName                     = None;

    # Variables for constructing the file name when writing the output to file
    filePrefix        = None;
    dataset           = None;

    ##########################################################################
    # Wrapper for write to file
    ##########################################################################
    def writeToFile(self,sm,fileOutName=None):
        writeToFile_helper(self,sm,fileOutName,noLLests=True);

    ##########################################################################
    # Main BFGS routine with bounds for zero-order optimisation
    ##########################################################################

    def bfgs_bounded(self,sm,sys,thSys):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set initial settings
        self.optMethod  = "bfgs_bounded"
        self.nPars      = thSys.nParInference;
        self.filePrefix = thSys.filePrefix;
        th0             = self.initPar[0:self.nPars];
        mybounds        = self.parBounds;
        self.th         = th0;
        self.nFuncEval  = 1;

        # Ugly workaround for Hessian estimate as args cannot be passed
        self.sm    = sm;
        self.sys   = sys;
        self.thSys = thSys;

        # Check algorithm settings and set to default if needed
        setSettings(self,"smoothOpt");

        #=====================================================================
        # Run BFGS with particle filter
        #=====================================================================
        res    = sp.optimize.fmin_l_bfgs_b(self.estimateLikelihood, x0=th0, bounds=mybounds, approx_grad=True, callback=self.saveTrace)

        #=====================================================================
        # Write output
        #=====================================================================
        self.thhat  = res[0];

    ##########################################################################
    # Main BFGS routine with bounds for zero-order optimisation
    ##########################################################################

    def bfgs(self,sm,sys,thSys):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set initial settings
        self.optMethod  = "bfgs"
        self.nPars      = thSys.nParInference;
        self.filePrefix = thSys.filePrefix;
        th0             = self.initPar[0:self.nPars];
        self.th         = th0;
        self.nFuncEval  = 1;

        # Ugly workaround for Hessian estimate as args cannot be passed
        self.sm    = sm;
        self.sys   = sys;
        self.thSys = thSys;

        # Check algorithm settings and set to default if needed
        setSettings(self,"bfgsOpt");

        #=====================================================================
        # Run BFGS with particle filter
        #=====================================================================
        res = sp.optimize.minimize(self.estimateLikelihood, x0=th0, method="BFGS", jac=None, callback=self.saveTrace, options={'gtol': 1e-6, 'disp': True, 'eps': 1e-1})

        #=====================================================================
        # Write output
        #=====================================================================
        self.res                = res;
        self.thhat              = res.x;
        self.invHessianEstimate = res.hess_inv;

    ##########################################################################
    # Helper: estimate the Hessian
    ##########################################################################
    def estimateHessian(self):
        hes  = nd.Hessian( self.estimateLikelihood )
        self.hessianEstimate    = hes(self.thhat);
        self.invHessianEstimate = np.linalg.pinv(self.hessianEstimate)

    ##########################################################################
    # Helper: save the trace of the algorithm
    ##########################################################################
    def saveTrace( self, Xi ):
        self.nFuncEval += 1;
        self.th         = np.vstack((self.th,Xi));

    ##########################################################################
    # Helper: calculate the log-likelihood
    ##########################################################################
    def estimateLikelihood(self,pp):
        sys = self.sys;
        th  = self.thSys;
        sm  = self.sm;

        th.storeParameters(pp,sys);
        th.transform();
        sm.filter(th);

        parm = ["%.4f" % v for v in pp];
        print("Current parameters: " + str(parm) + " with llest: " + str(-sm.ll) )
        return( sm.ll );