##############################################################################
##############################################################################
# Routines for
# Maximum-likelihood inference using optimisation
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy      as     np
from   ml_helpers import *

##############################################################################
# Main class
##############################################################################

class stMLopt(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # Should the algorithm write out its status during each iteration
    verbose           = None;

    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval  = None;
    writeOutProgressToFile          = None;
    fileOutName                     = None;

    # Variables for constructing the file name when writing the output to file
    filePrefix        = None;
    dataset           = None;

    # Exit conditions (the diff in norm of theta or maximum no iterations)
    tolLevel          = None;
    noisyTolLevel     = None;
    maxIter           = None;

    # Adapt step sizes as \gamma kk**(-\alpha)
    adaptStepSize     = None;
    adaptStepSizeFrom = None;
    alpha             = None;
    gamma             = None;

    # Give step size and inital parameters
    stepSize          = None;
    initPar           = None;

    # Initial diagonal Hessian matrix for noisy quasi-Newton algorithm
    # Make it small to avoid a very big first step.
    epsilon           = None;

    # Forgetting factor for new Hessian contribution for noisy quasi-Newton
    # Make it quite small to combat noise.
    cc                = None;

    # Step size for noisy quasi-Newton given by tau/(tau+kk) * eta
    tau               = None;
    eta               = None;

    # Regularisation factor of Hessian estimate for noisy quasi-Newton
    # Make it bigger if the estimate "explodes"
    lam               = None;

    # Memory length in L-BFGS algorithm
    memoryLength      = None;

    ##########################################################################
    # Wrapper for gradient based optimisation
    ##########################################################################
    def gradientbased(self,sm,sys,thSys):
        self.optMethod  = "gradientbased"
        self.direct_opt(sm,sys,thSys,False);

    ##########################################################################
    # Wrapper for Newton based optimisation
    ##########################################################################
    def newton(self,sm,sys,thSys):
        self.optMethod = "newton"
        self.direct_opt(sm,sys,thSys,True);

    ##########################################################################
    # Wrapper for noisy quasi-Newton based optimisation
    ##########################################################################
    def noisyQuasiNewton(self,sm,sys,thSys):
        self.optMethod    = "noisyQuasiNewton"
        self.memoryLength = sys.T
        self.bfgs_opt(sm,sys,thSys);

    ##########################################################################
    # Wrapper for noisy quasi-Newton based optimisation
    ##########################################################################
    def noisyLimitedMemoryQuasiNewton(self,sm,sys,thSys):
        self.optMethod   = "noisyLimitedMemoryQuasiNewton"
        self.lbgsVersion = "standard"
        self.cc          = 1.0;
        self.bfgs_opt(sm,sys,thSys);

    ##########################################################################
    # Wrapper for write to file
    ##########################################################################
    def writeToFile(self,sm,fileOutName=None):
        writeToFile_helper(self,sm,fileOutName);

    ##########################################################################
    # Main routine for direct optimisation
    ##########################################################################
    def direct_opt(self,sm,sys,thSys,useHessian):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set initial settings
        sm.calcGradientFlag = True;
        if ( useHessian == False ):
            sm.calcHessianFlag = False;
        else:
            if ( ( sm.calcHessianFlag == None ) | ( sm.calcHessianFlag == False ) ):
                print("ml-newton: No Hessian estimator given, defaulting to Segel-Weinstein identity");
                sm.calcHessianFlag  = "segelweinstein";

        self.nPars      = thSys.nParInference;
        self.filePrefix = thSys.filePrefix;
        runNextIter     = True;
        self.iter       = 1;

        # Check algorithm settings and set to default if needed
        setSettings(self,"simpleOpt");

        # Allocate vectors
        step        = np.zeros((self.maxIter,1))
        ll          = np.zeros((self.maxIter,1))
        llDiff      = np.zeros((self.maxIter,1))
        th          = np.zeros((self.maxIter,thSys.nParInference))
        thDiff      = np.zeros((self.maxIter,thSys.nParInference))
        gradient    = np.zeros((self.maxIter,thSys.nParInference))
        hessian     = np.zeros((self.maxIter,thSys.nParInference,thSys.nParInference))

        # Store the initial parameters
        thSys.storeParameters(self.initPar,sys);
        th[0,:]  = thSys.returnParameters();
        thSys.transform();

        # Compute the initial gradient
        sm.smoother(thSys);
        gradient[ 0, : ] = sm.gradient;
        ll[0]            = sm.ll;
        if ( useHessian ):
            hessian[0,:,:] = sm.hessian;

        # Set initial step size (if adaptive step size is used)
        if ( self.adaptStepSize == True ):
            step[0] = self.stepSize;

        #=====================================================================
        # Main loop
        #=====================================================================
        while ( runNextIter  ):

            # Adapt step size
            if ( ( self.adaptStepSize ) & ( self.iter > self.adaptStepSizeFrom ) ):
                step[self.iter,:] = self.gamma * self.iter**(-self.alpha);
            else:
                step[self.iter,:] = step[self.iter-1,:];

            if ( self.adaptStepSize == False ):
                step[self.iter,:] = self.stepSize;

            # Perform update
            if ( useHessian ):
                th[self.iter,:] = th[self.iter-1,:] + step[self.iter,:] * np.dot( np.linalg.pinv( hessian[self.iter-1,:] ) , gradient[self.iter-1,:] );
            else:
                th[self.iter,:] = th[self.iter-1,:] + step[self.iter,:] * gradient[self.iter-1,:];

            thDiff[self.iter] = th[self.iter] - th[self.iter-1,:];
            thSys.storeParameters(th[self.iter,:],sys);
            thSys.transform();

            # Compute the gradient
            sm.smoother(thSys);
            gradient[self.iter,:] = sm.gradient;
            ll[self.iter]         = sm.ll;
            if ( useHessian ):
                hessian[self.iter,:,:] = sm.hessian;

            # Calculate the difference in log-likelihood and check exit condition
            llDiff[self.iter] = np.abs( ll[self.iter] - ll[self.iter-1] );

            if (self.noisyTolLevel == False):
                if ( llDiff[self.iter] < self.tolLevel ):
                    runNextIter = False;
            else:
                if ( self.iter > self.noisyTolLevel ):
                    # Check if the condition has been fulfilled during the last iterations
                    if (np.sum( llDiff[range( int(self.iter-(self.noisyTolLevel-1)),self.iter+1)] < self.tolLevel ) == self.noisyTolLevel):
                        runNextIter = False;

            # Update iteration number and check exit condition
            self.iter += 1;

            if ( self.iter == self.maxIter ):
                runNextIter = False;

            # Print output to console
            if (self.verbose ):
                parm = ["%.4f" % v for v in th[self.iter-1]];
                print("Iteration: " + str(self.iter) + " with current parameters: " + str(parm) + " and lldiff: " + str(llDiff[self.iter-1]) )

        #=====================================================================
        # Compile output
        #=====================================================================
        tmp         = range(0,self.iter-1);
        self.th     = th[tmp,:];
        self.step   = step[tmp,:]
        self.thDiff = thDiff[tmp,:]
        self.llDiff = llDiff[tmp,:];
        self.ll     = ll[tmp]

    ##########################################################################
    # Main routine for noisy limited-memory quasi-Newton optimization (BFGS)
    # based on algorithm 2 and 3 in Schraudolph et al. (2007):
    # A stochastic quasi-Newton method for online convex optimization
    ##########################################################################

    def bfgs_opt(self,sm,sys,thSys):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set initial settings
        sm.calcGradientFlag = True;
        sm.calcHessianFlag  = False;
        self.nPars          = thSys.nParInference;
        self.filePrefix     = thSys.filePrefix;
        self.nParInference  = thSys.nParInference;

        # Check algorithm settings and set to default if needed
        if ( self.memoryLength != sys.T ):
            setSettings(self,"noisyLimitedMemoryQuasiNewtonOpt");
        else:
            setSettings(self,"noisyQuasiNewtonOpt");

        # Pre-allocate outputs
        th        = np.zeros((self.maxIter,self.nParInference));
        fgradDiff = np.zeros((self.maxIter,self.nParInference));
        p         = np.zeros((self.maxIter,self.nParInference));
        fgrad     = np.zeros((self.maxIter,self.nParInference));
        thDiff    = np.zeros((self.maxIter,self.nParInference));
        H         = np.zeros((self.maxIter,self.nParInference,self.nParInference));
        alpha     = np.zeros((self.maxIter,self.nParInference));
        gnorm     = np.zeros((self.maxIter));
        ll        = np.zeros((self.maxIter));
        llDiff    = np.zeros((self.maxIter));
        step      = np.zeros((self.maxIter));
        I         = np.eye(self.nParInference, dtype=int)

        # Estimate the inital log-likelihood and gradient
        thSys.storeParameters(self.initPar,sys);
        sm.smoother(thSys);
        th[0,:]     = self.initPar
        ll[0]       = sm.ll;
        fgrad[0,:]  = sm.gradient;

        # Setup initial Hessian for full-memory BFGS
        H[0,:,:]    = I * self.epsilon;

        # Check initial norm
        thDiff[0,:] = [2 * self.tolLevel]
        gnorm[0]    = vecnorm( fgrad[0,:] )
        self.iter   = 1

        #=====================================================================
        # Main update
        #=====================================================================
        self.exitCondition = False;

        while ( self.exitCondition == False ):

            # Compute search direction
            if ( self.memoryLength == sys.T ):
                # Update Hessian estimate
                H[self.iter,:,:] = self.bfgs_hessian_update( H[self.iter-1,:,:], fgradDiff[self.iter-1,:], thDiff[self.iter-1,:] )

                # Use usual update for BFGS
                p[self.iter,:] = np.dot( H[self.iter,:,:], fgrad[self.iter-1,:] )

            elif ( self.lbgsVersion == "doubleRecursion" ):
                # Use limited-memory update
                p[self.iter,:] = self.lbfgs_direction_update(fgradDiff,thDiff,fgrad[self.iter-1,:])

            elif ( self.lbgsVersion == "standard" ):
                # Update Hessian estimate
                H[self.iter,:,:] = self.lbfgs_hessian_update( fgradDiff, thDiff )

                # Use usual update for BFGS
                p[self.iter,:] = np.dot( H[self.iter,:,:], fgrad[self.iter-1,:] )

            # Use gain schedule for step sizes
            step[self.iter]     = self.tau / ( self.tau + float(self.iter) ) * self.eta;
            alpha[self.iter,:]  = step[self.iter] / self.cc * np.ones(thSys.nParInference);

            # Update the parameter)
            thDiff[self.iter,:]  = alpha[self.iter,:] * p[self.iter,:];
            th[self.iter,:]      = th[self.iter-1,:]  + thDiff[self.iter,:];

            # Estimate the log-likelihood and gradient
            thSys.storeParameters(th[self.iter,:],sys);
            sm.smoother(thSys);
            ll[self.iter]        = sm.ll;
            fgrad[self.iter,:]   = sm.gradient;

            # Compute the differences in log-likelihood and gradient (use dampening)
            llDiff[self.iter]      = np.abs(ll[self.iter] - ll[self.iter-1]);
            fgradDiff[self.iter,:] = fgrad[self.iter,:]   - fgrad[self.iter-1,:] + self.lam * thDiff[self.iter,:];

            # Check exit conditions
            gnorm[self.iter] = vecnorm( fgrad[self.iter,:] )

            if ( ( self.iter > 5 ) & ( np.sum( gnorm[ range(self.iter-5,self.iter) ] <= self.tolLevel ) == 5) ):
                self.exitCondition = True;
                print("Exiting as the tolerance level has been reached.")

            if ( self.iter+1 == self.maxIter ):
                self.exitCondition = True;
                print("Exiting as the maximum number of iterations has been reached.")

            # Print output to console
            if (self.verbose ):
                parm = ["%.4f" % v for v in th[self.iter]];
                print("Iteration: " + str(self.iter) + " with current parameters: " + str(parm) + " and lldiff: " + str(llDiff[self.iter]) )

            # Update iteration number
            self.iter += 1

        #=====================================================================
        # Compile output
        #=====================================================================
        tmp          = range(0,self.iter-1);
        self.th      = th[tmp,:];
        self.step    = step[tmp]
        self.llDiff  = llDiff[tmp];
        self.Hest    = H[tmp,:,:];
        self.norm    = gnorm[tmp]
        self.thDiff  = thDiff[tmp,:]
        self.ll      = ll[tmp]
        self.thdDiff = fgradDiff[tmp,:]

    ##########################################################################
    # Subroutine for noisy limited-memory quasi-Newton optimization (BFGS)
    # based on algorithm 2 and 3 in Schraudolph et al. (2007):
    # A stochastic quasi-Newton method for online convex optimization
    ##########################################################################

    def lbfgs_direction_update(self,ykk,skk,gk):
        alpha  = np.zeros(self.maxIter);

        # Set initial search direction to current gradient
        pkk = gk;

        if ( self.iter > 1 ):
            afactor = 0.0;
            idx     = range( np.max( (1,self.iter-self.memoryLength) ), self.iter );

            # Loop 1
            for ii in idx:
                alpha[ii]  = np.dot( skk[ii,:], pkk) / np.dot( skk[ii,:], ykk[ii,:]);
                pkk       -= alpha[ii] * ykk[ii,:];
                afactor   += np.dot( skk[ii,:], ykk[ii,:]) / np.dot( ykk[ii,:], ykk[ii,:])

            # Average out noise as in equation (14)
            pkk = pkk / np.min((self.iter,self.memoryLength)) * afactor;

            # Loop 2
            for ii in reversed(idx):
                beta  = np.dot( ykk[ii,:], pkk) / np.dot( ykk[ii,:], skk[ii,:]);
                pkk  += ( alpha[ii] - beta ) * skk[ii,:];

        else:
            pkk = self.epsilon * gk;

        return pkk;

    ##########################################################################
    # Subroutine for noisy quasi-Newton optimization (BFGS)
    # based on algorithm 2 in Schraudolph et al. (2007):
    # A stochastic quasi-Newton method for online convex optimization
    ##########################################################################

    def lbfgs_hessian_update(self,ykk,skk):

        I  = np.eye(self.nParInference, dtype=int);
        Hk = np.eye(self.nParInference) * self.epsilon;

        if ( self.iter > 1 ):
            idx     = range( np.max( (1,self.iter-self.memoryLength) ), self.iter );

            for ii in idx:
                # Compute rho and update Hessian estimate.
                rhok = 1.0 / ( np.dot( ykk[ii,:], skk[ii,:]) );
                A1   = I - skk[ii, :, np.newaxis] * ykk[ii, np.newaxis, :] * rhok
                A2   = I - ykk[ii, :, np.newaxis] * skk[ii, np.newaxis, :] * rhok
                Hk   = np.dot(A1, np.dot(Hk, A2)) + (self.cc * rhok * skk[ii, :, np.newaxis] * skk[ii, np.newaxis, :])

        # We have negative eigenvalues in the inverse Hessian
        if ( np.sum( np.linalg.eig(-Hk)[0] < 0) > 0 ):

            # Find the smallest eigenvalue
            eigmin = np.min( np.linalg.eig(Hk)[0] );

            # Regularise H by mirroring the eigenvalue
            Hk += np.diag(np.ones(self.nParInference)) * - 2.0 * eigmin;

            print("Found a non-PSD Hessian estimate so added " + str(-2.0*eigmin) + " to the diagonal elements.")

        return Hk;

    ##########################################################################
    # Subroutine for noisy quasi-Newton optimization (BFGS)
    # based on algorithm 2 in Schraudolph et al. (2007):
    # A stochastic quasi-Newton method for online convex optimization
    ##########################################################################

    def bfgs_hessian_update(self,Hk,ykk,skk):

        I     = np.eye(self.nParInference, dtype=int)

        if ( self.iter > 1 ):
            # Compute rho and update Hessian estimate.
            rhok  = 1.0 / ( np.dot( ykk, skk) );
            A1    = I - skk[:, np.newaxis] * ykk[np.newaxis, :] * rhok
            A2    = I - ykk[:, np.newaxis] * skk[np.newaxis, :] * rhok
            Hkk   = np.dot(A1, np.dot(Hk, A2)) + (self.cc * rhok * skk[:, np.newaxis] * skk[np.newaxis, :])

            # We have negative eigenvalues in the inverse Hessian
            if ( np.sum( np.linalg.eig(Hkk)[0] < 0) > 0 ):

                # Find the smallest eigenvalue
                eigmin = np.min( np.linalg.eig(Hkk)[0] );

                # Regularise H by mirroring the eigenvalue
                Hkk += np.diag(np.ones(self.nParInference)) * -2.0 * eigmin;

                print("Found a non-PSD Hessian estimate so added " + str(-2.0*eigmin) + " to the diagonal elements.")

        else:
            Hkk = I * self.epsilon;

        return Hkk;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
