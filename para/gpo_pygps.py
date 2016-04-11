##############################################################################
##############################################################################
# Routines for
# Parameter inference using GPO
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
import pyGPs
import pandas
from   DIRECT     import solve
from   ml_helpers import *

##############################################################################
# Main class
##############################################################################

class stGPO(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # Should the algorithm write out its status during each iteration
    verbose                         = None;

    # Variables for constructing the file name when writing the output to file
    filePrefix                      = None;
    dataset                         = None;

    # Exit conditions (the diff in norm of theta or maximum no iterations)
    tolLevel                        = None;
    maxIter                         = None;

    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval  = None;
    writeOutProgressToFile          = None;
    fileOutName                     = None;

    # GPO settings
    epsilon                         = None;
    preIter                         = None;
    upperBounds                     = None;
    lowerBounds                     = None;
    EstimateHyperparametersInterval = None;
    AQfunction                      = None;
    EstimateThHatEveryIteration     = None;

    # Give step size and inital parameters
    stepSize                        = None;
    initPar                         = None;

    # Input design
    inputDesignMethod               = None;

    # Jittering of parameters
    jitterParameters                = None;
    jitteringCovariance             = None;

    ##########################################################################
    # Wrappers for different versions of the algorithm
    ##########################################################################

    def ml(self,sm,sys,th):
        self.optType   = "MLparameterEstimation"
        self.optMethod = "gpo_ml"
        self.gpo(sm,sys,th);

    def bayes(self,sm,sys,th):
        self.optType   = "MAPparameterEstimation"
        self.optMethod = "gpo_map"
        self.gpo(sm,sys,th);

    ##########################################################################
    # Gaussian process optimisation (GPO)
    ##########################################################################
    def gpo(self,sm,sys,thSys):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set initial settings
        sm.calcGradientFlag = False;
        sm.calcHessianFlag  = False;
        self.nPars          = thSys.nParInference;
        self.filePrefix     = thSys.filePrefix;
        runNextIter         = True;
        self.iter           = 0;

        # Check algorithm settings and set to default if needed
        setSettings(self,"gpo");

        # Make a grid to evaluate the EI on
        l      = np.array(self.lowerBounds[0:thSys.nParInference], dtype=np.float64)
        u      = np.array(self.upperBounds[0:thSys.nParInference], dtype=np.float64)

        # Allocate vectors
        AQ    = np.zeros((self.maxIter+1,1))
        mumax = np.zeros((self.maxIter+1))
        thp   = np.zeros((self.maxIter+1,self.nPars))
        obp   = np.zeros((self.maxIter+1,1))
        thhat = np.zeros((self.maxIter,self.nPars))
        obmax = np.zeros((self.maxIter+1,1))

        #=====================================================================
        # Specify the GP regression model
        #=====================================================================

        # Load the GP regression model
        m           = pyGPs.GPR()

        # Specify the GP prior
        priorMean   = pyGPs.mean.Zero();
        #priorKernel = pyGPs.cov.Matern(d=5) + pyGPs.cov.Const();
        priorKernel = pyGPs.cov.RBFard(D=self.nPars) + pyGPs.cov.Const();

        m.setPrior(mean=priorMean, kernel=priorKernel)

        # Setup the optimization routine
        m.setOptimizer('Minimize', num_restarts=10)

        #=====================================================================
        # Pre-run using random sampling to estimate hyperparameters
        #=====================================================================

        # Pre allocate vectors
        thPre = np.zeros((self.preIter,self.nPars));
        obPre = np.zeros((self.preIter,1));

        #=====================================================================
        # Main loop
        #=====================================================================
        for kk in range(0,self.preIter):

            # Sample parameters

            #if (self.optType == "MAPparameterEstimation"):
            # Sample prior
            #thPre[kk,:] = thSys.samplePrior()
            #else:
            # Draw parameters uniform over the parameter bounds
            thPre[kk,:]  =  l + (u-l) * np.random.random( self.nPars )

            # Evaluate the objective function in the parameters
            thSys.storeParameters(thPre[kk,:],sys);
            obPre[kk] = self.evaluateObjectiveFunction( sm, sys, thSys );

            # Transform and save the parameters
            thSys.transform();
            thPre[kk,:] = thSys.returnParameters()[0:thSys.nParInference];

            # Write out progress if requested
            if (self.verbose):
                print("gpo: Pre-iteration: " + str(kk) + " of " + str(self.preIter) + " completed, sampled " + str(thPre[kk,:]) + " with " + str(obPre[kk]) + ".")

        #=====================================================================
        # Fit the GP regression
        #=====================================================================

        # Remove nan values for the objective function
        idxNotNaN = ~np.isnan( obPre );
        thPre     = thPre[(idxNotNaN).any(axis=1)];
        obPre     = obPre[(idxNotNaN).any(axis=1)];

        # Normalize the objective function evaluations
        ynorm = ( obPre - np.mean(obPre) ) / np.sqrt( np.var(obPre) )

        # Optimize the Hyperparameters
        m.optimize(thPre, ynorm)
        #yp = m.predict(np.arange(0.01,1.00,0.01))
        #m.plot()

        #=====================================================================
        # Write to output
        #=====================================================================

        self.thPre  = thPre;
        self.obPre  = obPre;
        self.m      = m;

        #=====================================================================
        # Main loop
        #=====================================================================

        # Save the initial parameters
        thSys.storeParameters(self.initPar,sys);
        thp[self.iter,:]  = thSys.returnParameters();
        thSys.transform();

        while ( runNextIter ):

            # Store the parameter
            thSys.storeParameters(thp[self.iter,:],sys);
            thSys.transform();

            #------------------------------------------------------------------
            # Evalute the objective function
            #------------------------------------------------------------------
            obp[self.iter] = self.evaluateObjectiveFunction( sm, sys, thSys );

            # Collect the sampled data (if the objective is finite)
            if np.isfinite(obp[self.iter]):
                x = np.vstack( (thPre,thp[range(self.iter),:]) );
                y = np.vstack( (obPre,obp[range(self.iter),:]) );

            # Normalize the objective function evaluations
            ynorm = ( y - np.mean(y) ) / np.sqrt( np.var(y) )

            #------------------------------------------------------------------
            # Fit the GP to the sampled data
            #------------------------------------------------------------------

            # Optimize the hyperparameters if needed
            #if ( np.remainder(self.iter+1,self.EstimateHyperparametersInterval) == 0):
            #    m.optimize(x, ynorm)
            #else:
            #    m.setData(x, ynorm)

            m.optimize(x, ynorm)

            # Extract the posterior values of the hyperparameters
            post = m.posterior;

            #------------------------------------------------------------------
            # Find the maximum expected value of the GP over the sampled parameters
            #------------------------------------------------------------------
            Mup, ys2, fmu, fs2, lp = m.predict_with_posterior(post, np.array( np.vstack( (thPre,thp[range(self.iter),:]) ) ) )
            mumax[self.iter] = np.max( Mup );

            #------------------------------------------------------------------
            # Compute the next point in which to sample the posterior
            #------------------------------------------------------------------

            # Optimize the AQ function
            aqThMax, aqMax, ierror = solve(self.AQfunction,l,u,user_data=(m,mumax[self.iter],self.epsilon,post),maxf=1000,maxT=1000);

            # Jitter the parameter estimates
            if ( self.jitterParameters == True ):
                aqThMax += np.random.multivariate_normal( np.zeros(self.nPars), self.jitteringCovariance[range(self.nPars),:][:,range(self.nPars)] );

            # Set the new point and save the estimate of the AQ
            thp[self.iter+1,:] = aqThMax;
            AQ[self.iter+1]    = -aqMax;

            # Update counter
            self.iter += 1;

            #------------------------------------------------------------------
            # Check exit conditions
            #------------------------------------------------------------------

            # AQ function criteria
            if ( AQ[self.iter] < self.tolLevel ):
                print("GPO: reaches tolLevel, so exiting...")
                runNextIter = False;

            # Max iteration criteria
            if ( self.iter == self.maxIter ):
                print("GPO: reaches maxIter, so exiting...")
                runNextIter = False;

            #------------------------------------------------------------------
            # Estimate the current parameters by maximizing the GP
            #------------------------------------------------------------------

            if ( ( self.EstimateThHatEveryIteration == True ) | runNextIter == False ):
                thhatCurrent, obmaxCurrent, ierror = solve(self.MUeval,l,u,user_data=(m,post),algmethod=1);
                thhat[self.iter-1,:] = thhatCurrent;
                obmax[self.iter-1,:] = obmaxCurrent;

            #------------------------------------------------------------------
            # Print output to console
            #------------------------------------------------------------------
            if ( self.verbose ):
                if ( self.EstimateThHatEveryIteration == True ):
                    parm = ["%.4f" % v for v in thhat[self.iter-1,:]];
                    print("##############################################################################################")
                    print("Iteration: " + str(self.iter) + " with current parameters: " + str(parm) + " and AQ: " + str(AQ[self.iter]) )
                    print("##############################################################################################")
                else:
                    parm = ["%.4f" % v for v in thp[self.iter-1,:]];
                    print("##############################################################################################")
                    print("Iteration: " + str(self.iter) + " sampled objective function at parameters: " + str(parm) + " with value: " + str(obp[self.iter-1]) )
                    print("##############################################################################################")


        #=====================================================================
        # Generate output
        #=====================================================================
        tmp         = range(self.iter-1);
        self.ob     = obmax[tmp];
        self.th     = thhat[tmp,:];
        self.thhat  = thhat[self.iter-1,:]
        self.aq     = AQ[range(self.iter)];
        self.obp    = obp[tmp];
        self.thp    = thp[range(self.iter),:];

        self.m      = m;
        self.x      = x;
        self.y      = y;

    ##########################################################################
    # Expected Improvement (EI) aquisition rule
    ##########################################################################

    def aq_ei(self,x,user_data):
        m       = user_data[0];
        obmax   = user_data[1];
        epsilon = user_data[2];
        post    = user_data[3];

        # Find the predicted value in x
        Mup, Mus, fmu, fs2, lp = m.predict_with_posterior( post, np.array(x).reshape((1,len(x))) )

        # Calculate auxillary quantites
        s   = np.sqrt(Mus);
        yres  = Mup - obmax - epsilon;
        ynorm = ( yres / s) * ( s > 0 );

        # Compute the EI and negate it
        ei  = yres * sp.stats.norm.cdf(ynorm) + s * sp.stats.norm.pdf(ynorm);
        ei  = np.max((ei,0));
        return -ei, 0

    ##########################################################################
    # Evaluate the surrogate function to find its maximum
    ##########################################################################

    def MUeval(self,x,user_data):
        m       = user_data[0];
        post    = user_data[1];

        Mup, Mus, fmu, fs2, lp = m.predict_with_posterior( post, np.array(x).reshape((1,len(x))) )
        return -Mup, 0

    ##########################################################################
    # Probability of Improvement (PI) aquisition rule
    ##########################################################################

    def aq_pi(self,x,user_data):
        m       = user_data[0];
        obmax   = user_data[1];
        epsilon = user_data[2];
        post    = user_data[3];

        # Find the predicted value in x
        Mup, Mus, fmu, fs2, lp = m.predict_with_posterior( post, np.array(x).reshape((1,len(x))) )

        # Calculate auxillary quantites
        s   = np.sqrt(Mus);
        yres  = Mup - obmax - epsilon;
        ynorm = ( yres / s) * ( s > 0 );

        # Compute the EI and negate it
        pi  = sp.stats.norm.cdf(ynorm);
        return -pi, 0

    ##########################################################################
    # Upper Confidence Bound (UCB) aquisition rule
    ##########################################################################

    def aq_ubc(self,x,user_data):
        m       = user_data[0];
        obmax   = user_data[1];
        epsilon = user_data[2];
        post    = user_data[3];

        # Find the predicted value in x
        Mup, Mus, fmu, fs2, lp = m.predict_with_posterior( post, np.array(x).reshape((1,len(x))) )

        # Calculate auxillary quantites
        s   = np.sqrt(Mus);

        # Compute the EI and negate it
        ucb  = Mup + epsilon * s;
        return -ucb, 0

    ##########################################################################
    # Evaluate the different objective functions
    ##########################################################################

    def evaluateObjectiveFunction( self, sm, sys, thSys ):

        # Evalute the objective function
        if (self.optType == "MLparameterEstimation"):

            # Sample the log-likelihood in the proposed parameters
            sm.filter(thSys)
            out  = sm.ll;

        elif (self.optType == "MAPparameterEstimation"):

            # Sample the log-target in the proposed parameters
            sm.filter(thSys)
            out  = sm.ll + thSys.prior();

        elif(self.optType == "InputDesign"):

            # Sample the log-likelihood in the proposed parameters
            out  = self.inputDesignMethod( sm, thSys );

        return out

    ##########################################################################
    # Helper: estimate the Hessian
    ##########################################################################
    def estimateHessian(self):
        hes  = nd.Hessian( self.evaluateSurrogate )
        self.hessianEstimate    = - hes( self.thhat );
        self.invHessianEstimate = np.linalg.pinv( self.hessianEstimate )

    ##########################################################################
    # Helper: calculate the log-likelihood
    ##########################################################################
    def evaluateSurrogate(self,x):
        m = self.m;

        Mup, Mus, fmu, fs2, lp = m.predict( np.array(x).reshape((1,len(x))) )

        parm = ["%.4f" % v for v in x];
        print("Current parameters: " + str( parm ) + " with objective: " + str(Mup) )
        return( Mup );

    ##########################################################################
    # Helper: compile the results and write to file
    ##########################################################################
    def writeToFile(self,sm=None,fileOutName=None):

        # Construct the columns labels
        columnlabels = [None]*(2*self.nPars+3);

        for ii in range(0,self.nPars):
            columnlabels[ii]            = "thhat"   + str(ii);
            columnlabels[ii+self.nPars]   = "thp"   + str(ii);

        columnlabels[2*self.nPars]        = "ob"
        columnlabels[2*self.nPars+1]      = "obp"
        columnlabels[2*self.nPars+2]      = "aq";

        # Compile the results for output
        tmp         = range(0,self.iter-1);
        out = np.hstack((self.th[tmp,:],self.thp[tmp,:],self.ob[tmp],self.obp[tmp],self.aq[tmp]));

        # Write out the results to file
        fileOut = pandas.DataFrame(out,columns=columnlabels);

        if ( fileOutName == None ):
            if ( sm.filterType == "kf" ):
                fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.optMethod) + '_' + sm.filterType + '/' + str(self.dataset) + '.csv';
            elif hasattr(sm, 'filterType'):
                fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.optMethod) + '_' + sm.filterType + '_N' + str(sm.nPart)  + '/' + str(self.dataset) + '.csv';
            else:
                fileOutName = 'gpo-output.csv'

        ensure_dir(fileOutName);
        fileOut.to_csv(fileOutName);

        print("writeToFile_helper: wrote results to file: " + fileOutName)


########################################################################
# End of file
########################################################################
