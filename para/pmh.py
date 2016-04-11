##############################################################################
##############################################################################
# Routines for
# Particle Metrpolis-Hastings for Bayesian parameter inference
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

import numpy       as     np
from   pmh_helpers import *
import pandas

##########################################################################
# Main class
##########################################################################

class stPMH(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # The self.stepSize size and inverse Hessian for the sampler
    stepSize                        = None;
    invHessian                      = None;

    # How many iterations should we run the sampler for and how long is the burn-in
    nIter                           = None;
    nBurnIn                         = None;
    proposeRVs                      = False;
    alwaysAccept                    = False;

    # PMH2: should we only use the diagonal elements of the Hessian
    makeHessianDiagonal             = None;

    # PMH2: what should we do if it is not PSD ( choose one )
    makeHessianPSDmethod            = None;
    makeHessianPSDregularise        = None;
    makeHessianPSDhybrid            = None;
    makeHessianPSDreject            = None;

    # PMH2: how many samples during the burn-in should we use to est the Hessian
    PSDmethodhybridSamps            = None;
    empHessian                      = None;

    # Adaptive MCMC
    adaptHessianAfterBurnIn         = None;
    adaptHessianNoSamples           = None;

    # When should we print a progress report? Should prior warnings be written to screen.
    nProgressReport                 = None;
    writeOutPriorWarnings           = None;

    # Write out to file during the run (for large simulations)
    writeOutProgressToFileInterval  = None;
    writeOutProgressToFile          = None;
    fileOutName                     = None;

    # Variables for constructing the file name when writing the output to file
    filePrefix                      = None;
    dataset                         = None;

    # Variables for quasi-Newton proposal
    qPMHadaptInitialHessian         = None;
    memoryLength                    = None;
    epsilon                         = None;

    # Wrappers
    zvpost_linear                   = zvpost_linear_prototype;
    calcIACT                        = calcIACT_prototype;
    calcSJD                         = calcSJD_prototype;
    calcESS                         = calculateESS_prototype;

    ##########################################################################
    # Main sampling routine
    ##########################################################################

    def runSampler(self,sm,sys,thSys,PMHtype):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set file prefix from model
        self.filePrefix = thSys.filePrefix;
        self.iter       = 0;
        self.PMHtype    = PMHtype;
        self.nPars      = thSys.nParInference;
        self.T          = sys.T;

        if ( sm.typeSampler == 'smc'):
            self.nPart      = sm.nPart;

        # Initialising settings and using default if no settings provided
        setSettings(self,PMHtype);

        # Allocate vectors
        self.ll             = np.zeros((self.nIter,1))
        self.llp            = np.zeros((self.nIter,1))
        self.th             = np.zeros((self.nIter,self.nPars))
        self.tho            = np.zeros((self.nIter,self.nPars))
        self.thp            = np.zeros((self.nIter,self.nPars))
        self.x              = np.zeros((self.nIter,self.T))
        self.xp             = np.zeros((self.nIter,self.T))
        self.aprob          = np.zeros((self.nIter,1))
        self.accept         = np.zeros((self.nIter,1))
        self.gradient       = np.zeros((self.nIter,self.nPars))
        self.gradientp      = np.zeros((self.nIter,self.nPars))
        self.hessian        = np.zeros((self.nIter,self.nPars,self.nPars))
        self.hessianp       = np.zeros((self.nIter,self.nPars,self.nPars))
        self.prior          = np.zeros((self.nIter,1))
        self.priorp         = np.zeros((self.nIter,1))
        self.J              = np.zeros((self.nIter,1))
        self.Jp             = np.zeros((self.nIter,1))
        self.proposalProb   = np.zeros((self.nIter,1))
        self.proposalProbP  = np.zeros((self.nIter,1))
        self.llDiff         = np.zeros((self.nIter,1))

        # Get the order of the PMH sampler
        if   ( ( PMHtype == "PMH0" ) | ( PMHtype == "pPMH0" ) ):
            self.PMHtypeN = 0;
        elif ( ( PMHtype == "PMH1" ) | ( PMHtype == "pPMH1" ) | ( PMHtype == "bPMH1" ) | ( PMHtype == "pbPMH1" ) ):
            self.PMHtypeN = 1;
            sm.calcGradientFlag = True;
        elif ( PMHtype == "PMH2" ):
            self.PMHtypeN = 2;
            sm.calcGradientFlag = True;
            if ( sm.calcHessianFlag == None ):
                sm.calcHessianFlag = "louis";
        elif ( PMHtype == "qPMH2" ):
            self.PMHtypeN = 2;
            sm.calcGradientFlag = True;
            sm.calcHessianFlag  = False;
            self.nHessianSamples = np.zeros((self.nIter,1))

        # Initialise the parameters in the proposal
        thSys.storeParameters(self.initPar,sys);

        # Run the initial filter/smoother
        self.estimateLikelihoodGradientsHessians(sm,thSys);
        self.acceptParameters(thSys);

        # Inverse transform and then save the initial parameters and the prior
        self.tho[0,:] = thSys.returnParameters();
        self.prior[0] = thSys.prior()
        self.J[0]     = thSys.Jacobian();

        thSys.invTransform();
        self.th[0,:]  = thSys.returnParameters();

        #=====================================================================
        # Main MCMC-loop
        #=====================================================================
        for kk in range(1,self.nIter):

            self.iter = kk;

            # Propose parameters
            self.sampleProposal();
            thSys.storeParameters( self.thp[kk,:], sys );
            thSys.transform();

            # Calculate acceptance probability
            self.calculateAcceptanceProbability( sm, thSys );

            # Accept/reject step
            if ( np.random.random(1) < self.aprob[kk] ):
                self.acceptParameters( thSys );
            else:
                self.rejectParameters( thSys );

            # Always accept?
            if ( self.alwaysAccept ):
                 self.acceptParameters( thSys );

            # Write out progress report
            if np.remainder( kk, self.nProgressReport ) == 0:
                progressPrint( self );

            # Write out progress at some intervals
            if ( self.writeOutProgressToFile ):
                if np.remainder( kk, self.writeOutProgressToFileInterval ) == 0:
                    self.writeToFile( sm );

        progressPrint(self);
        self.thhat = np.mean( self.th[ self.nBurnIn:self.nIter, : ] , axis=0 );
        self.xhats = np.mean( self.x[ self.nBurnIn:self.nIter, : ] , axis=0 );

    ##########################################################################
    # Sample the proposal
    ##########################################################################
    def sampleProposal(self,):

        if ( self.PMHtype == "PMH0" ):
            if ( self.nPars == 1 ):
                self.thp[self.iter,:] = self.th[self.iter-1,:] + self.stepSize * np.random.normal();
            else:
                self.thp[self.iter,:] = self.th[self.iter-1,:] + np.random.multivariate_normal( np.zeros(self.nPars), self.stepSize**2 * np.eye( self.nPars ) );

        elif ( self.PMHtype == "pPMH0" ):
            # Sample the preconditioned PMH0 proposal
            if ( self.nPars == 1 ):
                self.thp[self.iter,:] = self.th[self.iter-1,:] + self.stepSize * np.random.normal();
            else:
                self.thp[self.iter,:] = self.th[self.iter-1,:] + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian );

        elif ( self.PMHtype == "PMH1" ):
            self.thp[self.iter,:] = self.th[self.iter-1,:] + 0.5 * self.stepSize**2 * self.gradient[self.iter-1,:] + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * np.eye( self.nPars )   );

        elif ( self.PMHtype == "bPMH1" ):
            #"Binary" PMH1 proposal
            self.thp[self.iter,:] = self.th[self.iter-1,:] + self.stepSize * np.sign(self.gradient[self.iter-1,:]) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * np.eye( self.nPars )   );

        elif ( self.PMHtype == "pbPMH1" ):
            #"Binary" PMH1 proposal
            self.thp[self.iter,:] = self.th[self.iter-1,:] + self.stepSize * np.dot(self.invHessian, np.sign(self.gradient[self.iter-1,:]) ) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian   );

        elif( self.PMHtype == "pPMH1" ):
            # Sample the preconditioned PMH1 proposal
            self.thp[self.iter,:] = self.th[self.iter-1,:] + 0.5 * self.stepSize**2 * np.dot(self.invHessian,self.gradient[self.iter-1,:]) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.invHessian );

        elif ( self.PMHtype == "PMH2" ):
            self.thp[self.iter,:] = self.th[self.iter-1,:] + 0.5 * self.stepSize**2 * np.dot(self.gradient[self.iter-1,:], np.linalg.pinv(self.hessian[self.iter-1,:,:])) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * np.linalg.pinv(self.hessian[self.iter-1,:,:]) );

        elif ( self.PMHtype == "qPMH2" ):

            if ( self.iter > self.memoryLength ):
                self.thp[self.iter,:] = self.th[self.iter-self.memoryLength,:] + 0.5 * self.stepSize**2 * np.dot( self.gradient[self.iter-self.memoryLength,:], self.hessian[self.iter-self.memoryLength,:,:] ) + np.random.multivariate_normal(np.zeros(self.nPars), self.stepSize**2 * self.hessian[self.iter-self.memoryLength,:,:] );
            else:
                # Initial phase, use pPMH0
                self.thp[self.iter,:] = self.th[self.iter-1,:] + np.random.multivariate_normal( np.zeros(self.nPars), self.stepSize**2 * self.hessian[self.iter-1,:,:] );

    ##########################################################################
    # Calculate Acceptance Probability
    ##########################################################################
    def calculateAcceptanceProbability(self, sm,  thSys, ):

        # Check the "hard prior"
        if (thSys.priorUniform() == 0.0):
            if (self.writeOutPriorWarnings):
                print("The parameters " + str( self.thp[ self.iter,:] ) + " were proposed.");
            return None;

        # Run the smoother to get the ll-estimate, gradient and hessian-estimate
        self.estimateLikelihoodGradientsHessians(sm,thSys);

        # Compute the part in the acceptance probability related to the non-symmetric proposal
        if ( self.PMHtype == "PMH0" ):
            proposalP = 0;
            proposal0 = 0;

        elif ( self.PMHtype == "pPMH0" ):
            proposalP = 0;
            proposal0 = 0;

        elif ( self.PMHtype == "PMH1" ):
            proposalP = lognormpdf( self.thp[self.iter,:], self.th[self.iter-1,:]  + 0.5 * self.stepSize**2 * self.gradient[self.iter-1,:], self.stepSize**2 * np.eye( self.nPars )    );
            proposal0 = lognormpdf( self.th[self.iter-1,:],  self.thp[self.iter,:] + 0.5 * self.stepSize**2 * self.gradientp[self.iter,:],     self.stepSize**2 * np.eye( self.nPars )    );

        elif ( self.PMHtype == "bPMH1" ):
            #"Binary" PMH1 proposal
            proposalP = lognormpdf( self.thp[self.iter,:], self.th[self.iter-1,:]  + self.stepSize * np.sign( self.gradient[self.iter-1,:] ), self.stepSize**2 * np.eye( self.nPars )    );
            proposal0 = lognormpdf( self.th[self.iter-1,:],  self.thp[self.iter,:] + self.stepSize * np.sign( self.gradientp[self.iter,:]  ), self.stepSize**2 * np.eye( self.nPars )    );

        elif ( self.PMHtype == "pbPMH1" ):
            #"Binary" PMH1 proposal
            proposalP = lognormpdf( self.thp[self.iter,:], self.th[self.iter-1,:]  + self.stepSize * np.dot( self.invHessian, np.sign( self.gradient[self.iter-1,:]) ), self.stepSize**2 * self.invHessian    );
            proposal0 = lognormpdf( self.th[self.iter-1,:],  self.thp[self.iter,:] + self.stepSize * np.dot( self.invHessian, np.sign( self.gradientp[self.iter,:]  ) ), self.stepSize**2 * self.invHessian    );

        elif ( self.PMHtype == "pPMH1" ):
            proposalP = lognormpdf( self.thp[self.iter,:], self.th[self.iter-1,:]  + 0.5 * self.stepSize**2 * np.dot( self.invHessian,self.gradient[self.iter-1,:]),  self.stepSize**2 * self.invHessian  );
            proposal0 = lognormpdf( self.th[self.iter-1,:],self.thp[self.iter,:]   + 0.5 * self.stepSize**2 * np.dot( self.invHessian,self.gradientp[self.iter,:]) ,  self.stepSize**2 * self.invHessian  );

        elif ( self.PMHtype == "PMH2" ):
            proposalP = lognormpdf( self.thp[self.iter,:], self.th[self.iter-1,:]  + 0.5 * self.stepSize**2 * np.dot( self.gradient[self.iter-1,:],  np.linalg.pinv(self.hessian[self.iter-1,:,:])  ), self.stepSize**2 * np.linalg.pinv(self.hessian[self.iter-1,:,:])  );
            proposal0 = lognormpdf( self.th[self.iter-1,:],  self.thp[self.iter,:] + 0.5 * self.stepSize**2 * np.dot( self.gradientp[self.iter,:],   np.linalg.pinv(self.hessianp[self.iter,:,:]) ), self.stepSize**2 * np.linalg.pinv(self.hessianp[self.iter,:,:]) );

        elif ( self.PMHtype == "qPMH2" ):

            if ( self.iter > self.memoryLength ):
                proposalP = lognormpdf( self.thp[self.iter,:],                   self.th[self.iter-self.memoryLength,:]  + 0.5 * self.stepSize**2 * np.dot( self.gradient[self.iter-self.memoryLength,:],  self.hessian[self.iter-self.memoryLength,:,:])  , self.stepSize**2 * self.hessian[self.iter-self.memoryLength,:,:]  );
                proposal0 = lognormpdf( self.th[self.iter-self.memoryLength,:],  self.thp[self.iter,:]                   + 0.5 * self.stepSize**2 * np.dot( self.gradientp[self.iter,:],                   self.hessianp[self.iter,:,:]) ,                   self.stepSize**2 * self.hessianp[self.iter,:,:] );
            else:
                # Initial phase, use pPMH0
                proposalP = lognormpdf( self.thp[self.iter,:],   self.th[self.iter-1,:]   , self.stepSize**2 * self.hessian[self.iter-1,:,:] );
                proposal0 = lognormpdf( self.th[self.iter-1,:],  self.thp[self.iter,:]    , self.stepSize**2 * self.hessianp[self.iter,:,:]  );

        # Compute prior and Jacobian
        self.priorp[ self.iter ]    = thSys.prior();
        self.Jp[ self.iter ]        = thSys.Jacobian();

        # Compute the acceptance probability
        self.aprob[ self.iter ] = self.flag * np.exp( self.llp[ self.iter, :] - self.ll[ self.iter-1, :] + proposal0 - proposalP + self.priorp[ self.iter, :] - self.prior[ self.iter-1, :] + self.Jp[ self.iter, :] - self.J[ self.iter-1, :] );

        # Store the proposal calculations
        self.proposalProb[ self.iter ]  = proposal0;
        self.proposalProbP[ self.iter ] = proposalP;
        self.llDiff[ self.iter ]        = self.llp[ self.iter, :] - self.ll[ self.iter-1, :];

    ##########################################################################
    # Run the SMC algorithm and get the required information
    ##########################################################################
    def estimateLikelihoodGradientsHessians(self,sm,thSys,):

        # Flag if the Hessian is PSD or not.
        self.flag  = 1.0

        # PMH0, only run the filter and extract the likelihood estimate
        if ( self.PMHtypeN == 0 ):
            sm.filter(thSys);

        # PMH1, only run the smoother and extract the likelihood estimate and gradient
        if ( self.PMHtypeN == 1 ):
            sm.smoother(thSys);
            self.gradientp[ self.iter,: ]   = sm.gradient;

        # PMH2, only run the smoother and extract the likelihood estimate and gradient
        if ( self.PMHtype == "qPMH2" ):
            sm.smoother(thSys);
            self.gradientp[ self.iter,: ]   = sm.gradient;

            # Note that this is the inverse Hessian
            self.hessianp [ self.iter,:,: ] = self.lbfgs_hessian_update( );

            # Extract the diagonal if needed and regularise if not PSD
            self.checkHessian();

        elif ( self.PMHtypeN == 2 ):
            sm.smoother(thSys);
            self.gradientp[ self.iter,: ]   = sm.gradient;
            self.hessianp [ self.iter,:,: ] = sm.hessian;

            # Extract the diagonal if needed and regularise if not PSD
            self.checkHessian();

        # Create output
        self.llp[ self.iter ]        = sm.ll;
        self.xp[ self.iter, : ]      = sm.xtraj;

        return None;

    ##########################################################################
    # Extract the diagonal if needed and regularise if not PSD
    ##########################################################################
    def checkHessian(self):

        # Extract the diagonal if it is the only part that we want
        if ( self.makeHessianDiagonal ):
            self.hessianp [ self.iter,:,: ] = np.diag( np.diag( self.hessianp [ self.iter,:,: ] ) );

        # Pre-calculate posterior covariance estimate
        if ( ( self.makeHessianPSDmethod == "hybrid" ) & ( self.iter >= self.nBurnIn ) & ( self.empHessian == None ) ) :
            self.empHessian = np.cov( self.th[range( self.nBurnIn - self.PSDmethodhybridSamps, self.nBurnIn ),].transpose() );

        if ( ( self.makeHessianPSDmethod == "hybrid2" ) & ( self.iter >= self.nBurnIn ) & ( self.empHessian == None ) ) :
            self.empHessian = np.cov( self.th[range( self.nBurnIn - self.PSDmethodhybridSamps, self.nBurnIn ),].transpose() );

        # Check if it is PSD
        if ( ~isPSD( self.hessianp [ self.iter,:,: ] ) ):

            eigens = np.linalg.eig(self.hessianp [ self.iter,:,: ])[0];

            #=================================================================
            # Should we try to make the Hessian PSD using approach 1
            # Mirror the smallest eigenvalue in zero.
            #=================================================================

            # Add a diagonal matrix proportional to the largest negative eigv
            if ( self.makeHessianPSDmethod == "regularise" ):
                mineigv = np.min( np.linalg.eig( self.hessianp [ self.iter,:,: ] )[0] )
                self.hessianp [ self.iter,:,: ] = self.hessianp [ self.iter,:,: ] - 2.0 * mineigv * np.eye( self.nPars )
                print("Iteration: " + str(self.iter) + " has eigenvalues: " + str( eigens ) + " mirroring by adding " +  str( - 2.0 * mineigv ) );

            #=================================================================
            # Should we try to make the Hessian PSD using approach 2
            # During burn-in: mirror the smallest eigenvalue in zero.
            # After burn-in:  replace Hessian with covariance matrix from
            #                 the last iterations during the burn-in.
            #=================================================================

            # Add a diagonal matrix proportional to the largest negative eigv during burnin
            if ( ( self.makeHessianPSDmethod == "hybrid" ) & ( self.iter <= self.nBurnIn ) ):
                mineigv = np.min( np.linalg.eig( self.hessianp [ self.iter,:,: ] )[0] )
                self.hessianp [ self.iter,:,: ] = self.hessianp [ self.iter,:,: ] - 2.0 * mineigv * np.eye( self.nPars )
                print("Iteration: " + str(self.iter) + " has eigenvalues: " + str( eigens ) + " mirroring by adding " +  str( - 2.0 * mineigv ) );

            # Replace the Hessian with the posterior covariance matrix after burin
            if ( ( self.makeHessianPSDmethod == "hybrid" ) & ( self.iter > self.nBurnIn ) ):
                self.hessianp [ self.iter,:,: ] = self.empHessian;
                print("Iteration: " + str(self.iter) + " has eigenvalues: " + str( eigens ) + " replaced Hessian with pre-computed estimated." );

            #=================================================================
            # Should we try to make the Hessian PSD using approach 3
            # Reject the proposed parameters
            #=================================================================

            # Discard the estimate (make the algorithm reject)
            if ( self.makeHessianPSDmethod == "reject" ):
                self.flag = 0.0;

            #=================================================================
            # Should we try to make the Hessian PSD using approach 4
            # Flip the negative eigenvalues
            #=================================================================

            if ( self.makeHessianPSDmethod == "flipEigenvalues" ):
                foo = np.linalg.eig( self.hessianp [ self.iter,:,: ] );
                self.hessianp [ self.iter,:,: ] = np.dot( np.dot( foo[1], np.diag( np.abs( foo[0] ) ) ), foo[1] );

            #=================================================================
            # Should we try to make the Hessian PSD using approach 5
            # During burn-in: replace with qPMH2-Hessian initalisation
            # After burn-in:  replace Hessian with covariance matrix from
            #                 the last iterations during the burn-in.
            #=================================================================

            # Add a diagonal matrix proportional to the largest negative eigv during burnin
            if ( ( self.makeHessianPSDmethod == "hybrid2" ) & ( self.iter <= self.nBurnIn ) ):
                self.hessianp [ self.iter,:,: ] = np.eye(self.nPars) / self.epsilon;
                print("Iteration: " + str(self.iter) + " has eigenvalues: " + str( eigens ) + " replacing with initial Hessian." );

            # Replace the Hessian with the posterior covariance matrix after burin
            if ( ( self.makeHessianPSDmethod == "hybrid2" ) & ( self.iter > self.nBurnIn ) ):
                self.hessianp [ self.iter,:,: ] = self.empHessian;
                print("Iteration: " + str(self.iter) + " has eigenvalues: " + str( eigens ) + " replaced Hessian with pre-computed estimated." );

            #=================================================================
            # Check if it did not help
            #=================================================================

            if ~( isPSD( self.hessianp [ self.iter,:,: ] ) ):
                if ( self.makeHessianPSDmethod != "reject" ):
                    print("pmh: tried to correct for a non PSD Hessian but failed.");
                    self.flag = 0.0;


    ##########################################################################
    # Quasi-Netwon proposal
    ##########################################################################
    def lbfgs_hessian_update(self):

        I  = np.eye(self.nPars, dtype=int);
        Hk = np.eye(self.nPars) / self.epsilon;
        #Hk = np.linalg.inv( np.diag( self.epsilon ) );
        self.hessianBFGS = np.zeros( (self.memoryLength,self.nPars,self.nPars) )

        # BFGS update for Hessian estimate
        if ( self.iter > self.memoryLength ):

            # Extract estimates of log-likelihood and gradients from the
            # last moves
            self.extractUniqueElements();

            if ( self.nHessianSamples[ self.iter ] > 2 ):
                # Extract the last unique parameters and their gradients
                idx = np.sort( np.unique(self.ll,return_index=True)[1] )[-2:];

                if ( np.max( self.iter - idx ) < self.memoryLength ):

                    # The last accepted step is inside of the memory length
                    skk = self.th[ idx[1] , : ]       - self.th[ idx[0], : ];
                    ykk = self.gradient[ idx[1] , : ] - self.gradient[ idx[0], : ];
                    foo = np.dot( skk, ykk) / np.dot( ykk, ykk);
                    Hk  = np.eye(self.nPars) * foo;
                    self.hessianBFGS[0,:,:] = Hk

                # Add the contribution from the last memoryLength samples
                for ii in range( (self.thU).shape[0] ):

                    # Calculate difference in gradient (ykk) and theta (skk)
                    ykk = self.gradientU[ii,:] - self.gradientU[ii-1,:];
                    skk = self.thU[ii,:]       - self.thU[ii-1,:];

                    # Check if we have moved, otherwise to not add contribution to Hessian
                    if ( np.sum( skk ) != 0.0 ):

                        # Compute rho
                        rhok = 1.0 / ( np.dot( ykk, skk) );

                        # Update Hessian estimate
                        A1   = I - skk[:, np.newaxis] * ykk[np.newaxis, :] * rhok
                        A2   = I - ykk[:, np.newaxis] * skk[np.newaxis, :] * rhok
                        Hk   = np.dot(A1, np.dot(Hk, A2)) + ( rhok * skk[:, np.newaxis] * skk[np.newaxis, :] )
                        self.hessianBFGS[ii+1,:,:] = Hk

                # Return the negative Hessian estimate
                Hk = -Hk;

        return Hk;

    ##########################################################################
    # Helper to Quasi-Netwon proposal:
    # Extract the last n unique parameters and their gradients
    ##########################################################################

    def extractUniqueElements(self):

        # Find the unique elements
        idx            = np.sort( np.unique(self.ll[0:(self.iter-1)],return_index=True)[1] );

        # Extract the ones inside the memory length
        idx            = [ii for ii in idx if ii >= (self.iter - self.memoryLength) ]

        # Sort the indicies according to the log-likelihood
        idx2           = np.argsort( self.ll[idx], axis=0 )[:,0]

        # Get the new indicies
        idx3 = np.zeros( len(idx), dtype=int )
        for ii in idx2:
            idx3[ii] = idx[ii]

        # Extract and export the parameters and their gradients
        self.thU       = self.th[idx3,:];
        self.gradientU = self.gradient[idx3,:];

        # Save the number of indicies
        self.nHessianSamples[ self.iter ] = np.max((0,(self.thU).shape[0] - 1));

    ##########################################################################
    # Helper if parameters are accepted
    ##########################################################################
    def acceptParameters(self,thSys,):
        self.th[self.iter,:]        = self.thp[self.iter,:];
        self.tho[self.iter,:]       = thSys.returnParameters();
        self.x[self.iter,:]         = self.xp[self.iter,:];
        self.ll[self.iter]          = self.llp[self.iter];
        self.gradient[self.iter,:]  = self.gradientp[self.iter,:];
        self.hessian[self.iter,:,:] = self.hessianp[self.iter,:];
        self.accept[self.iter]      = 1.0;
        self.prior[self.iter,:]     = self.priorp[self.iter,:];
        self.J[self.iter,:]         = self.Jp[self.iter,:];

    ##########################################################################
    # Helper if parameters are rejected
    ##########################################################################
    def rejectParameters(self,thSys,):
        if ( ( self.PMHtype == "qPMH2" ) & ( self.iter > self.memoryLength ) ):
            self.th[self.iter,:]        = self.th[self.iter-self.memoryLength,:];
            self.tho[self.iter,:]       = self.tho[self.iter-self.memoryLength,:];
            self.x[self.iter,:]         = self.x[self.iter-self.memoryLength,:];
            self.ll[self.iter]          = self.ll[self.iter-self.memoryLength];
            self.prior[self.iter,:]     = self.prior[self.iter-self.memoryLength,:]
            self.gradient[self.iter,:]  = self.gradient[self.iter-self.memoryLength,:];
            self.hessian[self.iter,:,:] = self.hessian[self.iter-self.memoryLength,:,:];
            self.J[self.iter,:]         = self.J[self.iter-self.memoryLength,:];
        else:
            self.th[self.iter,:]        = self.th[self.iter-1,:];
            self.tho[self.iter,:]       = self.tho[self.iter-1,:];
            self.x[self.iter,:]         = self.x[self.iter-1,:];
            self.ll[self.iter]          = self.ll[self.iter-1];
            self.prior[self.iter,:]     = self.prior[self.iter-1,:]
            self.gradient[self.iter,:]  = self.gradient[self.iter-1,:];
            self.hessian[self.iter,:,:] = self.hessian[self.iter-1,:,:];
            self.J[self.iter,:]         = self.J[self.iter-1,:];

    ##########################################################################
    # Adapt the Hessian using the burn-in
    ##########################################################################
    def adaptHessian(self):
        self.invHessian = np.cov( self.th[range(self.nBurnIn-int(self.adaptHessianNoSamples),self.nBurnIn),:].transpose() );
        print('pmh: adapted Hessian using the last ' + str(self.adaptHessianNoSamples) + ' samples of the chain during burn-in with diagonal ' + str( np.round( np.diag(self.invHessian), 3 ) ) + ".");

    ##########################################################################
    # Helper: compile the results and write to file
    ##########################################################################
    def writeToFile(self,sm=None,fileOutName=None):

        # Set file name from parameter
        if ( ( self.fileOutName != None ) & (fileOutName == None) ):
            fileOutName = self.fileOutName;

        # Calculate the natural gradient
        ngrad = np.zeros((self.nIter,self.nPars));

        if ( self.PMHtype == "PMH1" ):
            ngrad = self.gradient;
        elif ( self.PMHtype == "bPMH1" ):
            for kk in range(0,self.nIter):
                ngrad[kk,:] = np.sign( self.gradient );
        elif ( self.PMHtype == "pPMH1" ):
            for kk in range(0,self.nIter):
                ngrad[kk,:] = np.dot( self.gradient[kk,:], self.invHessian );
        elif ( self.PMHtype == "pbPMH1" ):
            for kk in range(0,self.nIter):
                ngrad[kk,:] = np.dot( np.sign( self.gradient[kk,:] ), self.invHessian );
        elif ( self.PMHtype == "PMH2" ):
            for kk in range(0,self.nIter):
                ngrad[kk,:] = np.dot( self.gradient[kk,:], np.linalg.pinv(self.hessian[kk,:,:]) );

        # Construct the columns labels
        columnlabels = [None]*(3*self.nPars+3);
        for ii in xrange(3*self.nPars+3):  columnlabels[ii] = ii;

        for ii in range(0,self.nPars):
            columnlabels[ii]               = "th" + str(ii);
            columnlabels[ii+self.nPars]    = "thp" + str(ii);
            columnlabels[ii+2*self.nPars]  = "ng" + str(ii);

        columnlabels[3*self.nPars] = "acceptProb";
        columnlabels[3*self.nPars+1] = "loglikelihood";
        columnlabels[3*self.nPars+2] = "acceptflag";

        # Compile the results for output
        out = np.hstack((self.th,self.thp,ngrad,self.aprob,self.ll,self.accept));

        # Write out the results to file
        fileOut = pandas.DataFrame(out,columns=columnlabels);
        if (fileOutName == None):

            # Compile a filename for PMH0
            if ( ( self.PMHtype == "PMH0" ) | ( self.PMHtype == "pPMH0" ) ):
                if hasattr(sm, 'filterType'):
                    if ( sm.filterType == "kf" ):
                        fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '_' + str(sm.filterType) + '/' + str(self.dataset) + '.csv';
                    else:
                        fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '_' + str(sm.filterType) + '_N' + str(sm.nPart) + '/' + str(self.dataset) + '.csv';
                else:
                    # Fallback
                    fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '/' + str(self.dataset) + '.csv';

            elif hasattr(sm, 'smootherType'):
                # Compile a file name for PMH algortihms using a smoother
                if ( sm.smootherType == "rts" ):
                    fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '_' + str(sm.filterType) +  '_' + str(sm.smootherType) + '/' + str(self.dataset) + '.csv';
                else:
                    fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '_' + str(sm.filterType) +  '_' + str(sm.smootherType) + '_N' + str(sm.nPart) + '/' + str(self.dataset) + '.csv';
            else:
                # Fallback
                fileOutName = 'results/' + str(self.filePrefix) + '/' + str(self.PMHtype) + '/' + str(self.dataset) + '.csv';

        ensure_dir(fileOutName);
        fileOut.to_csv(fileOutName);

        print("writeToFile: wrote results to file: " + fileOutName)

#############################################################################################################################
# End of file
#############################################################################################################################
