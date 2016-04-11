##############################################################################
##############################################################################
# Routines for
# Particle filtering and smoothing
# Version 2014-12-03
#
# Copyright (c) 2014 Johan Dahlin [ johan.dahlin (at) liu.se ]
# Distributed under the MIT license.
#
##############################################################################
##############################################################################

from smc_resampling            import *
from smc_helpers               import *
from smc_additivefunctionals   import *
from smc_filters               import *
from smc_filters_abc           import *
from smc_filters_smooth        import *
from smc_filters_correlatedRVs import *
from smc_smoothers             import *


##############################################################################
# Main class
##############################################################################

class smcSampler(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # Identifier
    typeSampler      = 'smc';

    # No particles in the filter and the number of backward paths in FFBSi
    nPart            = None;
    nPaths           = None;
    nPart2           = None;

    # For the rejection-sampling FFBSi with early stopping
    nPathsLimit      = None;
    rho              = None;

    # Seed for the smooth particle filters
    seed             = None;

    # Lag for the fixed-lag smooother and Newey-West estimator for Hessian
    fixedLag         = None;
    NeweyWestLag     = None;

    # Threshold for ESS to resample and type of resampling scheme
    resampFactor     = None;
    resamplingType   = None;
    # resamplingType: systematic (default), multinomial, stratified

    # Should the gradient and Hessian ( of log-target ) be calculated
    calcGradientFlag = None;
    calcHessianFlag  = None;
    # calcHessianFlag: louis (default), segalweinstein, neweywest (approximate)

    # Should q-function (for EM algorithm) be calculated
    calcQFlag        = None;

    # Initial state for the particles
    xo               = None;
    genInitialState  = None;

    # ABC-specific flags
    rejectionSMC     = None;
    propAlive        = None;
    adaptTolLevel    = None;
    tolLevel         = None;
    weightdist       = None;

    # Partially adapted PF
    doPartialOptForAll = False;

    sortParticles      = None;

    ##########################################################################
    # Particle filtering: wrappers for special cases
    ##########################################################################

    def SIS(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 0;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "SIS";
        self.pf(sys);

    def SISrv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 0;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "SIS-fixedrvs";
        self.rvpf(sys);

    def bPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "bPF";
        self.pf(sys);

    # Bootstrap particle filter with fixed random variables
    def bPFrv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "bPF-fixedRVs";
        self.rvpf(sys);

    # Smooth bootstrap particle filter with fixed random variables
    def sbPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "smooth-bPF";
        self.sPF(sys);

    # Continious fully-adapted particle filter with fixed random variables
    def sfaPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "smooth-fapf";
        self.sPF(sys);

    # Smooth bootstrap particle filter with fixed random variables
    def sbPFrv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "smooth-bPF-fixedrvs";
        self.rvsPF(sys);

    # Continious fully-adapted particle filter with fixed random variables
    def sfaPFrv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "smooth-fapf-fixedrvs";
        self.rvsPF(sys);

    # Smooth bootstrap particle filter with fixed random variables (Version in Pitt, 2002)
    def sbPF2(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "smooth-bPF";
        self.sPFpitt(sys);

    # Continious fully-adapted particle filter with fixed random variables (Version in Pitt, 2002)
    def sfaPF2(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "smooth-fapf";
        self.sPFpitt(sys);

    # Smooth bootstrap particle filter with fixed random variables (Version in Pitt, 2002)
    def sbPF2rv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "smooth-bPF-fixedrvs";
        self.rvsPFpitt(sys);

    # Continious fully-adapted particle filter with fixed random variables (Version in Pitt, 2002)
    def sfaPF2rv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "smooth-fapf-fixedrvs";
        self.rvsPFpitt(sys);

    # Conditional particle filter
    def cPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 1;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "bPF";
        self.smootherType             = "cPF";
        self.pf(sys);

    # Conditional particle filter with ancestor sampling
    def casPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 1;
        self.ancestorSamplingInternal = 1;
        self.filterType               = "bPF";
        self.smootherType             = "cPF-AS";
        self.pf(sys);

    # Fully adapted particle filter
    def faPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted";
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "faPF";
        self.pf(sys);

    # Fully adapted particle filter with fixed random variables
    def faPFrv(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted";
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "faPF-fixedRVs";
        self.rvpf(sys);

    # Partially adapted particle filter
    def paPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "partiallyadapted";
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "paPF";
        self.pf_pa(sys);

    # Fully adapted conditional particle filter with ancestor sampling
    def facasPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted";
        self.condFilterInternal       = 1;
        self.ancestorSamplingInternal = 1;
        self.filterType               = "faPF";
        self.smootherType             = "facPF-AS";
        self.pf(sys);

    # Fully adapted conditional particle filter with ancestor sampling
    def pacasPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "partiallyadapted";
        self.condFilterInternal       = 1;
        self.ancestorSamplingInternal = 1;
        self.filterType               = "paPF";
        self.smootherType             = "pacPF-AS";
        self.pf_pa(sys);

    # Fully adapted conditional particle filter
    def facPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "fullyadapted";
        self.condFilterInternal       = 1;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "faPF";
        self.smootherType             = "facPF";
        self.pf(sys);

    # bootstrap particle filter with ABC
    def bPFabc(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap";
        self.condFilterInternal       = 1;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "bPF";
        self.smootherType             = "bPF-ABC";
        self.pf_abc(sys);

    # bootstrap particle filter with ABC and alive adaptation
    def bPFabcAlive(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap";
        self.condFilterInternal       = 1;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "bPF-ABC-Alive";
        self.pf_abc_alive(sys);

    ##########################################################################
    # Particle filtering and smoothing
    ##########################################################################

    # Auxiliuary particle filter
    pf           = proto_pf

    # Auxiliuary particle filter with fixed random variables
    rvpf         = proto_rvpf

    # Partially adapted particle filter
    pf_pa        = proto_papf

    # Smooth/continious particle filter
    sPFpitt      = proto_sPFpitt;
    sPF          = proto_sPF;

    # Smooth/continious particle filter with fixed random numbers
    rvsPFpitt    = proto_rvsPFpitt;
    rvsPF        = proto_rvsPF;


    # Particle filters based on ABC
    pf_abc       = proto_pf_abc
    pf_abc_alive = proto_pf_abc_alive
    sPF_abc      = proto_sPF_abc

    # Particle smoothers
    flPS         = proto_flPS
    fsPS         = proto_fsPS
    ffbsmPS      = proto_ffbsmPS
    ffbsiPS      = proto_ffbsiPS

    # Wrapper for trajectory reconstruction
    reconstructTrajectories = reconstructTrajectories_helper;

    # Write state estimate to file
    writeToFile = writeToFile_helper

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
