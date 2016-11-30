##############################################################################
##############################################################################
# Routines for
# Particle filtering and smoothing
#
# Copyright (c) 2016 Johan Dahlin 
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

from smc_resampling import *
from smc_helpers import *
from smc_additivefunctionals import *
from smc_filters import *
from smc_filters_abc import *
from smc_smoothers import *


##############################################################################
# Main class
##############################################################################

class smcSampler(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # Identifier
    typeSampler = 'smc'

    # No particles in the filter
    nPart = None

    # Seed for the smooth particle filters
    seed = None

    # Lag for the fixed-lag smooother
    fixedLag = None

    # Threshold for ESS to resample and type of resampling scheme
    resampFactor = None
    resamplingType = None
    # resamplingType: systematic (default), multinomial, stratified

    # Should the gradient and Hessian ( of log-target ) be calculated
    calcGradientFlag = None
    calcHessianFlag = None
    # calcHessianFlag: louis (default), segalweinstein, neweywest (approximate)

    # Should q-function (for EM algorithm) be calculated
    calcQFlag = None

    # Initial state for the particles
    xo = None
    genInitialState = None

    # ABC-specific flags
    rejectionSMC = None
    propAlive = None
    adaptTolLevel = None
    tolLevel = None
    weightdist = None

    sortParticles = None

    ##########################################################################
    # Particle filtering: wrappers for special cases
    ##########################################################################

    def SIS(self, sys):
        self.filePrefix = sys.filePrefix
        self.resamplingInternal = 0
        self.filterTypeInternal = "bootstrap"
        self.condFilterInternal = 0
        self.ancestorSamplingInternal = 0
        self.filterType = "SIS"
        self.pf(sys)

    def bPF(self, sys):
        self.filePrefix = sys.filePrefix
        self.resamplingInternal = 1
        self.filterTypeInternal = "bootstrap"
        self.condFilterInternal = 0
        self.ancestorSamplingInternal = 0
        self.filterType = "bPF"
        self.pf(sys)

    # Fully adapted particle filter
    def faPF(self, sys):
        self.filePrefix = sys.filePrefix
        self.resamplingInternal = 1
        self.filterTypeInternal = "fullyadapted"
        self.condFilterInternal = 0
        self.ancestorSamplingInternal = 0
        self.filterType = "faPF"
        self.pf(sys)

    # bootstrap particle filter with ABC
    def bPFabc(self, sys):
        self.filePrefix = sys.filePrefix
        self.resamplingInternal = 1
        self.filterTypeInternal = "bootstrap"
        self.condFilterInternal = 1
        self.ancestorSamplingInternal = 0
        self.filterType = "bPF"
        self.smootherType = "bPF-ABC"
        self.pf_abc(sys)

    ##########################################################################
    # Particle filtering and smoothing
    ##########################################################################

    # Auxiliuary particle filter
    pf = proto_pf

    # Particle filters based on ABC
    pf_abc = proto_pf_abc

    # Particle smoothers
    flPS = proto_flPS
    ffbsiPS = proto_ffbsiPS

    # Wrapper for trajectory reconstruction
    reconstructTrajectories = reconstructTrajectories_helper

    # Write state estimate to file
    writeToFile = writeToFile_helper

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
