##############################################################################
##############################################################################
# Estimating the Value-at-Risk for a portfolio of oil futures
# using a stochastic volatility (SV) model with Gaussian log-returns.
#
# The SV model is inferred using the GPO-SMC algorithm. The dependence between
# the assets are modelled using a Student's t-copula.
#
#
# For more details, see https://github.com/compops/gpo-abc2015
#
# (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

import sys
sys.path.insert(0, '/media/sf_home/src/gpo-abc2015')

# Load packages and helpers
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from misc.portfolio import getOilData, estModel, estVaR, estVol
from misc.portfolio import ensure_dir

# Set the seed for re-producibility
np.random.seed(87655678)

##############################################################################
# Get log-returns
##############################################################################

# Setup files
output_file = 'results/example3/example3-gposmc'

# Get the data
log_ret, T, Test, nAssets, dates = getOilData()


##############################################################################
# Settings
##############################################################################

settings = {'gpo_initPar':     np.array([0.00, 0.95, 0.50, 1.80]),
            'gpo_upperBounds': np.array([1.00, 1.00, 1.00, 2.00]),
            'gpo_lowerBounds': np.array([0.00, 0.00, 0.01, 1.20]),
            'gpo_estHypParInterval': 25,
            'gpo_preIter': 50,
            'gpo_maxIter': 450,
            'smc_nPart': 2000
            }


##############################################################################
# Run model inference using estimation data
##############################################################################

log_vol = np.zeros((Test, nAssets))
m = np.zeros((3, nAssets))
m_var = np.zeros((3, nAssets))

for ii in range(nAssets):
    log_vol[:, ii], m[:, ii], m_var[:, ii] = estModel(
        'GSV', log_ret[0:Test, ii], settings)

##############################################################################
# Estimate the log-volatility using the model and all data
##############################################################################

log_vol = np.zeros((T, nAssets))

for ii in range(nAssets):
    log_vol[:, ii] = estVol('GSV', log_ret[0:T, ii], m[:, ii], settings)


##############################################################################
# Estimate the Value-at-Risk
##############################################################################

dof, corr, var = estVaR(log_vol, log_ret[:, 0:nAssets], Test, 0.01)

# Plot the VaR-estimate and the log-returns
plt.plot(np.mean(var[10:], axis=1))
plt.plot(np.mean(log_ret[10:, 0:nAssets], axis=1), 'k.')

# Count number of violations on validation data
np.sum(np.mean(var[Test:], axis=1) >
       np.mean(log_ret[Test:, 0:nAssets], axis=1))


#############################################################################
# Write results to file
##############################################################################

ensure_dir(output_file + '.csv')

# Log-volatility
fileOut = pd.DataFrame(log_vol)
fileOut.to_csv(output_file + '-volatility.csv')

# Model parameters
fileOut = pd.DataFrame(m)
fileOut.to_csv(output_file + '-model.csv')

# Variance of model parameters
fileOut = pd.DataFrame(m_var)
fileOut.to_csv(output_file + '-modelvar.csv')

# VaR-estimate
fileOut = pd.DataFrame(var)
fileOut.to_csv(output_file + '-var.csv')

# DoF-estimate
fileOut = pd.DataFrame(dof)
fileOut.to_csv(output_file + '-dof.csv')

# Spearman correlation-estimate
fileOut = pd.DataFrame(corr)
fileOut.to_csv(output_file + '-corr.csv')

# Log-returns
fileOut = pd.DataFrame(log_ret, index=dates)
fileOut.to_csv(output_file + '-returns.csv')

##############################################################################
# End of file
##############################################################################
