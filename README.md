# gpo-abc2015
This code was downloaded from < https://github.com/compops/gpo-abc2015 > and contains the code used to produce the results in

J. Dahlin, M. Villani and  T. B. Schön, "Approximate Bayesian inference using Gaussian process optimisation applied to intractable likelihood models". Pre-print, arXiv:1506.06975v1, 2015.

The paper is available as a preprint from < http://arxiv.org/pdf/1506.06975 > and < http://liu.johandahlin.com/ >.

## Dependencies

The code is written and tested for Python 2.7.6. The implementation makes use of NumPy 1.9.2, SciPy 0.15.1, Matplotlib 1.4.3, Pandas 0.13.1, GPy 0.6.0, pyDOE 0.3.7, sobol 0.9 and DIRECT 1.0.1. Please have these packages installed, on Ubuntu they can be installed using "sudo pip install --upgrade package-name ". For more information about, the GPy package see < https://sheffieldml.github.io/GPy/ > and the DIRECT package see < http://pythonhosted.org/DIRECT/ >

## Minimal working examples (scripts-mwe)

These are minimal working examples to present how to make use of the GPO-SMC-ABC algorithm in some simple cases.

### Linear Gaussian state space model (gpo-lgss)
We consider parameter inference in a linear Gaussian state space (LGSS) model given by x_{t+1} \sim \mathcal{N}(x_{t+1}; \phi x_t, \sigma_v^2) and y_t \sim \mathcal{N}(y_t; x_t, \sigma_e^2) with the parameters {\phi,\sigma_v,\sigma_e}. We assume that \sigma_e is known and set it to 0.1. We simulate 100 data points using the parameters \phi=0.75 and \sigma_v=1 from the model. We make use of N=100 particles and 150 iterations of the GPO algorithm. 

### Stochastic volatility model with simulated data (gpo-sv)
We consider parameter inference in a stochastic volatility (SV) model given by x_{t+1} = \mu + \phi ( x_t - \mu ) +  \sigma_v v_t and y_t = exp(x_t/2) e_t, where v_t and e_t are standard Gaussian random variables with correlation \rho. In this model, we have the parameters {\mu,\phi,\sigma_v,\rho}. We simulate 500 data points using the parameters {0.20,0.90.0.15,-0.70} from the model. We make use of N=50 particles and 250 iterations of the GPO algorithm to estimate the paramters {\mu,\phi,\sigma_v}.

### Stochastic volatility model with real-world data (gpo-bitcoin)

## Replication scripts for paper (scripts-draft1)
