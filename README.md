# gpo-abc2015
This code was downloaded from < https://github.com/compops/gpo-abc2015 > and contains the code used to produce the results in

J. Dahlin, M. Villani and  T. B. Schön, "Approximate Bayesian inference using Gaussian process optimisation applied to intractable likelihood models". Pre-print, arXiv:1506.06975v1, 2015.

The paper is available as a preprint from < http://arxiv.org/pdf/1506.06975 > and < http://liu.johandahlin.com/ >.

## Dependencies

The code is written and tested for Python 2.7.6. The implementation makes use of NumPy 1.9.2, SciPy 0.15.1, Matplotlib 1.4.3, Pandas 0.13.1, GPy 0.6.0, pyDOE 0.3.7, sobol 0.9 and DIRECT 1.0.1. Please have these packages installed, on Ubuntu they can be installed using "sudo pip install --upgrade package-name ". For more information about, the GPy package see < https://sheffieldml.github.io/GPy/ > and the DIRECT package see < http://pythonhosted.org/DIRECT/ >

## Minimal working examples (scripts-mwe)

These are minimal working examples to present how to make use of the GPO-SMC-ABC algorithm in some simple cases.

### Linear Gaussian state space model (gpo-lgss)
We consider parameter inference in a linear Gaussian state space (LGSS) model given by 
``` python
x_{t+1} | x_t ~ N( x_{t+1}; phi x_t, sigma_v^2)
y_t     | x_t ~ N( y_t;         x_t, sigma_e^2)
```
with the parameters {*phi*,*sigma_v*,*sigma_e*}. We assume that *sigma_e* is known and set it to 0.1. We simulate 2,000 data points using the parameters *phi*=0.75 and *sigma_v*=1 from the model. We make use of N=100 particles in a fully adapted particle filter and 150 iterations in the GPO algorithm. 

``` python
gpo.initPar                         = np.array([ 0.50, 1.00 ]);
gpo.upperBounds                     = np.array([ 0.90, 2.00 ]);
gpo.lowerBounds                     = np.array([ 0.00, 0.50 ]);
```

``` python
# Parameter estimates
gpo.thhat
# array([ 0.49938272,  1.01131687])

# Estimate of inverse Hessian
gpo.estimateHessian()

# Compute half-length of 95% confidence intervals
1.96*np.sqrt(np.diag(gpo.invHessianEstimate))
# array([ 0.03792376,  0.03140083])
```

### Stochastic volatility model with simulated data (gpo-sv)
We consider parameter inference in a stochastic volatility (SV) model given by 
``` python
x_{t+1} = mu + phi * ( x_t - mu ) + sigma_v v_t
y_t     = exp( 0.5 * x_t ) e_t
```
where *v_t* and *e_t* are standard Gaussian random variables with correlation *rho*. In this model, we have the parameters {*mu*,*phi*,*sigma_v*,*rho*}. We simulate 1,000 data points using the parameters {0.20,0.90.0.15,-0.70} from the model. We make use of N=100 particles in a bootstrap particle filter and 150 iterations in the GPO algorithm to estimate the paramters {*mu*,*phi*,*sigma_v*}.

``` python
gpo.initPar                         = np.array([ 0.20, 0.95, 0.14 ]);
gpo.upperBounds                     = np.array([ 0.50, 1.00, 0.50 ]);
gpo.lowerBounds                     = np.array([-0.50, 0.80, 0.05 ]);
```

``` python
# Parameter estimates
gpo.thhat
# array([ 0.2345679 ,  0.91975309,  0.14166667])

# Estimate of inverse Hessian
gpo.estimateHessian()

# Compute half-length of 95% confidence intervals
1.96*np.sqrt(np.diag(gpo.invHessianEstimate))
# array([ 0.12445807,  0.04638067,  0.05792237])
```

### Stochastic volatility model with real-world data (gpo-bitcoin)
We consider parameter inference in a stochastic volatility with alpha stable returns (aSV) model given by 
``` python
x_{t+1} | x_t ~ N( x_{t+1}; mu + phi * ( x_t - mu ), sigma_v^2)
y_t     | x_t ~ A( y_t;     alpha,                   exp(x_t))
```
where *A(y_t,alpha,eta)* denotes a zero-mean alpha stable distribution with stability parameter *alpha* and scale *eta*.  In this model, we have the parameters {*mu*,*phi*,*sigma_v*,*alpha*}.

``` python
gpo.initPar                         = np.array([ 0.20, 0.95, 0.14, 1.8 ]);
gpo.upperBounds                     = np.array([ 0.50, 1.00, 0.90, 2.0 ]);
gpo.lowerBounds                     = np.array([-0.50, 0.80, 0.05, 1.2 ]);
```

``` python
sm.filter          = sm.bPFabc;
sm.nPart           = 2000;
sm.resampFactor    = 2.0;
sm.weightdist      = "gaussian"

sm.rejectionSMC    = False;
sm.adaptTolLevel   = False;
sm.propAlive       = 0.00;
sm.tolLevel        = 0.10;
```

``` python
# Parameter estimates
gpo.thhat
# array([ 0.14814815,  0.94938272,  0.62191358,  1.54074074])

# Estimate of inverse Hessian
gpo.estimateHessian()

# Compute half-length of 95% confidence intervals
1.96*np.sqrt(np.diag(gpo.invHessianEstimate))
# array([ 0.37539896,  0.03176862,  0.16280434,  0.1632567 ])
```

### Some settings
``` python
gpo.maxIter                         = 100;
gpo.preIter                         = 50;

gpo.jitteringCovariance             = 0.01 * np.diag(np.ones(th.nParInference));
gpo.preSamplingMethod               = "latinHyperCube";

gpo.EstimateHyperparametersInterval = 25;
gpo.EstimateThHatEveryIteration     = False;
gpo.EstimateHessianEveryIteration   = False;
```


## Replication scripts for paper (scripts-draft1)
