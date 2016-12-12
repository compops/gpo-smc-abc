# gpo-abc2015
This code was downloaded from < https://github.com/compops/gpo-abc2015 > and contains the code used to produce the results in

J. Dahlin, M. Villani and  T. B. Schön, **Bayesian optimisation for fast approximate inference in state-space models with intractable likelihoods**. Pre-print, arXiv:1506.06975v2, 2015.

The paper is available as a preprint from < http://arxiv.org/pdf/1506.06975 > and < http://liu.johandahlin.com/ >.

## Dependencies

The code is written and tested for Python 2.7.12. The implementation makes use of NumPy 1.11.1rc1, SciPy 0.17.1, Matplotlib 1.5.2rc1, Pandas 0.17.1, GPy 1.5.6, pyDOE 0.3.8, sobol 0.9, numdifftools 0.9.17 and DIRECT 1.0.1. Please have these packages installed, on Ubuntu they can be installed using "sudo pip install --upgrade package-name ". For more information about, the GPy package see < https://sheffieldml.github.io/GPy/ > and the DIRECT package see < http://pythonhosted.org/DIRECT/ >

## Minimal working examples (scripts-mwe)

These are minimal working examples to present how to make use of the GPO-SMC-ABC algorithm in some simple cases.

### Linear Gaussian state space model (gpo-lgss)
We consider parameter inference in a linear Gaussian state space (LGSS) model given by 
``` python
x_{t+1} | x_t ~ N( x_{t+1}; phi x_t, sigma_v^2)
y_t     | x_t ~ N( y_t;         x_t, sigma_e^2)
```
with the parameters {*phi*,*sigma_v*,*sigma_e*}. We assume that *sigma_e* is known and set it to 0.1. We simulate 2,000 data points using the parameters *phi*=0.75 and *sigma_v*=1 from the model. We make use of N=100 particles in a fully adapted particle filter and 150 iterations in the GPO algorithm. 

We make use of the following settings for the GPO algorithm:

``` python
gpo.verbose = True

gpo.initPar = np.array([0.50, 1.00])
gpo.upperBounds = np.array([0.90, 2.00])
gpo.lowerBounds = np.array([0.00, 0.50])

gpo.maxIter = 100
gpo.preIter = 50

gpo.jitteringCovariance = 0.01 * np.diag(np.ones(th.nParInference))
gpo.preSamplingMethod = "latinHyperCube"

gpo.EstimateHyperparametersInterval = 25
gpo.EstimateThHatEveryIteration = False
gpo.EstimateHessianEveryIteration = False
```

This means that the algorithm writes out its state at every iteration (*gpo.verbose*) and that the parameter space (*gpo.upperBounds* and *gpo.lowerBounds*) is defined by *phi* in [0.00, 0.90] and *sigma_v* in [0.50, 2.00]. The latter is required by the DIRECT algorithm to optimise the acquisition function. If the parameter estimates are on the boundary of the parameter space, you should enlarge the parameter space to obtain good estimates. The initial parameter (*gpo.initPar*) has a minor impact on the estimate and can be chosen as a random point in the parameter space. 

The algorithm is run for 100 iterations (*gpo.maxIter*) and makes use of 50 random samples (*gpo.preIter*) to estimate the hyper-parameters in the GP prior. These choices are determined by trial-and-error and it is useful to study the parameter trace of the GPO algorithm to find a good value for the maximum number of iterations. Note also, that the GPO algorithm is run until the different in the value of the AQ is below 0.001 (*gpo.tolLevel*) between two consecutive iterations. This value can be changed to alter the convergence criteria.

The resulting parameter that optimises the AQ function is *jittered* by adding some independent zero-mean Gaussian noise with covariance 0.01 (*gpo.jitteringCovariance*). This is done to increase the performance of the algorithm. We make use of Latin hypercube sampling (*gpo.preSamplingMethod*) for the first 50 iterations. Alternatives are quasi-random numbers (*sobol*) and uniform sampling (*uniform*). 

The hyper-parameters of the GP prior are re-estimated using empirical Bayes every 25th iteration (*gpo.EstimateHyperparametersInterval*). Note that this step is computationally costly when the number of samples increases. Hence, re-estimating the hyper-parameters at every iteration is not advisable. Finally, we do not estimate the parameters (*gpo.EstimateThHatEveryIteration*) and the inverse Hessian of the log-posterior (*gpo.EstimateHessianEveryIteration*) to save computations. To save the trace of the algorithm, these parameters can be switched to *True* instead. More settings are displayed when running the algorithm but the standard settings should be suitable for most problems. 

Running the algorithm results in the following output:
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
We note that the 95% confidence intervals for the parameter estimates are *phi*: [0.461, 0.537] and *sigma_v*: [0.980, 1.043], which contain the parameters used to generate the data.

### Stochastic volatility model with simulated data (gpo-sv)
We consider parameter inference in a stochastic volatility (SV) model given by 
``` python
x_{t+1} = mu + phi * ( x_t - mu ) + sigma_v * v_t
y_t     = exp( 0.5 * x_t ) * e_t
```
where *v_t* and *e_t* are standard Gaussian random variables with correlation *rho*. In this model, we have the parameters {*mu*,*phi*,*sigma_v*,*rho*}. We simulate 1,000 data points using the parameters {0.20,0.90.0.15,-0.70} from the model. We make use of N=100 particles in a bootstrap particle filter and 150 iterations in the GPO algorithm to estimate the parameters {*mu*,*phi*,*sigma_v*}. We make use of the same settings for the GPO algorithm but changes the parameter space to:

``` python
gpo.initPar = np.array([0.20, 0.95, 0.14])
gpo.upperBounds  = np.array([0.50, 1.00, 0.50])
gpo.lowerBounds  = np.array([-0.50, 0.80, 0.05])
```
Running the algorithm results in the following output:
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
We note that the 95% confidence intervals for the parameter estimates are *mu*: [0.110, 0.359], *phi*: [0.873, 0.966] and *sigma_v*: [0.08, 0.200], which contain the parameters used to generate the data.

### Stochastic volatility model with real-world data (gpo-bitcoin)
We consider parameter inference in a stochastic volatility with alpha stable returns (aSV) model given by 
``` python
x_{t+1} | x_t ~ N( x_{t+1}; mu + phi * ( x_t - mu ), sigma_v^2)
y_t     | x_t ~ A( y_t;     alpha,                   exp(x_t))
```
where *A(y_t,alpha,eta)* denotes a zero-mean alpha stable distribution with stability parameter *alpha* and scale *eta*.  In this model, we have the parameters {*mu*,*phi*,*sigma_v*,*alpha*}. We make use of real-world data in the form of log-returns for the Bitcoin currency in terms of the Euro collected from Quandl < https://www.quandl.com/data/BAVERAGE/EUR-EUR-BITCOIN-Weighted-Price > for the period between September 3, 2013 to September 3, 2014. Note that this period is relatively volatility as the value of the currency crashed in December, 2013. For the GPO algorithm, we make use of the same settings as before and define the parameter space as: 
``` python
gpo.initPar = np.array([0.20, 0.95, 0.14, 1.8]);
gpo.upperBounds = np.array([0.50, 1.00, 0.90, 2.0]);
gpo.lowerBounds = np.array([-0.50, 0.80, 0.05, 1.2]);
```
which is based on prior knowledge that *mu* is close to zero, *phi* is close to one, *sigma_v* is usually between 0.10 and 0.50 and *alpha* is somewhere between one and two. Note that *alpha=2* corresponds to Gaussian returns and *alpha=1* to Cauchy returns. 

We make use of the following settings for the bootstrap particle filter with ABC:
``` python
sm.filter = sm.bPFabc
sm.nPart = 2000
sm.resampFactor = 2.0

sm.weightdist = "gaussian"
sm.tolLevel = 0.10
```
This means that 2,000 particles are used (*sm.nPart*) and re-sampling is applied at every step (*sm.resampFactor*) as 2.0 times the number of particles are always larger then the effective sample size (ESS). Hence, if we change 2.0 to 0.5, we re-sample if ESS is less than half of the number of particles. We make use of smooth ABC and apply Gaussian noise to the observations (*sm.weightdist*) with standard deviation 0.10 (*sm.tolLevel*). Uniform noise can also be applied by setting the parameter to *boxcar* with the half-width determined by *sm.tolLevel*. 

Running the algorithm results in the following output:
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
We note that the 95% confidence intervals for the parameter estimates are *mu*: [-0.227, 0.524], *phi*: [0.918, 0.981], *sigma_v*: [0.459, 0.785] and *alpha*: [1.377, 1.704]. This corresponds to a log-volatility with a large autocorrelation and fairly large innovations and heavy-tailed log-returns. The script also estimates the log-volatility and the estimate seems reasonable when comparing with the log-observations.

## Replication scripts for paper (scripts-paper)

### Example 1: Stochastic volatility model with Gaussian log-returns (Sec 6.1)
**example1-gposmc.py** Makes use of GPO with a standard bootstrap particle filter to estimate the parameters of a SV model with synthetic data. The results are presented in Figure 1 as the solid lines (left) and the solid area (right). 

**example1-gpoabc.py** Makes use of GPO with a bootstrap particle filter with ABC and a Gaussian kernel to estimate the parameters of a SV model with synthetic data. The results are presented in Figure 1 as the grey lines (right) when varying the standard deviation of the kernel. 

**example1-pmhsmc.py** Makes use of particle Metropolis-Hastings (PMH) with a standard bootstrap particle filter to estimate the parameters of a SV model with synthetic data. The results are presented in Figure 1 as the histograms (left). The qPMH2 algorithm provides the *ground truth* to which we compare the GPO algorithm.

**example1-spsa.py** Makes use of the simultaneous perturbation and stochastic approximation (SPSA) algorithm proposed by Spall (1987) < http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4789489&tag=1 > with a standard bootstrap particle filter to estimate the parameters of a SV model with synthetic data. The results are presented in Figure 2.

### Example 2: Stochastic volatility model with alpha-stable log-returns (Sec. 6.2)
**example2-gpoabc.py** Makes use of GPO with a bootstrap particle filter with ABC and a Gaussian kernel to estimate the parameters of alpha-stable SV model for log-returns of coffee futures. The results are presented in Figure 3 as the solif lines (middle and lower).

**example2-pmhabc.py** Makes use of PMH with a bootstrap particle filter with ABC and a Gaussian kernel to estimate the parameters of alpha-stable SV model for log-returns of coffee futures. The results are presented in Figure 3 as the histograms (middle and lower).

### Example 3: Computing Value-at-Risk for a portfolio of oil futures (Sec. 6.3)
**example3-gpoabc.py** Estimates a Student's t-copula with SV models with alpha-stable log-returns as the model for each margin. GPO-SMC-ABC is used to estimate the parameters of each marginal model (one for each type of asset). The parameters of the copula model is estimated using a quasi-Newton method and a moment method, see paper for details.

**example3-gposmc.py** Estimates a Student's t-copula with SV models with Gaussian log-returns as the model for each margin. GPO-SMC is used to estimate the parameters of each marginal model (one for each type of asset). The parameters of the copula model is estimated using a quasi-Newton method and a moment method, see paper for details.

### Example 3: Computing Value-at-Risk for a portfolio of stocks (Sec. 6.4)
**example4-gpoabc.py** Estimates a Student's t-copula with SV models with alpha-stable log-returns as the model for each margin. GPO-SMC-ABC is used to estimate the parameters of each marginal model (one for each type of asset). The parameters of the copula model is estimated using a quasi-Newton method and a moment method, see paper for details.

**example4-gposmc.py** Estimates a Student's t-copula with SV models with Gaussian log-returns as the model for each margin. GPO-SMC is used to estimate the parameters of each marginal model (one for each type of asset). The parameters of the copula model is estimated using a quasi-Newton method and a moment method, see paper for details.

## Replication scripts for figures in paper (scripts-paper-plots)
These R scripts (one for each of the four examples) reproduces the plots in Figures 2-7 in the paper. The scripts make use of the output from the runs of each example, which also are provided in the results-folder in this repo. Some dependencies are required to generate the plots, running "install.packages(c("Quandl","RColorBrewer","stabledist","copula","zoo"))" should install all required libraries. 