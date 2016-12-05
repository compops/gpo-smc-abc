###################################################################################
###################################################################################
#
# Makes plots from the run of the files 
# scripts-paper/example2-qpmh2abc.py and scripts-paper/example2-gpoabc.py
# The plot shows the estimated log-volatility and the parameter posteriors
#
#
# For more details, see https://github.com/compops/gpo-abc2015
#
# (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
###################################################################################
###################################################################################


library("Quandl")
library("stabledist")

# Setup plot colors
library("RColorBrewer")
plotColors = brewer.pal(6, "Dark2");

# Change the working directory to be correct on your system
setwd("C:/home/src/gpo-abc2015/scripts-paper-plots")


###################################################################################
# Get the data and compute the probability transformed residuals
###################################################################################

# Settings for plotting
nMCMC <- 11000
burnin <- 5000
plotM <- seq(burnin, nMCMC, 1)

# Load the data from Quandl
dCoffee <- Quandl("CHRIS/ICE_KC2", start_date = "2013-06-01", end_date = "2015-01-01", 
    authcode = "k9SxHYWnsLJNVS_LmS1L")

# Load data from runs
GPOthhat <- read.table("../results/example2/example2-coffee-asvmodel-gpoabc-tthat.csv", 
    header = TRUE, sep = ",")[, -1]
dpmh3 <- read.table("../results/example2/qPMH2/0.csv", header = TRUE, sep = ",")[plotM, 
    ]

for (ii in 1:9)
{
    d <- read.table(paste(paste("../results/example2/qPMH2/", ii, sep = ""), 
        ".csv", sep = ""), header = TRUE, sep = ",", stringsAsFactors = FALSE)[plotM, 
        ]
    dpmh3 <- rbind(dpmh3, d)
}


###################################################################################
# Compute approximate 95% confidence intervals for log-returns using simulation
# The stability parameters are inferred in the GPO-step
###################################################################################

d <- read.table("../results/example2/state_abcPF_bPF-ABC_N5000.csv", header = TRUE, 
    sep = ",")

CI <- matrix(0, nrow = length(d$xhats), ncol = 2)
for (ii in 1:length(d$xhats))
{
    CI[ii, ] <- qstable(c(0.025, 0.975), mean(GPOthhat$th3), 0, exp(0.5 * 
        d$xhats[ii]), 0)
}


###################################################################################
# Make the plots
###################################################################################

cairo_pdf('example2.pdf', height = 10, width = 8)
layout(matrix(c(1, 1, 2, 3, 4, 5), 3, 2, byrow = TRUE))
par(mar = c(4, 4, 1, 4.5))

idx <- which(is.na(dCoffee$Settle))

dateCoffee <- rev(dCoffee$Date[-idx])
exraCoffee <- rev(dCoffee$Settle[-idx])
logrCoffee <- 100 * diff(log(exraCoffee))

plot(as.Date(dateCoffee[-1]), logrCoffee, type = "l", ylab = "daily log-returns", 
    xlab = "date", col = plotColors[5], bty = "n", ylim = c(-10, 10), xaxt = "n")

r <- as.POSIXct(range(dateCoffee[-1]), "1 months")
atVector1 <- seq(r[1], as.POSIXct("2015-01-31 01:00:00 CET"), by = "1 months")
atVector2 <- seq(r[1], as.POSIXct("2015-01-31 01:00:00 CET"), by = "2 months")
axis.Date(1, at = atVector1, labels = NA)
axis.Date(1, at = atVector2, format = "%b %y")

grid <- as.Date(dateCoffee[-c(1, 2)])
polygon(c(grid, rev(grid)), c(CI[, 1], rev(CI[, 2])), border = NA, col = rgb(t(col2rgb(plotColors[5]))/256, 
    alpha = 0.15))

par(new = TRUE)
plot(grid, d$xhats, lwd = 1.5, col = "grey30", type = "l", xaxt = "n", 
    yaxt = "n", xlab = "", ylab = "", bty = "n", ylim = c(-1, 3))
axis(4)
mtext("smoothed log-volatility", side = 4, line = 3, cex = 0.75)

par(mar = c(4, 4, 1, 1))

#==================================================================================
# Mu
#==================================================================================

# Histograms
hist(dpmh3$th0, breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = FALSE, 
    col = rgb(t(col2rgb(plotColors[1]))/256, alpha = 0.25), border = NA, 
    xlab = expression(mu), ylab = "posterior estimate", xlim = c(-0.4, 
        0.8), ylim = c(0, 4))

# Prior for phi
grid <- seq(-0.4, 0.8, 0.01)
dist <- dnorm(grid, 0, 0.2)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid <- seq(-0.4, 0.8, 0.01)

for (ii in 0:9)
{
    GPOhessian <- read.table(paste(paste("../results/example2/example2-coffee-asvmodel-gpoabc-thessian-", 
        ii, sep = ""), ".csv", sep = ""), header = TRUE, sep = ",")[, -1]
    dist <- dnorm(grid, GPOthhat[ii + 1, 1], sqrt(GPOhessian[1, 1]))
    lines(grid, dist, lwd = 1, col = plotColors[1])
}

#==================================================================================
# Phi
#==================================================================================

# Histograms
hist(dpmh3$th1, breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = FALSE, 
    col = rgb(t(col2rgb(plotColors[2]))/256, alpha = 0.25), border = NA, 
    xlab = expression(phi), ylab = "posterior estimate", xlim = c(0.75, 
        1), ylim = c(0, 12))

# Prior for sigma_v
grid <- seq(0.75, 1, 0.01)
dist <- dnorm(grid, 0.9, 0.05)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid <- seq(0.75, 1, 0.01)
for (ii in 0:9)
{
    GPOhessian <- read.table(paste(paste("../results/example2/example2-coffee-asvmodel-gpoabc-thessian-", 
        ii, sep = ""), ".csv", sep = ""), header = TRUE, sep = ",")[, -1]
    dist <- dnorm(grid, GPOthhat[ii + 1, 2], sqrt(GPOhessian[2, 2]))
    lines(grid, dist, lwd = 1, col = plotColors[2])
}

#==================================================================================
# Sigma
#==================================================================================

# Histograms
hist(dpmh3$th2, breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = FALSE, 
    col = rgb(t(col2rgb(plotColors[3]))/256, alpha = 0.25), border = NA, 
    xlab = expression(sigma[v]), ylab = "posterior estimate", xlim = c(0, 
        0.7), ylim = c(0, 8))

# Prior for sigma
grid <- seq(0, 0.7, 0.01)
dist <- dgamma(grid, shape = 2, rate = 20)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid <- seq(0, 0.7, 0.01)
for (ii in 0:9)
{
    GPOhessian <- read.table(paste(paste("../results/example2/example2-coffee-asvmodel-gpoabc-thessian-", 
        ii, sep = ""), ".csv", sep = ""), header = TRUE, sep = ",")[, -1]
    dist <- dnorm(grid, GPOthhat[ii + 1, 3], sqrt(GPOhessian[3, 3]))
    lines(grid, dist, lwd = 1, col = plotColors[3])
}

#==================================================================================
# Alpha
#==================================================================================

# Histograms
hist(dpmh3$th3, breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = FALSE, 
    col = rgb(t(col2rgb(plotColors[4]))/256, alpha = 0.25), border = NA, 
    xlab = expression(alpha), ylab = "posterior estimate", xlim = c(1, 
        2), ylim = c(0, 5))

# Prior for alpha
grid <- seq(1, 2, 0.01)
dist <- dbeta(grid/2, 6, 2)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid <- seq(1, 2, 0.01)
for (ii in 0:9)
{
    GPOhessian <- read.table(paste(paste("../results/example2/example2-coffee-asvmodel-gpoabc-thessian-", 
        ii, sep = ""), ".csv", sep = ""), header = TRUE, sep = ",")[, -1]
    dist <- dnorm(grid, GPOthhat[ii + 1, 4], sqrt(GPOhessian[4, 4]))
    lines(grid, dist, lwd = 1, col = plotColors[4])
}

dev.off()

###################################################################################
###################################################################################
# End of file
###################################################################################
###################################################################################