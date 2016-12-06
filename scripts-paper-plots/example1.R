###################################################################################
###################################################################################
#
# Makes plots from the run of the files 
# scripts-paper/example1-gpoabc.py, scripts-paper/example1-gposmc.py
# scripts-paper/example1-spsa.py and scripts-paper/example1-qpmh2smc.py
# The plot shows the estimated parameter posteriors and trace plots
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
# Posterior estimates
# Comparisons between PMH and GPO (left-hand side)
###################################################################################

# Settings for plotting
nMCMC = 15000
burnin = 5000
plotM = seq(burnin, nMCMC, 1)

# Estimates and uncertaintes from run of GPO algorithm
GPOthhat3 = c(0.23045267, 0.87037037, 0.24897119)
GPOvar3 = c(0.00852169932, 0.00161496991, 0.00359943216)

# Load data
dpmh3 <- read.table("../results/example1/qPMH2_bPF_N2000_3par.csv", header = TRUE, sep = ",")[plotM, ]

# Make plot
cairo_pdf("example1.pdf", height = 10, width = 8)
layout(matrix(1:6, 3, 2, byrow = FALSE))
par(mar = c(4, 5, 1, 1))


#==================================================================================
# Mu
#==================================================================================

# Histograms
hist(dpmh3$th0, breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = F, 
    col = rgb(t(col2rgb(plotColors[1]))/256, alpha = 0.25), border = NA, 
    xlab = expression(mu), ylab = "posterior estimate", xlim = c(-0.2, 
        0.6), ylim = c(0, 6))

# Prior for mu
grid = seq(-0.2, 0.6, 0.01)
dist = dnorm(grid, 0, 0.2)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid = seq(-0.2, 0.6, 0.01)
dist = dnorm(grid, GPOthhat3[1], sqrt(GPOvar3[1]))
lines(grid, dist, lwd = 1, col = plotColors[1])


#==================================================================================
# Phi
#==================================================================================

# Histograms
hist(tanh(dpmh3$th1), breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = F, 
    col = rgb(t(col2rgb(plotColors[2]))/256, alpha = 0.25), border = NA, 
    xlab = expression(phi), ylab = "posterior estimate", xlim = c(0.7, 
        1), ylim = c(0, 12))

# Prior for phi
grid = seq(0.7, 1, 0.01)
dist = dnorm(grid, 0.9, 0.05)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid = seq(0.7, 1, 0.01)
dist = dnorm(grid, GPOthhat3[2], sqrt(GPOvar3[2]))
lines(grid, dist, lwd = 1, col = plotColors[2])


#==================================================================================
# Sigma_v
#==================================================================================

# Histograms
hist(exp(dpmh3$th2), breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = F, 
    col = rgb(t(col2rgb(plotColors[3]))/256, alpha = 0.25), border = NA, 
    xlab = expression(sigma[v]), ylab = "posterior estimate", xlim = c(0, 
        0.5), ylim = c(0, 8))

# Prior for sigma_v
grid = seq(0, 0.5, 0.01)
dist = dgamma(grid, shape = 2, rate = 20)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid = seq(0, 0.5, 0.01)
dist = dnorm(grid, GPOthhat3[3], sqrt(GPOvar3[3]))
lines(grid, dist, lwd = 1, col = plotColors[3])


###################################################################################
# Posterior estimates
# Comparisons for GPO with different values of epsilon (right-hand side)
###################################################################################

## Estimates from runs of the GPO algorithm

# epsilon = 0.10
GPOthhat2abc = c(0.28806584, 0.86625514, 0.3600823)
GPOvar2abc = c(0.0145148, 0.0006584, 0.0005114)

# epsilon = 0.20
GPOthhat3abc = c(0.14814815, 0.89917695, 0.26131687)
GPOvar3abc = c(0.01002031, 0.00142153, 0.00292853)

# epsilon = 0.20
GPOthhat4abc = c(0.18930041, 0.88271605, 0.25720165)
GPOvar4abc = c(0.0099959, 0.00207768, 0.00574637)

# epsilon = 0.40
GPOthhat5abc = c(0.22222222, 0.88271605, 0.25308642)
GPOvar5abc = c(0.0117579934, 0.00142609302, 0.00448085561)

# epsilon = 0.50
GPOthhat6abc = c(0.18930041, 0.88271605, 0.2654321)
GPOvar6abc = c(0.01077807, 0.00144916, 0.00441841)


#==================================================================================
# Mu
#==================================================================================

# GPO-SMC
grid = seq(-0.2, 0.6, 0.01)
dist = dnorm(grid, GPOthhat3[1], sqrt(GPOvar3[1]))
plot(grid, dist, lwd = 0.5, col = plotColors[1], type = "l", main = "", 
    xlab = expression(mu), ylab = "posterior estimate", xlim = c(-0.2, 
        0.6), ylim = c(0, 6), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
    col = rgb(t(col2rgb(plotColors[1]))/256, alpha = 0.25))

# GPO-ABC
dist = dnorm(grid, GPOthhat2abc[1], sqrt(GPOvar2abc[1]))
lines(grid, dist, lwd = 1, col = "grey70")

dist = dnorm(grid, GPOthhat3abc[1], sqrt(GPOvar3abc[1]))
lines(grid, dist, lwd = 1, col = "grey60")

dist = dnorm(grid, GPOthhat4abc[1], sqrt(GPOvar4abc[1]))
lines(grid, dist, lwd = 1, col = "grey50")

dist = dnorm(grid, GPOthhat5abc[1], sqrt(GPOvar5abc[1]))
lines(grid, dist, lwd = 1, col = "grey40")

dist = dnorm(grid, GPOthhat6abc[1], sqrt(GPOvar6abc[1]))
lines(grid, dist, lwd = 1, col = "grey30")

legend(0.4, 5.8, c("0.1", "0.2", "0.3", "0.4", "0.5"), col = c("grey70", 
    "grey60", "grey50", "grey40", "grey30"), box.col = "white", lwd = 2)


#==================================================================================
# Phi
#==================================================================================

# GPO-SMC
grid = seq(0.7, 1, 0.01)
dist = dnorm(grid, GPOthhat3[2], sqrt(GPOvar3[2]))
plot(grid, dist, lwd = 0.5, col = plotColors[2], type = "l", main = "", 
    xlab = expression(phi), ylab = "posterior estimate", xlim = c(0.7, 
        1), ylim = c(0, 12), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
    col = rgb(t(col2rgb(plotColors[2]))/256, alpha = 0.25))

# GPO-ABC
dist = dnorm(grid, GPOthhat2abc[2], sqrt(GPOvar2abc[2]))
lines(grid, dist, lwd = 1, col = "grey70")

dist = dnorm(grid, GPOthhat3abc[2], sqrt(GPOvar3abc[2]))
lines(grid, dist, lwd = 1, col = "grey60")

dist = dnorm(grid, GPOthhat4abc[2], sqrt(GPOvar4abc[2]))
lines(grid, dist, lwd = 1, col = "grey50")

dist = dnorm(grid, GPOthhat5abc[2], sqrt(GPOvar5abc[2]))
lines(grid, dist, lwd = 1, col = "grey40")

dist = dnorm(grid, GPOthhat6abc[2], sqrt(GPOvar6abc[2]))
lines(grid, dist, lwd = 1, col = "grey30")


#==================================================================================
# Sigma_v
#==================================================================================

# GPO-SMC
grid = seq(0, 0.5, 0.01)
dist = dnorm(grid, GPOthhat3[3], sqrt(GPOvar3[3]))
plot(grid, dist, lwd = 0.5, col = plotColors[3], type = "l", main = "", 
    xlab = expression(sigma[v]), ylab = "posterior estimate", xlim = c(0, 
        0.5), ylim = c(0, 8), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
    col = rgb(t(col2rgb(plotColors[3]))/256, alpha = 0.25))

# GPO-ABC
dist = dnorm(grid, GPOthhat2abc[3], sqrt(GPOvar2abc[3]))
lines(grid, dist, lwd = 1, col = "grey70")

dist = dnorm(grid, GPOthhat3abc[3], sqrt(GPOvar3abc[3]))
lines(grid, dist, lwd = 1, col = "grey60")

dist = dnorm(grid, GPOthhat4abc[3], sqrt(GPOvar4abc[3]))
lines(grid, dist, lwd = 1, col = "grey50")

dist = dnorm(grid, GPOthhat5abc[3], sqrt(GPOvar5abc[3]))
lines(grid, dist, lwd = 1, col = "grey40")

dist = dnorm(grid, GPOthhat6abc[3], sqrt(GPOvar6abc[3]))
lines(grid, dist, lwd = 1, col = "grey30")

dev.off()

###################################################################################
# Parameter trace
# Comparisons between GPO and SPSA
###################################################################################

nIter = 700

# Load data from runs
dgpo3 <- read.table("../results/example1/gposmc_map_bPF_N1000_3par_650iter.csv", header = TRUE, sep = ",")
dspsa <- read.table("../results/example1/spsa_map_bPF_N1000_3par.csv", header = TRUE, sep = ",")[, -1]

# Make plot
cairo_pdf("example1-comparison-spsa.pdf", height = 3, width = 8)
layout(matrix(1:3, 1, 3, byrow = TRUE))
par(mar = c(4, 5, 0, 0))

grid2 = seq(50, nIter - 2)
grid3 = seq(1, nIter - 2, 2)
kk = seq(1, nIter/2, 25)

plot(grid2, as.numeric(dgpo3$thhat0), type = "l", col = plotColors[1], 
    lwd = 1.5, ylab = expression(hat(mu)), yaxt = "n", xlab = "no. log-posterior samples", 
    xlim = c(0, 700), ylim = c(0.2, 0.55), bty = "n")
lines(grid3, as.numeric(dspsa[1, ]), col = plotColors[1], lwd = 1.5)
points(grid3[kk], as.numeric(dspsa[1, kk]), col = plotColors[1], cex = 0.75, 
    pch = 19)
lines(c(0, nIter), c(1, 1) * mean(dpmh3$th0), lty = "dotted")
axis(2, at = seq(0.2, 0.55, 0.05), labels = c("0.20", "", "0.30", "", "0.40", 
    "", "0.50", ""))

plot(grid2, as.numeric(dgpo3$thhat1), type = "l", col = plotColors[2], 
    lwd = 1.5, ylab = expression(hat(phi)), xlab = "no. log-posterior samples", 
    xlim = c(0, 700), ylim = c(0.8, 0.95), bty = "n")
lines(grid3, as.numeric(dspsa[2, ]), col = plotColors[2], lwd = 1.5)
points(grid3[kk], as.numeric(dspsa[2, kk]), col = plotColors[2], cex = 0.75, 
    pch = 19)
lines(c(0, nIter), c(1, 1) * mean(tanh(dpmh3$th1)), lty = "dotted")

plot(grid2, as.numeric(dgpo3$thhat2), type = "l", col = plotColors[3], 
    lwd = 1.5, ylab = expression(hat(sigma[v])), yaxt = "n", xlab = "no. log-posterior samples", 
    xlim = c(0, 700), ylim = c(0.2, 0.55), bty = "n")
lines(grid3, as.numeric(dspsa[3, ]), col = plotColors[3], lwd = 1.5)
points(grid3[kk], as.numeric(dspsa[3, kk]), col = plotColors[3], cex = 0.75, 
    pch = 19)
lines(c(0, nIter), c(1, 1) * mean(exp(dpmh3$th2)), lty = "dotted")
axis(2, at = seq(0.2, 0.55, 0.05), labels = c("0.20", "", "0.30", "", "0.40", 
    "", "0.50", ""))


dev.off()


###################################################################################
###################################################################################
# End of file
###################################################################################
###################################################################################