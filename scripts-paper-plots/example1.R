###################################################################################
###################################################################################
#
# Makes plots from the run of the files 
# scripts-paper/example1-gpoabc.py, scripts-paper/example1-gposmc.py
# scripts-paper/example1-spsa.py and scripts-paper/example1-pmhsmc.py
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

# Load data and models
dpmh3 <- read.table("../results/example1/example1-pmhsmc-run.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)[plotM, ]
mgpo  <- read.table("../results/example1/example1-gposmc-model.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
mgpov <- read.table("../results/example1/example1-gposmc-modelvar.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

thhat_gposmc = mgpo$X0
var_gposmc = diag(matrix(unlist(mgpov[,-1]), nrow = length(thhat_gposmc)))

# Make plot
cairo_pdf("example1-posteriors.pdf", height = 10, width = 8)
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
dist = dnorm(grid, thhat_gposmc[1], sqrt(var_gposmc[1]))
lines(grid, dist, lwd = 1, col = plotColors[1])


#==================================================================================
# Phi
#==================================================================================

# Histograms
hist(dpmh3$th1, breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = F, 
     col = rgb(t(col2rgb(plotColors[2]))/256, alpha = 0.25), border = NA, 
     xlab = expression(phi), ylab = "posterior estimate", xlim = c(0.7, 
                                                                   1), ylim = c(0, 14))

# Prior for phi
grid = seq(0.7, 1, 0.01)
dist = dnorm(grid, 0.9, 0.05)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid = seq(0.7, 1, 0.01)
dist = dnorm(grid, thhat_gposmc[2], sqrt(var_gposmc[2]))
lines(grid, dist, lwd = 1, col = plotColors[2])


#==================================================================================
# Sigma_v
#==================================================================================

# Histograms
hist(dpmh3$th2, breaks = floor(sqrt(dim(dpmh3)[1])), main = "", freq = F, 
     col = rgb(t(col2rgb(plotColors[3]))/256, alpha = 0.25), border = NA, 
     xlab = expression(sigma[v]), ylab = "posterior estimate", xlim = c(0, 
                                                                        0.5), ylim = c(0, 8))

# Prior for sigma_v
grid = seq(0, 0.5, 0.01)
dist = dgamma(grid, shape = 2, rate = 20)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid = seq(0, 0.5, 0.01)
dist = dnorm(grid, thhat_gposmc[3], sqrt(var_gposmc[3]))
lines(grid, dist, lwd = 1, col = plotColors[3])


###################################################################################
# Posterior estimates
# Comparisons for GPO with different values of epsilon (right-hand side)
###################################################################################

## Estimates from runs of the GPO algorithm
thhat_gpoabc = matrix(0, nrow = 5, ncol = length(thhat_gposmc))
var_gpoabc = matrix(0, nrow = 5, ncol = length(thhat_gposmc))

mgpo  <- read.table("../results/example1/example1-gpoabc-model.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

for (ii in 1:5) {
  file_name = paste(paste("../results/example1/example1-gpoabc-modelvar-", ii-1, sep = ""), '.csv', sep = "")
  mgpov <- read.table(file_name, header = TRUE, sep = ",", stringsAsFactors = FALSE)
  
  thhat_gpoabc[ii,] = matrix(unlist(mgpo[,-1]), nrow = 5)[ii,]
  var_gpoabc[ii,] = diag(matrix(unlist(mgpov[,-1]), nrow = length(thhat_gposmc)))
}

#==================================================================================
# Mu
#==================================================================================

# GPO-SMC
grid = seq(-0.2, 0.6, 0.01)
idx = seq(1,length(grid),10)

dist = dnorm(grid, thhat_gposmc[1], sqrt(var_gposmc[1]))
plot(grid, dist, lwd = 0.5, col = plotColors[1], type = "l", main = "", 
     xlab = expression(mu), ylab = "posterior estimate", xlim = c(-0.2, 
                                                                  0.6), ylim = c(0, 6), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
        col = rgb(t(col2rgb(plotColors[1]))/256, alpha = 0.25))

# GPO-ABC
for (ii in 1:5) {
  dist = dnorm(grid, thhat_gpoabc[ii,1], sqrt(var_gpoabc[ii,1]))
  lines(grid, dist, col = "grey30")
  points(grid[idx], dist[idx], pch=ii, col = "grey30", cex=0.75)
}

legend(-0.2, 5.8, c("0.1", "0.2", "0.3", "0.4", "0.5"), pch=1:5, cex=0.75,
       box.col = "white", lwd = 1)


#==================================================================================
# Phi
#==================================================================================

# GPO-SMC
grid = seq(0.7, 1, 0.01)
idx = seq(1,length(grid),3)

dist = dnorm(grid, thhat_gposmc[2], sqrt(var_gposmc[2]))
plot(grid, dist, lwd = 0.5, col = plotColors[2], type = "l", main = "", 
     xlab = expression(phi), ylab = "posterior estimate", xlim = c(0.7, 
                                                                   1), ylim = c(0, 14), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
        col = rgb(t(col2rgb(plotColors[2]))/256, alpha = 0.25))

# GPO-ABC
for (ii in 1:5) {
  dist = dnorm(grid, thhat_gpoabc[ii,2], sqrt(var_gpoabc[ii,2]))
  lines(grid, dist, col = "grey30")
  points(grid[idx], dist[idx], pch=ii, col = "grey30", cex=0.75)
}


#==================================================================================
# Sigma_v
#==================================================================================

# GPO-SMC
grid = seq(0, 0.5, 0.01)
idx = seq(1,length(grid),5)

dist = dnorm(grid, thhat_gposmc[3], sqrt(var_gposmc[3]))
plot(grid, dist, lwd = 0.5, col = plotColors[3], type = "l", main = "", 
     xlab = expression(sigma[v]), ylab = "posterior estimate", xlim = c(0, 
                                                                        0.5), ylim = c(0, 8), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
        col = rgb(t(col2rgb(plotColors[3]))/256, alpha = 0.25))

# GPO-ABC
for (ii in 1:5) {
  dist = dnorm(grid, thhat_gpoabc[ii,3], sqrt(var_gpoabc[ii,3]))
  lines(grid, dist, col = "grey30")
  points(grid[idx], dist[idx], pch=ii, col = "grey30", cex=0.75)
}

dev.off()

###################################################################################
# Parameter trace
# Comparisons between GPO and SPSA
###################################################################################

nIter = 700

# Load data from runs
dgpo3 <- read.table("../results/example1/example1-gposmc-run.csv", header = TRUE, sep = ",")
dspsa <- read.table("../results/example1/example1-spsa-model.csv", header = TRUE, sep = ",")[, -1]

# Make plot
cairo_pdf("example1-spsa.pdf", height = 3, width = 8)
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
lines(c(0, nIter), c(1, 1) * mean(dpmh3$th1), lty = "dotted")

plot(grid2, as.numeric(dgpo3$thhat2), type = "l", col = plotColors[3], 
     lwd = 1.5, ylab = expression(hat(sigma[v])), yaxt = "n", xlab = "no. log-posterior samples", 
     xlim = c(0, 700), ylim = c(0.2, 0.55), bty = "n")
lines(grid3, as.numeric(dspsa[3, ]), col = plotColors[3], lwd = 1.5)
points(grid3[kk], as.numeric(dspsa[3, kk]), col = plotColors[3], cex = 0.75, 
       pch = 19)
lines(c(0, nIter), c(1, 1) * mean(dpmh3$th2), lty = "dotted")
axis(2, at = seq(0.2, 0.55, 0.05), labels = c("0.20", "", "0.30", "", "0.40", 
                                              "", "0.50", ""))

dev.off()


###################################################################################
###################################################################################
# End of file
###################################################################################
###################################################################################