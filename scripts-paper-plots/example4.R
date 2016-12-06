###################################################################################
###################################################################################
#
# Makes plot from the run of the files 
# scripts-paper/example4-gposmc.py and scripts-paper/example4-gpoabc.py
# The plot is the Value-At-Risk of the corresponding portfolio is also computed.
#
#
# For more details, see https://github.com/compops/gpo-abc2015
#
# (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
###################################################################################
###################################################################################


library("zoo")
library("RColorBrewer")
plotColors = brewer.pal(8, "Dark2")

# Change the working directory to be correct on your system
setwd("C:/home/src/gpo-abc2015/scripts-paper-plots")


###################################################################################
# Get the data and compute the probability transformed residuals
###################################################################################

# Load Value-at-Risk estimates
VaRABC <- read.table("../results/example4/example4-gpoabc-var.csv", header = TRUE, 
    sep = ",", stringsAsFactors = F)
VaRSMC <- read.table("../results/example4/example4-gposmc-var.csv", header = TRUE, 
    sep = ",", stringsAsFactors = F)

# Load log-returns
y <- read.table("../results/example4/example4-gpoabc-returns.csv", header = TRUE, sep = ",")

T <- dim(y)[1]
nAssets <- dim(y)[2] - 1


###################################################################################
# Make the plot for the Value-at-Risk estimate
###################################################################################

grid <- as.Date(as.yearmon(as.character(y[, 1]), "%Y%m"))

cairo_pdf("example4-portfolio.pdf", height = 3, width = 8)

layout(matrix(1, 1, 1, byrow = TRUE))
par(mar = c(4, 4, 0, 0))

plot(grid, rowMeans(y[, -1]), pch = 19, col = "darkgrey", bty = "n", ylab = "log-returns", 
    xlab = "time", cex = 0.5, ylim = c(-30, 50), xaxt = "n", cex.lab = 0.75, 
    cex.axis = 0.75)

atVector1 <- seq(grid[1], grid[T], by = "5 years")
axis.Date(1, grid, atVector1, labels = NA, cex.lab = 0.75, cex.axis = 0.75)

atVector2 <- seq(grid[1], grid[T], by = "10 years")
axis.Date(1, grid, atVector2, format = "%Y", cex.lab = 0.75, cex.axis = 0.75)

lines(grid[seq(10, T)], -rowMeans(VaRABC[, -1])[10:T], col = plotColors[4], 
    lwd = 1.5)

lines(grid[seq(10, T)], -rowMeans(VaRSMC[, -1])[10:T], col = plotColors[5], 
    lwd = 1.5)

abline(v = grid[round(2 * T/3)], lty = "dashed")

dev.off()

# Get the number of violations
Test <- 805
sum(rowMeans(VaRSMC[, -1])[Test:T] > rowMeans(y[Test:T, -1]))
sum(rowMeans(VaRABC[, -1])[Test:T] > rowMeans(y[Test:T, -1]))


###################################################################################
###################################################################################
# End of file
###################################################################################
###################################################################################