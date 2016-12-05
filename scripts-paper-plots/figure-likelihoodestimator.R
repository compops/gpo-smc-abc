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


# Setup plot colors
library("RColorBrewer")
plotColors = brewer.pal(6, "Dark2");

# Change the working directory to be correct on your system
setwd("C:/home/src/gpo-abc2015/scripts-paper-plots")

###################################################################################
# Make plot
###################################################################################

llGrid = read.table("../results/likelihoodfig/figure-likelihoodestimator-grid.csv", sep = ",", header = TRUE)
llRep = read.table("../results/likelihoodfig/figure-likelihoodestimator-histogram.csv", sep = ",", header = TRUE)

cairo_pdf('figure-likelihoodestimator.pdf', height = 3, width = 9)
layout(matrix(c(1, 2, 3), 1, 3, byrow = TRUE))
par(mar = c(4, 4, 1, 4.5))

plot(llGrid[, 2], llGrid[, 3], type = "p", ylab = "log-posterior estimates", 
    xlab = expression(mu), col = plotColors[1], bty = "n", pch = 19)

hist(llRep[, 2], breaks = floor(sqrt(length(llRep[, 2]))), main = "", freq = F, 
    col = rgb(t(col2rgb(plotColors[2]))/256, alpha = 0.25), border = NA, 
    xlab = "log-posterior estimates", ylab = "density estimate", xlim = c(-44640, 
        -44500))

kde = density(llRep[, 2], from = -44640, to = -44500)
lines(kde, lwd = 2, col = plotColors[2])
lines(kde$x, dnorm(kde$x, mean(kde$x), sd(kde$x)), lwd = 2, col = plotColors[3])

qqnorm(llRep[, 2], bty = "n", main = "", pch = 19, col = plotColors[2])
qqline(llRep[, 2], lwd = 2, col = plotColors[3])

dev.off()


###################################################################################
###################################################################################
# End of file
###################################################################################
###################################################################################