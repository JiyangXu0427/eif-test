# Title     : TODO
# Objective : TODO
# Created by: ThinkPad
# Created on: 5/7/2021

set.seed(100)
n <- 200
x <- c(rnorm(n/2, mean=-2, sd=1), rnorm(n/2, mean=3, sd=0.8))

len_x <- length(x)
x.CDF <- kCDF(x,ngrid = len_x)
print(x.CDF$Fhat)

class(x.CDF)

#x.CDF
#plot(x.CDF, alpha=0.05, main="Kernel estimate of distribution function", CI=FALSE)
#curve(pnorm(x, mean=-2, sd=1)/2 + pnorm(x, mean=3, sd=0.8)/2, from =-6, to=6, add=TRUE, lty=2, col="blue")
#