# Load the libraries
# To install pcalg library you may first need to execute the following commands:
#source("https://bioconductor.org/biocLite.R")
#biocLite("graph")
#biocLite("RBGL")
library(vars)
# Read the input data
setwd("~/Desktop/DS/AMLHW/Wk5")
df <- read.csv("./causal/Input Data/data.csv")
summary(df)
cor(df)
plot( x=df$RPRICE , y=df$Move )
# Build a VAR model
# Select the lag order using the Schwarz Information Criterion with a maximum lag of 10
# see ?VARSelect to find the optimal number of lags and use it as input to VAR()
VARselect(df, lag.max = 10)
var <- VAR(df,p=1)
plot(var)
# Extract the residuals from the VAR model
# see ?residuals
res <- residuals(var)

# Check for stationarity using the Augmented Dickey-Fuller test
# see ?ur.df
summary(ur.df(res[,'Move'])) #p-value: < 2.2e-16
summary(ur.df(res[,'RPRICE'])) # p-value: < 2.2e-16
summary(ur.df(res[,'MPRICE'])) #p-value: < 2.2e-16
# Check whether the variables follow a Gaussian distribution
# see ?ks.test
ks.test(x = res[,'Move'],  y= "pnorm") #p-value < 2.2e-16
ks.test(x = res[,'RPRICE'],  y= "pnorm") #p-value = 1.832e-11
ks.test(x = res[,'MPRICE'],  y= "pnorm") #p-value = 3.509e-11

# Write the residuals to a csv file to build causal graphs using Tetrad software
write.csv(res, file = "residuals.csv",  row.names = FALSE)

# OR Run the PC and LiNGAM algorithm in R as follows,
# see ?pc and ?LINGAM
# PC Algorithm

# LiNGAM Algorithm
