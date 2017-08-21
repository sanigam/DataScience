# Many of the factor levels have nonstandard characters, 
# such as "%," commas, and other values. 
# When these are converted to dummy variable columns, 
# the values violate the rules for naming new variables. 
# To bypass this issue, the names are being re-encoded 
# to make them more simplistic using this function 
recodeLevels <- function(x)
   {
     x <- as.numeric(x)
     ## Add zeros to the text version:
     x <- gsub(" ", "0",format(as.numeric(x)))
     factor(x)
}

# To obtain different performance measures, 
# two wrapper functions are created:
## For accuracy, Kappa, the area under the ROC curve,
## sensitivity and specificity:
fiveStats <- function(...) {
  c(twoClassSummary(...), defaultSummary(...))
}

## Everything but the area under the ROC curve:
fourStats <- function (data, lev = levels(data$obs), model = NULL)
{
    accKapp <- postResample(data[, "pred"], data[, "obs"])
    out <- c(accKapp,
             sensitivity(data[, "pred"], data[, "obs"], lev[1]),
             specificity(data[, "pred"], data[, "obs"], lev[2]))
    names(out)[3:4] <- c("Sens", "Spec")
    out
}
  