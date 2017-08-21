
#******************HW6*************************************

# Read the eBay auctions data set
eBay_data <- read.csv('eBayAuctions.csv')

# Factorize the Duration column
eBay_data$Duration <- factor(eBay_data$Duration)

set.seed(4444) # to make results  reproducible
# Sample the data set into training set and testing set with a ratio of 60:40 
train_index <- sample(1:nrow(eBay_data), 0.6*nrow(eBay_data))
train_data <-eBay_data[train_index, ]
test_data <- eBay_data[-train_index, ]


# Create a vector of categorical predictors in the data set
categorical_predictors <- c('Category', 'currency', 'Duration', 'endDay')

for(predictor in categorical_predictors){
  
  # Get a list of all unique categories in the categorical variable
  all_categories <- unique(train_data[, predictor])
  
  # Get a distribution of samples for each category per class label of the response variable
  sample_dist <- table(train_data[, predictor], train_data[, 'Competitive.'])
  sample_dist_2 <- apply(sample_dist, 1, function(x){x/sum(x)})
  sample_dist_2 <- t(sample_dist_2)
  
  # Sort the distributions in sample_dist_2 in descending order.
  # Note that you can also sort them in ascending order and then identify the 
  # categories with similar distribution
  sort_idx <- sort(sample_dist_2[, 1], decreasing=TRUE, index.return=TRUE)
  sample_dist_2 <- sample_dist_2[sort_idx$ix, ]
  categories <- row.names(sample_dist_2)
  
  # A boolean map which keeps track of which categories are merged
  bool_mat <- matrix(0, nrow=nrow(sample_dist_2), ncol=nrow(sample_dist_2), dimnames=list(row.names(sample_dist_2), row.names(sample_dist_2)))
  
  # Iterate through the sample distribution of categories and merge them if they are within a threshold of 0.05 with each other
  for(i in 1:nrow(sample_dist_2)){
    j <- i+1
    while(j <= nrow(sample_dist_2)){
      # Check if the i_th category has been merged or not
      if(sum(bool_mat[i, ])==0){
        # If not then check with if the j_th category falls within the threshold of 0.05
        merge <- ifelse(sample_dist_2[i, 1]-sample_dist_2[j, 1]<=0.05, 1, 0)
        if(merge==1){
          # if yes then merge the two categories and replace them in the original data with a new category
          bool_mat[i, j] <- bool_mat[j, i] <- 1
          new_category <- paste(categories[i], categories[j], sep='')
          category1_samples <- which(train_data[, predictor]==categories[i])
          category2_samples <- which(train_data[, predictor]==categories[j])
          levels(train_data[, predictor]) <- c(levels(train_data[, predictor]), new_category)
          train_data[c(category1_samples, category2_samples), predictor] <- new_category
        }
      }
      j<-j+1
    }
  }
  train_data[, predictor] <- factor(train_data[, predictor])
}

# We have merged the categories and now let's generate dummy variables from the remaining categories
eBay_dummy_data <- model.matrix(~.,train_data)

# Logistic regresison model using the glm function
fit.all <- glm(Competitive.~., data=data.frame(eBay_dummy_data),  family = binomial(link="logit"))
summary(fit.all)

#Get variable with highest estimate
single.var<- names(which(coefficients(fit.all)== max(abs(coef(fit.all)),na.rm = TRUE) ) )
print(single.var) #"CategoryElectronicsPhotography"

# Model with single predictor
fit.single <- glm(paste("Competitive.",single.var,sep = "~") , data=data.frame(eBay_dummy_data),  family = binomial(link="logit"))
summary(fit.single)

# Sort coefficients of fit.all for absolute values for Problem 2
sort(abs(coef(fit.all)), decreasing = TRUE)


# Model with reduced set of  predictors
fit.reduced <- glm(Competitive.~ CategoryBooksToys.Hobbies + CategoryClothing.AccessoriesMusic.Movie.Game + 
                     CategoryCollectiblesAntique.Art.Craft  + CategoryHome.GardenComputer  +
                     CategoryBusiness.IndustrialSportingGoods  + CategoryElectronicsPhotography + currencyGBP +
                     currencyUS + sellerRating +Duration7 + endDaySatSun + ClosePrice + OpenPrice,  
                     data=data.frame(eBay_dummy_data),
                     family = binomial(link="logit"))
summary(fit.reduced)

# Anova Test
anova(fit.reduced, fit.all, test = "Chisq")


#Checking for over-dispersion
Overdispersion_reduced <- 1240.8 /1182
Overdispersion_all <- 1225.5 /1182
library(qcc)
qcc.overdispersion.test(train_data$Competitive., size=rep(nrow(train_data), nrow(train_data)))

