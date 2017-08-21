# ========================================
# Multiple Hypothesis Testing
# Part 1: K-fold Cross-Validation Paired t-Test
# Part 2: Analysis of Variance (ANOVA) Test
# Part 3: Wilcoxon Signed Rank test
# ========================================
rm(list=ls())
setwd("~/Desktop/DS/AMLHW/Wk2")

# Load the required R packages
library(caret)
library(C50)
library (e1071 )
library(kernlab)
library(stats)
library(glmnet)
# **********************************************
# Part 1: K-fold Cross-Validation Paired t-Test
# *****************************************

# Load the iris data set
set.seed(400)
data(iris)
nm <- colnames(iris)
iris_data <- read.csv("datasets/iris_data.txt")
colnames(iris_data) <- nm
# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
folds <- createFolds(iris_data$Species, k = 10, list = TRUE, returnTrain = FALSE)
# Use the training set to train a C5.0 decision tree and Support Vector Machine
# Make predictions on the test set and calculate the error percentages made by both the trained models
allidx <- c(1:nrow(iris_data))
errorC50 <- {}
errorKSVM <- {}
for(i in 1:10){
  testidx <- folds[[i]]
  trainidx <-  setdiff(allidx, testidx)
  dt_model <- C5.0(x=iris_data[trainidx,-5], y= iris_data[trainidx,5] )
  predictions <- predict(dt_model, iris_data[testidx,-5])
  CM <- confusionMatrix(predictions,iris_data[testidx,]$Species)
  errorC50[i] <- 1-(CM$overall)[[1]]  #1 -Accuracy
  k_model <- ksvm(x=as.matrix(iris_data[trainidx,-5]), y= iris_data[trainidx,5] )
  predictions <- predict(k_model, iris_data[testidx,-5])
  CM <- confusionMatrix(predictions,iris_data[testidx,]$Species)
  errorKSVM[i] <- 1-(CM$overall)[[1]]  #1 -Accuracy
}
# Perform K-fold Cross-Validation Paired t-Test to compare the means of the two error percentages
hist(errorC50)
hist(errorKSVM)
mean(errorC50)
mean(errorKSVM)
#Visualizing using box plot
boxplot(errorC50, errorKSVM)
t.test(x=errorC50, y=errorKSVM, alternative = "two.sided", paired = TRUE )
#***************************CONCLUSION *******************************#
# We fail to reject NULL hypotheisis (p-value = 0.3443).
# That means of error with 2 models have same mean
#************************************* *******************************#
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#






# *****************************************
# Part 2: Analysis of Variance (ANOVA) Test
# *****************************************
set.seed(400)
# Load the Breast Cancer data set
df <- read.csv("datasets/Wisconsin_Breast_Cancer_data.txt")
colnames(df)[2] = "class"
# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)
# 	3. Naive Bayes	(?naiveBayes in e1071 package)
# 	4. Logistic Regression (?glm in stats package)
# Make predictions on the test set and calculate the error percentages made by the trained models
allidx <- c(1:nrow(df))
errorC50 <- {}
errorKSVM <- {}
errorglm <- {}
errorNaive <- {}
folds <- createFolds(df$class, k = 10, list = TRUE, returnTrain = FALSE)
for(i in 1:10){
  testidx <- folds[[i]]
  trainidx <-  setdiff(allidx, testidx)

  dt_model <- C5.0(class ~., data = df[trainidx,] )
  predictions <- predict(dt_model, df[testidx,-2])
  CM <- confusionMatrix(predictions,df[testidx,]$class)
  errorC50[i] <- 1-(CM$overall)[[1]]  #1 -Accuracy

  k_model <- ksvm (class ~., data = df[trainidx,] )
  predictions <- predict(k_model, df[testidx,-2])
  CM <- confusionMatrix(predictions,df[testidx,]$class)
  errorKSVM[i] <- 1-(CM$overall)[[1]]  #1 -Accuracy

  n_model <- naiveBayes (class ~., data = df[trainidx,] )
  predictions <- predict(n_model, df[testidx,-2])
  CM <- confusionMatrix(predictions,df[testidx,]$class)
  errorNaive[i] <- 1-(CM$overall)[[1]]  #1 -Accuracy

  y= "class"
  x = names(df[,-2])
#fmla <- paste(y, paste(x, collapse="+"), sep="~")
#fit <- glm(fmla, data = df, family=binomial(logit) , control=glm.control(maxit=50))
#predictions <- predict(fit, as.matrix(df[trainidx, -2]) )
#CM <- confusionMatrix(predictions,df[testidx,]$class)
#errorglm[i] <- 1-(CM$overall)[[1]]  #1 -Accuracy
}
# Compare the performance of the different classifiers using ANOVA test (see ?aov)
errordf <- data.frame(cbind(errorC50, errorKSVM, errorNaive))
#Visualizing using box plot
boxplot(errorC50, errorKSVM, errorNaive)
# Normaility Test
hist(errorC50)
shapiro.test(errorC50)  # Normal
hist(errorKSVM)
shapiro.test(errorKSVM)  # Not Normal
hist(errorNaive)
shapiro.test(errorNaive)  # Normal

summary(errordf)
stacked_df <- stack(errordf)
anova_results <- aov(values~ind, data= stacked_df)
anova_results
summary(anova_results)
#***************************CONCLUSION *******************************#
# We reject NULL hypotheisis (p-value = 0.0.00204).
# That mean we can say the errors have stastistical different means
#************************************* *******************************#
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#






# *****************************************
# Part 3: Wilcoxon Signed Rank test
# *****************************************
set.seed(777)
# Load the following data sets,
# 1. Iris
data(iris)
nm <- colnames(iris)
iris_data <- read.csv("datasets/iris_data.txt")
df <- read.csv("datasets/iris_data.txt")
#iris_data <- read.csv("datasets/datasets/iris_data.txt")
#df <- read.csv("datasets/datasets/iris_data.txt")
colnames(iris_data) <- nm
colnames(df)[-1] = "class"
# 2. Ecoli
df2 <- read.csv("datasets/ecoli_data.csv")
colnames(df2)[9] = "class"
# 3. Wisconsin Breast Cancer
df1 <- read.csv("datasets/Wisconsin_Breast_Cancer_data.txt")
colnames(df1)[2] = "class"
# 4. Glass
df3 <- read.csv("datasets/Glass_data.txt")
colnames(df3)[11] = "class"
# 5. Yeast
df4 <- read.csv("datasets/Yeast_data.csv")
colnames(df4)[10] = "class"

# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds
folds <- createFolds(df$class, k = 10, list = TRUE, returnTrain = FALSE)
folds1 <- createFolds(df1$class, k = 10, list = TRUE, returnTrain = FALSE)
folds2 <- createFolds(df2$class, k = 10, list = TRUE, returnTrain = FALSE)
folds3 <- createFolds(df3$class, k = 10, list = TRUE, returnTrain = FALSE)
folds4 <- createFolds(df4$class, k = 10, list = TRUE, returnTrain = FALSE)

# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)

# Make predictions on the test set and calculate the error percentages made by the trained models
# Iris Data
allidx <- c(1:nrow(df))
errorC50 <- {}
errorKSVM <- {}
for(i in 1:10){
  testidx <- folds[[i]]
  trainidx <-  setdiff(allidx, testidx)

  dt_model <- C5.0(x=iris_data[trainidx,-5], y= iris_data[trainidx,5] )
  predictions <- predict(dt_model, iris_data[testidx,-5])
  CM1 <- confusionMatrix(predictions,iris_data[testidx,]$Species)
  errorC50[i] <- 1-(CM1$overall)[[1]]  #1 -Accuracy

  k_model <- ksvm(x=as.matrix(iris_data[trainidx,-5]), y= iris_data[trainidx,5] )
  predictions <- predict(k_model, iris_data[testidx,-5])
  CM2 <- confusionMatrix(predictions,iris_data[testidx,]$Species)
  errorKSVM[i] <- 1-(CM2$overall)[[1]]  #1 -Accuracy

}
#Visualizing using box plot
boxplot(errorC50, errorKSVM)

# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)
wilcox.test(errorC50, errorKSVM, paired=TRUE)

#***************************CONCLUSION *******************************#
# We fail to reject NULL hypotheisis (p-value = 0.4076).
# This means models are giving same performance
#************************************* *******************************#
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
# Make predictions on the test set and calculate the error percentages made by the trained models
# Wisconsin Breast Cancer Data
allidx <- c(1:nrow(df1))
errorC50 <- {}
errorKSVM <- {}
for(i in 1:10){
  testidx <- folds1[[i]]
  trainidx <-  setdiff(allidx, testidx)

  dt_model <- C5.0(class ~., data = df1[trainidx,] )
  predictions <- predict(dt_model, df1[testidx,-2])
  CM1 <- confusionMatrix(predictions,df1[testidx,]$class)
  errorC50[i] <- 1-(CM1$overall)[[1]]  #1 -Accuracy

  k_model <- ksvm (class ~., data = df1[trainidx,] )
  predictions <- predict(k_model, df1[testidx,-2])
  CM2 <- confusionMatrix(predictions,df1[testidx,]$class)
  errorKSVM[i] <- 1-(CM2$overall)[[1]]  #1 -Accuracy
}
#Visualizing using box plot
boxplot(errorC50, errorKSVM)
# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)
wilcox.test(errorC50, errorKSVM, paired=TRUE)

#***************************CONCLUSION *******************************#
# We reject NULL hypotheisis (p-value = 0.02014).
#This means true location shift is not equal to 0
#************************************* *******************************#
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
# Make predictions on the test set and calculate the error percentages made by the trained models
# Ecoli Data
allidx <- c(1:nrow(df2))
errorC50 <- {}
errorKSVM <- {}
for(i in 1:10){
  testidx <- folds2[[i]]
  trainidx <-  setdiff(allidx, testidx)

  dt_model <- C5.0(class ~., data = df2[trainidx,] )
  predictions <- predict(dt_model, df2[testidx,-9])
  CM1 <- confusionMatrix(predictions,df2[testidx,]$class)
  errorC50[i] <- 1-(CM1$overall)[[1]]  #1 -Accuracy

  k_model <- ksvm (class ~., data = df2[trainidx,] )
  predictions <- predict(k_model, df2[testidx,-9])
  CM2 <- confusionMatrix(predictions,df2[testidx,]$class)
  errorKSVM[i] <- 1-(CM2$overall)[[1]]  #1 -Accuracy
}
boxplot(errorC50, errorKSVM)

# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)
wilcox.test(errorC50, errorKSVM, paired=TRUE)

#***************************CONCLUSION *******************************#
# We reject NULL hypotheisis (p-value = 0.01859).
# This means models are not giving same performance
#************************************* *******************************#
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#



# Make predictions on the test set and calculate the error percentages made by the trained models
# Glass Data
allidx <- c(1:nrow(df3))
errorC50 <- {}
errorKSVM <- {}
for(i in 1:10){
  testidx <- folds3[[i]]
  trainidx <-  setdiff(allidx, testidx)
  df3$class <- factor(df3$class)
  dt_model <- C5.0(class ~., data = df3[trainidx,] )
  predictions <- predict(dt_model, df3[testidx,-11])
  CM1 <- confusionMatrix(predictions,df3[testidx,]$class)
  errorC50[i] <- 1-(CM1$overall)[[1]]  #1 -Accuracy

  k_model <- ksvm (class ~., data = df3[trainidx,] )
  predictions <- predict(k_model, df3[testidx,-11])
  CM2 <- confusionMatrix(predictions,df3[testidx,]$class)
  errorKSVM[i] <- 1-(CM2$overall)[[1]]  #1 -Accuracy

}
#Visualizing using box plot
boxplot(errorC50, errorKSVM)
# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)
wilcox.test(errorC50, errorKSVM, paired=TRUE)
#***************************CONCLUSION *******************************#
# We reject NULL hypotheisis (p-value = 0.02174).
# Reject NULL hypothesis. This means models are not giving same performance
#************************************* *******************************#
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#



# Make predictions on the test set and calculate the error percentages made by the trained models
# Yeast Data
allidx <- c(1:nrow(df4))
errorC50 <- {}
errorKSVM <- {}
for(i in 1:10){
  testidx <- folds4[[i]]
  trainidx <-  setdiff(allidx, testidx)

  dt_model <- C5.0(class ~., data = df4[trainidx,] )
  predictions <- predict(dt_model, df4[testidx,-10])
  CM1 <- confusionMatrix(predictions,df4[testidx,]$class)
  errorC50[i] <- 1-(CM1$overall)[[1]]  #1 -Accuracy

  k_model <- ksvm (class ~., data = df4[trainidx,] )
  predictions <- predict(k_model, df4[testidx,-10])
  CM2 <- confusionMatrix(predictions,df4[testidx,]$class)
  errorKSVM[i] <- 1-(CM2$overall)[[1]]  #1 -Accuracy

}
#Visualizing using box plot
boxplot(errorC50, errorKSVM)
# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)
wilcox.test(  errorC50 , errorKSVM,mu=0, alt="two.sided", paired=TRUE)

#***************************CONCLUSION *******************************#
# We reject NULL hypotheisis (p-value = 0.005857).
#  This means models are not giving same performance
#************************************* *******************************#
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#

