
# Install the relevant libraries.

if (!"caret" %in% rownames(installed.packages())) {
  install.packages("caret", dependencies=TRUE)
}
if (!"pROC" %in% rownames(installed.packages())) {
  install.packages("pROC")
}
if (!"rpart" %in% rownames(installed.packages())) {
  install.packages("rpart")
}
if (!"DMwR" %in% rownames(installed.packages())) {
  install.packages("DMwR", dependencies=TRUE)
}
if (!"C50" %in% rownames(installed.packages())) {
  install.packages("C50")
}
if (!"kernlab" %in% rownames(installed.packages())) {
  install.packages("kernlab")
}
if (!"randomForest" %in% rownames(installed.packages())) {
  install.packages("randomForest")
}


# Load the relevant libraries.

library(caret)
library(pROC)
library(rpart)
library(DMwR)
library(C50)
library(kernlab)
library(randomForest)