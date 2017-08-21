# working dir and data dir, file names for the data to be loaded
wd <- "~/Desktop/DS"
setwd(wd)
dd <- "~/Desktop/DS/cisco-ensemble-intrusion-detection-hw/data"
dataFileName <- "intrusion.detection.csv"
featureFile <- "feature.names.types.txt"

library("randomForest")
library("rpart")
library("randomForest")
library(caret)
library(C50)
library (class)
library (e1071)

# Load the data: make sure the data set is the data dir

featureSet <- paste(dd,"/",featureFile,sep="")
dataSet <- paste(dd,"/",dataFileName,sep="")
print(paste("Loading data set: ", dataSet))
data <- read.table(dataSet, header=F, sep=',',stringsAsFactors=T) #494,021 Obs.

#Read Feature/Varibales Names
featureNames <- read.table(featureSet, header=T, sep=',', stringsAsFactors=F)
# Assign variable names
names(data) <- c(as.vector(featureNames[ ,1]),"Type") 


#Subset of data where class is normal. or satan.
new_data <- subset(data, Type=="normal."|Type=="satan." ) #gives 98,867 Obs.

# Dropping extra levels from subset of data to be analyzed
new_data$Type <- droplevels(new_data$Type) # # Drops levels to 2 from original 23
new_data$service <- droplevels(new_data$service) # Drops levels to 34 from original 66


set.seed(444)

# Stratified sampling as classes are imbalanced, Positive class satan. is only  1.61%
class1_sample <- which(new_data$Type=="normal.")
cnt_class1 <- length (class1_sample) # Rec Count 97278 for Normal Class
class2_sample <- which(new_data$Type=="satan.")
cnt_class2 <- length(class2_sample) # Rec count 1589 for satan. class

# Percentage of postive class satan. 
cnt_class2/(cnt_class1+cnt_class2)*100  #1.61%

# Dividing data indexes into 3 buckets for each class - Training 70%, Validation 15%, Test 15%

# Sampling for Type = normal.
train_sample_1 <- sample(class1_sample, .7*cnt_class1) #70% training
test_sample_1 <- sample(setdiff(class1_sample,train_sample_1), .15*cnt_class1) #15% test
validation_sample_1 <- setdiff(setdiff(class1_sample,train_sample_1),test_sample_1) #  Remaining Validation
print("Total Observations for Class 1")
length(class1_sample)
print("Training Observations for Class 1")
length(train_sample_1)
print("Testing Observations for Class 1")
length(test_sample_1)
print("Validation Observations for Class 1")
length(validation_sample_1)

# Sampling for Type = satan.
train_sample_2 <- sample(class2_sample, .7*cnt_class2)
test_sample_2 <- sample(setdiff(class2_sample,train_sample_2), .15*cnt_class2)
validation_sample_2 <- setdiff(setdiff(class2_sample,train_sample_2),test_sample_2)
print("Total Observations for Class 2")
length(class2_sample)
print("Training Observations for Class 2")
length(train_sample_2)
print("Testing Observations for Class 2")
length(test_sample_2)
print("Validation Observations for Class 2")
length(validation_sample_2)

# Combined data of 2 classes in to Training, Validation and Test buckets
train_data <- new_data[c(train_sample_1, train_sample_2), ]
test_data <- new_data[c(test_sample_1, test_sample_2), ]
validation_data <- new_data[c(validation_sample_1, validation_sample_2), ]

# Decison Tree
dt_model <- rpart(Type~., data=train_data)
# Predict the model on the validation data
predictions <- predict(dt_model, validation_data[,-42], type='class')
# Use caret library to  test data from model ( non-spam is positive class)
confusionMatrix(predictions,validation_data$Type, positive = "satan.")

# Naive Bayes
classifier <- naiveBayes (train_data[,-42], train_data[,42]) 
predictions <- predict(classifier,validation_data[,-42])
confusionMatrix(predictions,validation_data$Type, positive = "satan.")

# Boosting
boosted_dt <- C5.0(Type~., data=train_data, trials=10)
predictions <- predict(boosted_dt, validation_data[,-42])
confusionMatrix(predictions,validation_data$Type, positive = "satan.")

# Random Forest
modelRF <- randomForest(x=train_data[,-42], 
                      y=train_data[,42],
                      ntree=6,
                      nodesize=2,
                      importance=T, replace = T)
predictions <- predict(modelRF,validation_data[,-42])
confusionMatrix(predictions,validation_data$Type, positive = "satan.")

# 10 most important variables
varImportance <- importance(modelRF)
varImportance[1:10, ]



##############################################################################
#            FINAL TEST USING TEST DATA
##############################################################################

# Finally Test Data [Test Data is getting used first time, for final test]
predictions <- predict(modelRF,test_data[,-42])
confusionMatrix(predictions,test_data$Type, positive = "satan.")

