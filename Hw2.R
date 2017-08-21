#A.
# R function that computes various performance measures of a BINARY classification model
evaluateClassificationModel <- function (predictedClass, actualClass )
{
  confusion_matrix <- table( actualClass, predictedClass)
  # Postive Class Risky, Negative Class: safe manual data
  # Postive Class: non-spam for spam data
  TP <-confusion_matrix[1,1]
  TN <-confusion_matrix[2,2]
  FP <-confusion_matrix[2,1]
  FN <-confusion_matrix[1,2]
  #print ( c(TP, TN, FP, FN) )
  ACC <-  (TP + TN)/(TP + FP + TN + FN)
  ERR <-  (FP + FN)/(TP + FP + TN + FN)
  PRE <-  TP/(TP+FP)
  REC <-  TP/(TP+FN)
  SEN <-  TP/(TP+FN)
  SPE <-  TN/(TN+FP)
  Out <- list(CM=confusion_matrix, ACC=ACC,ErrorRate=ERR,Precision=PRE,
              Recall=REC,Sensitivity=SEN,Specificity=SPE)
  #print(Out)
  return(Out)
}



##################################################################################
#B.
# R Driver/test program that tests your function on the real data.

# Using manualy created , Actual and Predicted Data
# I created 2 vectors , each with 10 elements and calculated values by hand which are matching with function's output
# Risky as postive class
manualActual <- c("Risky", "Safe",  "Risky", "Safe","Safe", "Risky", "Safe", "Risky", "Safe", "Risky")
manualPredicted <- c("Risky", "Safe", "Safe", "Risky", "Safe", "Risky", "Safe", "Risky", "Risky" , "Risky")
l <- evaluateClassificationModel(manualPredicted,manualActual)
print(l$CM)
print(l$ACC)
print(l$ErrorRate)
print(l$Precision)
print(l$Recall)
print(l$Sensitivity)
print(l$Specificity)

# Now Real data from decision tree model
library(C50)
data <- read.csv('spamD.tsv', sep='\t')

# 10% Test data
test_idx <- sample(1:nrow(data), .10*nrow(data))
test_data <- data[test_idx,]
summary(test_data$spam)
# Rest of 90% is training data
train_data <- data[-test_idx,]
summary(train_data$spam)

dt_model <- C5.0(spam~., data=train_data)
#summary(dt_model)
# Predict the model on the test data
predictions <- predict(dt_model, test_data, type='class')

# Use function to test data from model ( #non-spam is positive class)
l <- evaluateClassificationModel(predictions,test_data$spam)
print(l$CM)
print(l$ACC)
print(l$ErrorRate)
print(l$Precision)
print(l$Recall)
print(l$Sensitivity)
print(l$Specificity)

#################################################################################
#C.
# R driver/test program that uses the corresponding methods in the caret package

#install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)

# Use caret library to  test manual (Risky is positive class)
confusionMatrix(manualPredicted,manualActual)

# Use caret library to  test data from model ( non-spam is positive class)
evaluateClassificationModel(predictions,test_data$spam)
confusionMatrix(predictions,test_data$spam)

#Different metrics are exactly matching with hand calculated as well as function's output
#Only difference is in presentaion of confusion metrics
#In my function Predicted Calss comes on top while using caret package Reference/Ground-truth is on top
