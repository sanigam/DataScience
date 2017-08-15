rm(list=ls())
library("rpart")
library(caret)
library(C50)
library(gdata)
library(tm)
library(plyr)
library(class)

setwd("~/Desktop/DS")
tacdata <- read.csv("All_SRs_FB_APPLE_LINKEDIN_May 2014_May 2016.csv")
k=5
rec= trunc(nrow(tacdata)/k)
remaink <- c(1:nrow(tacdata))

test1 <- sample(remaink, rec, replace = FALSE)
remaink <- setdiff(remaink, test1)
test2 <- sample(remaink, rec, replace = FALSE)
remaink <- setdiff(remaink, test2)
test3 <- sample(remaink, rec, replace = FALSE)
remaink <- setdiff(remaink, test3)
test4 <- sample(remaink, rec, replace = FALSE)
remaink <- setdiff(remaink, test4)
test5 <- sample(remaink, rec, replace = FALSE)

allrows <- c(1:nrow(tacdata))
train1 <- setdiff(allrows, test1)
train2 <- setdiff(allrows, test2)
train3 <- setdiff(allrows, test3)
train4 <- setdiff(allrows, test4)
train5 <- setdiff(allrows, test5)


var<- c( "SR.Number",  "SR.Create.Month","Account.Name",
         "Technology", "Sub.technology", "HW.Platform", "HW.Family",
         "RMA.Count", "Customer.Symptom")
rm(tac)
options(stringsAsFactors = TRUE)
tac <- tacdata[,var]
tac$RMA.Count <- ifelse(trim(tac$RMA.Count)=="",0,tac$RMA.Count)
dim(tac)
tac[,"RMA.Count"]<- ifelse(tac$RMA.Count == 0, "N", "Y")
tac$RMA.Count <- factor(tac$RMA.Count)

## Text analytics functions
cleanCorpus <- function(corpus) {
  corpus.cln <- tm_map(corpus, content_transformer(removePunctuation))
  corpus.cln <- tm_map(corpus.cln, content_transformer(stripWhitespace ))
  corpus.cln <- tm_map(corpus.cln, content_transformer(tolower))
  corpus.cln <- tm_map(corpus.cln, content_transformer(removeWords), stopwords(kind = "en"))
  return(corpus.cln)
}
generateDTM <- function (namecol, textcol) {
  cor <- Corpus(VectorSource(textcol))
  cor.cl <- cleanCorpus(cor)
  dtm <- DocumentTermMatrix(cor.cl)
  dtm <- removeSparseTerms(dtm, .8)
  return (list (name=namecol, dtm=dtm) )
}
dfFromDTM <- function(dtm) {
  mat <- data.matrix(dtm[["dtm"]])
  df <- as.data.frame(mat, stringsAsFactors = FALSE )
  df <- cbind(dtm[["name"]], df)
  colnames(df)[1] <- "name"
  return(df)
}

kfold_validation <- function (train, test) {
  ### Training
  dtm <- generateDTM(tac[train,]$RMA.Count, tac[train,]$Customer.Symptom)
  dfDTM <- dfFromDTM(dtm)
  # Model for text predictors
  dt_text <- rpart(name~., data=dfDTM, method = "class")
  #Model for Text predictor
  dt_cat <- rpart(RMA.Count~., data=tac[train,-c(1,2,7,9)], method = "class")
  # Model to combine the 2 probabilities for RMA Y/N prediction
  pred_text <- predict(dt_text, data=dfDTM[,-1], type = "prob")
  pred_cat <- predict(dt_cat, tac[train,-c(1,2,7,8,9) ], type = "prob")
  tac1 <- cbind(tac[train,]$RMA.Count, as.data.frame(pred_text[,2]), as.data.frame(pred_cat[,2]) )
  colnames(tac1)<- c("RMA", "pred_text", "pred_cat")
  fit <- train(RMA~ ., data = tac1, method = "bayesglm")

  ### Testing
  dtm_test <- DocumentTermMatrix(
    cleanCorpus(Corpus(VectorSource(tac[test,]$Customer.Symptom))),
    control=list(dictionary = Terms(dtm[["dtm"]])))
  df_test <- as.data.frame(data.matrix(dtm_test), stringsAsFactors = FALSE )
  pred_text <- predict(dt_text, df_test, type = "prob")
  pred_cat <- predict(dt_cat, tac[test,-c(1,2,7,8,9) ], type = "prob")
  tac1 <- cbind( as.data.frame(pred_text[,2]), as.data.frame(pred_cat[,2]) )
  colnames(tac1)<- c("pred_text", "pred_cat")
  tac1$RMA <- predict(fit, newdata = tac1)
  CM <-confusionMatrix(tac1$RMA,tac[test,]$RMA.Count, positive = "Y")
  return(CM)
}

CM1 <- kfold_validation(train1, test1)
CM2 <- kfold_validation(train2, test2)
CM3 <- kfold_validation(train3, test3)
CM4 <- kfold_validation(train4, test4)
CM5 <- kfold_validation(train5, test5)


accuracy <-  c(CM1$overall[1], CM2$overall[1], CM3$overall[1], CM4$overall[1],CM5$overall[1] )
precison <-  c(CM1$byClass[3], CM2$byClass[3], CM3$byClass[3], CM4$byClass[3],CM5$byClass[3] )
sensitivity <-  c(CM1$byClass[1], CM2$byClass[1], CM3$byClass[1], CM4$byClass[1],CM5$byClass[1] )
specificity <-  c(CM1$byClass[2], CM2$byClass[2], CM3$byClass[2], CM4$byClass[2],CM5$byClass[2] )

modelvalidation <- rbind(accuracy,precison, sensitivity,specificity  )
write.csv(modelvalidation, file = "cisco_practicals_kfold.csv")

nullmodel <- nrow(tac[which(tac$RMA.Count=="Y"),])/nrow(tac)
print (paste("NULL Model for RMA " , round(nullmodel*100, 2), '%' ))

