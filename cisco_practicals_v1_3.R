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
sort(names(tacdata))
sort(sapply(tacdata, class))
summary(tacdata)

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

#summary(tac)



## Functions to be used for text analytics
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
  dtm <- removeSparseTerms(dtm, .9)
  return (list (name=namecol, dtm=dtm) )
}
dfFromDTM <- function(dtm) {
  mat <- data.matrix(dtm[["dtm"]])
  df <- as.data.frame(mat, stringsAsFactors = FALSE )
  df <- cbind(dtm[["name"]], df)
  colnames(df)[1] <- "name"
  return(df)
}


set.seed(777)

# Creaining training and test index
train <- sample(1:nrow(tac), .9*nrow(tac), replace = FALSE) #90% training
test <- setdiff(1:nrow(tac), train)

# Generating Document Term Matrix for Customer Symptom
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


## Following code block provide sample testing and may be removed for UI implementation
#######################################################################################
dtm_test <- DocumentTermMatrix(
  cleanCorpus(Corpus(VectorSource(tac[test,]$Customer.Symptom))),
  control=list(dictionary = Terms(dtm[["dtm"]])))
df_test <- as.data.frame(data.matrix(dtm_test), stringsAsFactors = FALSE )

pred_text <- predict(dt_text, df_test, type = "prob")
pred_cat <- predict(dt_cat, tac[test,-c(1,2,7,8,9) ], type = "prob")
tac1 <- cbind( as.data.frame(pred_text[,2]), as.data.frame(pred_cat[,2]) )
colnames(tac1)<- c("pred_text", "pred_cat")
tac1$RMA <- predict(fit, newdata = tac1)
confusionMatrix(tac1$RMA,tac[test,]$RMA.Count, positive = "Y")

# Dumping test set into csv for validations
demoset <- cbind(tac[test,],tac1$pred_cat,tac1$pred_text, tac1$RMA)
write.csv(demoset, file = "demoset.csv")
########################################################################

##### Function to predict Probablity of RMA in UI
pred <- function(iVec, iText) {
  iDf <- data.frame(t(iVec))
  names(iDf) <- c("Account.Name","Technology", "Sub.technology", "HW.Platform")

  dtm_test <- DocumentTermMatrix( cleanCorpus(Corpus(VectorSource(iText))),
              control=list(dictionary = Terms(dtm[["dtm"]])))   # Dictonary of terms from train set
  df_test <-  as.data.frame(data.matrix(dtm_test), stringsAsFactors = FALSE )
  pred_text <- predict(dt_text, df_test, type = "prob")
  pred_cat <- predict(dt_cat, iDf, type = "prob")
  tac1 <- cbind( as.data.frame(pred_text[,2]), as.data.frame(pred_cat[,2]) )
  colnames(tac1)<- c("pred_text", "pred_cat")
  tac1$if_rma <- predict(fit, newdata = tac1)
  colnames(tac1) <- c("pred_text", "pred_cat", "prediction" )
  return(tac1)
}

# Test Pred function (Not needed for UI)
##########################################################################
a <- c("FACEBOOK","XR-Routing-Platforms"
       ,"XR OS ASR9000 - Routing and Forwarding Issues","No Data")
b <- "External 01-MAY-2014 23:57:43 -GMT Problem Categor  X Routin Platforms

Problem Subcategor  XR OS ASR9000  Routing and Forwarding Issues

Problem Typ  Hardware Failure

Software Versio    1

Problem Detail  R  RP CPU May  1 1 4 33 PS  pf nod r 35  PLATFOR DIAG  PUN FABRI DAT PAT FAILED  Se onlin dia rs 21721 System Pun Fabri data Path Tes 0x200000 failure threshold is   slo  N  faile   1 CPU
R  RP CPU May  1 1 4 33 PS  pf nod r 35  PLATFOR DIAG  PUN FABRI DAT PAT FAILED  Se onlin dia rs 21721 System Pun Fabri data Path Tes 0x200000 failure threshold is   slo  N  faile   1 CPU     1 CPU
R  RP CPU May  1 1 4 33 PS  pf nod r 35  PLATFOR DIAG  PUN FABRI DAT PAT FAILED  Clea onlin dia rs 21721 System Pun Fabri data Path Tes 0x200000 failure threshold is   slo  N  faile   1 CPU
Detail"
pred(a,b)
#######################################################

# Following is to order months in Chronological order (not alphabatical) for charts
tac$SR.Create.Month <- factor(tac$SR.Create.Month, c(
  "MAY 2014", "JUN 2014", "JUL 2014", "AUG 2014", "SEP 2014", "OCT 2014", "NOV 2014", "DEC 2014",
  "JAN 2015", "FEB 2015", "MAR 2015", "APR 2015", "MAY 2015", "JUN 2015", "JUL 2015", "AUG 2015", "SEP 2015", "OCT 2015", "NOV 2015", "DEC 2015",
  "JAN 2016", "FEB 2016", "MAR 2016", "APR 2016", "MAY 2016","JUN 2016", "JUL 2016", "AUG 2016", "SEP 2016", "OCT 2016"
))

##### Following code is to support Shiny App #############
#### Dropdowns for inputs
tech <- as.vector(sort(unique(tac$Technology)))
subtech <- as.vector(sort(unique(tac$Sub.technology)))
hwp <- as.vector(sort(unique(tac$HW.Platform)))
hwp <- c("", hwp)
ac <- as.vector(sort(unique(tac$Account.Name)))

## Trend plot funtions for shiny##########
library(ggplot2)
poverall <- function() {
  abc <- aggregate(RMA.Count~SR.Create.Month, data = tac[,], FUN=length)
  write.csv(abc, file = "SR_Act.csv")
  p1<-  ggplot( aes(x=SR.Create.Month,y=RMA.Count), data=abc )+ geom_bar(stat="identity") + ylab("Number of SRs") + xlab("Month SR Created") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("Overall SR Trend by Month") +
    geom_line(aes(x=SR.Create.Month,y=RMA.Count, group=2), data=abc, colour="red" )
  return(p1)
}
ptech <- function(itech) {
  abc <- aggregate(RMA.Count~SR.Create.Month, data = tac[which( tac$Technology ==itech ),], FUN=length)
  p1<-  ggplot( aes(x=SR.Create.Month,y=RMA.Count), data=abc )+ geom_bar(stat="identity") + ylab("Number of SRs") + xlab("Month SR Created") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("SRs for Technology by Month")+
    geom_line(aes(x=SR.Create.Month,y=RMA.Count, group=2), data=abc, colour="red" )
  return(p1)
}
psubtech <- function(isubtech) {
  abc <- aggregate(RMA.Count~SR.Create.Month, data = tac[which(tac$Sub.technology ==isubtech ),], FUN=length)
  p1<-  ggplot( aes(x=SR.Create.Month,y=RMA.Count), data=abc )+ geom_bar(stat="identity") + ylab("Number of SRs") + xlab("Month SR Created") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("SRs for Sub Technology by Month")+
    geom_line(aes(x=SR.Create.Month,y=RMA.Count, group=2), data=abc, colour="red" )
  return(p1)
}
phwp <- function(ihwp) {
  abc <- aggregate(RMA.Count~SR.Create.Month, data = tac[which(tac$HW.Platform ==ihwp ),], FUN=length)
  p1<-  ggplot( aes(x=SR.Create.Month,y=RMA.Count), data=abc )+ geom_bar(stat="identity") + ylab("Number of SRs") + xlab("Month SR Created") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("SRs for Hardware Platform by Month")+
    geom_line(aes(x=SR.Create.Month,y=RMA.Count, group=2), data=abc, colour="red" )
  return(p1)
}

phw <- function(ihw) {
  abc <- aggregate(RMA.Count~SR.Create.Month, data = tac[which(tac$HW.Family==ihw ),], FUN=length)
  p1<-  ggplot( aes(x=SR.Create.Month,y=RMA.Count), data=abc )+ geom_bar(stat="identity") + ylab("Number of SRs") + xlab("Month SR Created") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("SRs for Hardware Family By Month")+
    geom_line(aes(x=SR.Create.Month,y=RMA.Count, group=2), data=abc, colour="red" )
  return(p1)
}
pac <- function(iac) {
  abc <- aggregate(RMA.Count~SR.Create.Month, data = tac[which(tac$Account.Name==iac ),], FUN=length)
  p1<-  ggplot( aes(x=SR.Create.Month,y=RMA.Count), data=abc )+ geom_bar(stat="identity") + ylab("Number of SRs") + xlab("Month SR Created") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("SRs for Account by Month")+
    geom_line(aes(x=SR.Create.Month,y=RMA.Count, group=2), data=abc, colour="red" )
  return(p1)
}
## END Trend plot funtions for shiny##########




######## Code for Shiny Application ##############
### After code above is run, following code can be run for shiny app
library(shiny)

ui <- fluidPage(
  tags$h1("RMA and Service Requests Analysis"),
  tags$h3("This application predicts the probabilty of RMA in a Service Request. It also provides the trends for SR creation by various attributes."),
  tags$p("Please enter the values using list boxes."),
  tags$p("Please press Calculate RMA Prediction button, to get RMA Probability"),
  tags$p("Please press Plot Trends For Service Requests button to see trends for Service Requests Creation."),
  inputPanel(
    selectInput(inputId = "iac", label = "Account", choices = ac ),
    selectInput(inputId = "itech", label = "Technology", choices = tech  ),
    selectInput(inputId = "isubtech", label = "Sub Technology", choices = subtech ),
    selectInput(inputId = "ihwp", label = "HW Platform", choices = hwp )
  ),
  wellPanel(
    textInput(inputId = "itext", label = "Customer Symptoms" , width = '100%' , value = ""),
    tags$p("[ Please copy full Customer Symptom from SR  ]")
  ),
  wellPanel(actionButton((inputId="go"), width = '33%', label = "Calculate RMA Prediction"),
            actionButton((inputId="tr"), width = '33%', label = "Plot Trends For Service Requests")),
  wellPanel(
    tags$h4("RMA Probability & Prediction"),
    textOutput("prob_cat"),
    textOutput("prob_text"),
    textOutput("rma")
  ),
  wellPanel(
    tags$h4("Service Request Trends by Atrribute")
  ),
  splitLayout(
    plotOutput("pac"),
    plotOutput("ptech")
  ),
  splitLayout(
    plotOutput("psubtech"),
    plotOutput("phwp")
  ),
  splitLayout(
    plotOutput("poverall")
  )
)



server <- function(input, output){
  data <- eventReactive(input$go, {
    pred ( c(input$iac,input$itech,input$isubtech,input$ihwp), input$itext)
  })

  # Plot events
  pltall <-  eventReactive(input$tr, {
    poverall()
  })
  plttech <-  eventReactive(input$tr, {
    ptech(input$itech)
  })
  pltsubtech <-  eventReactive(input$tr, {
    psubtech(input$isubtech)
  })
  plthwp <-  eventReactive(input$tr, {
    phwp(input$ihwp)
  })
  plthw <-  eventReactive(input$tr, {
    phw(input$ihw)
  })
  pltac <-  eventReactive(input$tr, {
    pac(input$iac)
  })

  # Render function for RMA prediction
  output$rma <- renderText({
    paste("Overall Prediction for RMA:", data()$prediction)
  })

  output$prob_cat <- renderText({
    paste("Probability of RMA from Customer Symptom: ", round(data()$pred_cat*100,digits=2) , "%")
  })

  output$prob_text <- renderText({
    paste("Probability of RMA from Account, Tech. SubTech & HW Family: ", round(data()$pred_text*100,digits=2) , "%")
  })

  # Render function for Plots
  output$poverall <- renderPlot({
    pltall()
  })
  output$ptech <- renderPlot({
    plttech()
  })

  output$psubtech <- renderPlot({
    pltsubtech()
  })

  output$phwp <- renderPlot({
    plthwp()
  })

  output$phw <- renderPlot({
    plthw()
  })

  output$pac <- renderPlot({
    pltac()
  })
}

shinyApp(ui = ui, server = server)
