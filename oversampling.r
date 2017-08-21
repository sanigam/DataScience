if (!"unbalanced" %in% rownames(installed.packages())) {
  install.packages("unbalanced", dependencies=TRUE)
}
library(unbalanced)
data(ubIonosphere)
n<-nrow(ubIonosphere)

# Proportion of minority examples
sum(ubIonosphere$Class==1)/n

# Proportion of majority examples
sum(ubIonosphere$Class==0)/n

# SMOTE Oversampling Method
output<-ubIonosphere$Class
input<-ubIonosphere[, -ncol(ubIonosphere)]
data<-ubSMOTE(X=input, Y=output)
class(data)

new_data<-cbind(data$X, data$Y)
new_n<-nrow(new_data)

# Asfter SMOTE: Proportion of minority examples
sum(data$Y==1)/new_n

# Asfter SMOTE: Proportion of majority examples
sum(data$Y==0)/new_n



