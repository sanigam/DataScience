############  Code for Ans 1 C #################
# A has mean 0 and sd 1
set.seed(700)
A <- rnorm(50,mean = 0, sd=1)
A
mean(A)
median(A)

# B has mean 0 and sd 1
B <- rnorm(5,mean=10,sd=(sqrt(2)))
B
mean(B)
median(B)

# C has 90% of orginal data and 10% of corrupted data
C <- c( A[1:45],B)
C
mean(C)
median(C)

############  Code for Ans 2 #################
library(MASS)
library(car)
#library(lattice)
#library(Hmisc)
crabs
str(crabs)
summary(crabs)
head(crabs)
scatterplotMatrix(crabs[,c("FL", "RW", "CL", "CW", "BD")],
                  diagonal = "histogram",smoother= FALSE, re.line=FALSE)
cor(crabs[,c("FL", "RW", "CL", "CW", "BD")])

describe(crabs)
table(crabs$sp, crabs$sex)


# Looking at Box and Whiker plot for all continous variables
boxplot(crabs[,c("FL", "RW", "CL", "CW", "BD")])

# Plotting variables with sp
par(mfrow=c(3,2))
boxplot(FL~sp, data = crabs, ylab="FL")
boxplot(RW~sp, data = crabs,  ylab="RW")
boxplot(CL~sp, data = crabs,  ylab="CL")
boxplot(CW~sp, data = crabs, ylab="CW")
boxplot(BD~sp, data = crabs,  ylab="BD")

# Plotting variables with sp and sex
par(mfrow=c(3,2))
boxplot(FL~sp+sex, data = crabs, ylab="FL")
boxplot(RW~sp+sex, data = crabs,  ylab="RW")
boxplot(CL~sp+sex, data = crabs,  ylab="CL")
boxplot(CW~sp+sex, data = crabs, ylab="CW")
boxplot(BD~sp+sex, data = crabs,  ylab="BD")

options(digits = 2)
aggregate(crabs[, c(4:8)], by=list(crabs$sp, crabs$sex), FUN=mean)




# creating a variable if_male
crabs$if_male <- ifelse(crabs$sex=="M", 1, 0)
set.seed(700)
train <- sample(1:nrow(crabs), .7*nrow(crabs), replace = FALSE) #70% training
test <- setdiff(1:nrow(crabs), train)

#### running SVM for prediction######
library(e1071)

# Running with FL, CL, BD and if_male variables
predict_vars <- c("FL", "CL", "BD","if_male")
model <- svm(y = crabs[train,"sp"], x=crabs[train,predict_vars], scale = FALSE, type = 'C-classification', kernel = 'linear', cost = 50000)
predictions<- predict(model, crabs[test,predict_vars])
table(predictions, crabs[test,"sp"])



# Running with FL, CL, BD, CW, and if_male variables
predict_vars <- c("FL", "CL", "BD","CW","if_male")
model <- svm(y = crabs[train,"sp"], x=crabs[train,predict_vars], scale = FALSE, type = 'C-classification', kernel = 'linear', cost = 50000)
predictions<- predict(model, crabs[test,predict_vars])
table(predictions, crabs[test,"sp"])
