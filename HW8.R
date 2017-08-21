#install.packages("arules")
library(arules)

# Creating data-set in HW8
a1 <-  c("Milk", "Pizza" , "Cake",  "Pretzels", "Salad")
a2 <- c("Water", "Lasagna", "Ice Cream",  "Chips", "Coleslaw")
a3 <- c("Milk", "Spaghetti",  "Cake",  "Pretzels",  "Salad")
a4 <- c("Water",  "Spaghetti", "Cake", "Chips", "Coleslaw")
a5 <- c("Soda", "Spaghetti",  "Cake",  "Chips",  "Salad" )
data <- as.data.frame (rbind( a1, a2,a3,a4,a5) )

# Creating transactions from data
tr <- as(data, "transactions")

#Frequency of items
itemFrequencyPlot(tr, support = 0.1)


#Creating rules with minimum support of .4
rules <- apriori(tr, parameter=list(support=.4, confidence=0, minlen=3,maxlen=3))

summary(rules)

# Getting Highest Support items in C3
inspect(head(sort(rules, by="support"),60))

# Getting Highest Confidence items in C3
inspect(head(sort(rules, by="confidence", decreasing = TRUE),60))
