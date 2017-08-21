
# ***************************************************************************
#######################HW7####################################
# ***************************************************************************
# Functions
sqr_edist <- function(x, y) {             	 
  sum((x-y)^2)
}

wss.cluster <- function(clustermat) {     	
  c0 <- apply(clustermat, 2, FUN=mean)    	 
  sum(apply(clustermat, 1, FUN=function(row){sqr_edist(row,c0)}))     	
}

wss.total <- function(dmatrix, labels) {                               	
  wsstot <- 0
  k <- length(unique(labels))
  for(i in 1:k)
    wsstot <- wsstot + wss.cluster(subset(dmatrix, labels==i))         	
  wsstot
}

totss <- function(dmatrix) {                 	
  grandmean <- apply(dmatrix, 2, FUN=mean)
  sum(apply(dmatrix, 1, FUN=function(row){sqr_edist(row, grandmean)}))
}

library(gdata)
data_dir <- "cisco-cluster-analysis-text-analytics/data"
air <- read.xls( paste(data_dir,"EastWestAirlinesCluster.xls",sep = "/"), sheet="data", header = TRUE )
# Standardize the data set by subtracting the mean and dividing by the standard deviation
air_scaled <- apply(air[, 2:11], 2, function(x) (x-mean(x))/sd(x))
dim(air_scaled)
d <- dist(air_scaled, method="euclidean")
class(d)


# Find the optimal number of clusters
num_samples <- nrow(air_scaled)
max_clusters <- 20

totss <- totss(air_scaled) 
wss <- numeric(max_clusters)
wss[1] <- (num_samples-1)*sum(apply(air_scaled, 2, var))

for(k in 2:max_clusters){
  d <- dist(air_scaled, method="euclidean")
  air_hclust <- hclust(d, method="ward.D")
  labels <- cutree(air_hclust, k=k)
  wss[k] <- wss.total(air_scaled, labels)
}

plot(wss, type='b', xlab="Clusters", ylab="Total Within Sum of Squares")

abline(v=9, lty=2  )

# Perform hierarchical clustering using hclust()
air_clust <- hclust(d, method="ward.D")
plot(air_clust)


air_groups <- cutree(air_clust, 9)
table(air_groups)

# Print the contents of each cluster
summary(air[air_groups==1, ])
summary(air[air_groups==2, ])

head(air[air_groups==1, ])
head(air[air_groups==2, ])

##########Centroid#################
#A centroid is a vector containing one number for each variable, where each number is the mean 
#of a variable for the observations in that cluster

# Centroid of scaled data
aggregate(air_scaled, list(air_groups), mean)

# Centroid of full data data
options(digits=2)
aggregate(air[,2:11], list(air_groups), mean)
aggregate(air[,2:12], list(air_groups), median)

dim(air)

#### Repeat clustering for 95% of sampled records####
air_95 <-  air[sample(1:nrow(air), .95*nrow(air), replace=FALSE ),]

air_scaled <- apply(air_95[, 2:11], 2, function(x) (x-mean(x))/sd(x))


d <- dist(air_scaled, method="euclidean")
class(d)

# Perform hierarchical clustering using hclust()
air_clust <- hclust(d, method="ward.D")
plot(air_clust)

air_groups <- cutree(air_clust, 9)
table(air_groups)
# Centroid of full data data
options(digits=2)
aggregate(air_95[,2:11], list(air_groups), mean)
aggregate(air_95[,2:12], list(air_groups), median)


####### K Means Clusting #########
air_scaled <- apply(air[, 2:11], 2, function(x) (x-mean(x))/sd(x))
dim(air_scaled)
air_kmeans <- kmeans(air_scaled, centers = 9, iter.max=20)
options(digits = 2)
aggregate(air[,2:11], list(air_kmeans$cluster), mean)
aggregate(air[,2:12], list(air_kmeans$cluster), median)