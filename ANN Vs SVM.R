#=======================================================================================================================
#=======================================================================================================================
#  
#                                       Title: ANN Vs SVM - Better?
#                                        Author: Shantesh Mani
#                                        
#                                        
#                                        Created: 11 July 2018
#                                        Last Edit: 20 June 2018
#                             
#                                          Version: 0.01.06
#                             
#                                          Copyright: 2018
# Notes:

# Script name - ANN Vs SVM SKM 13448444.R

# ANN Vs SVM - Is one technique better than the other?.
#---------------------------------------------------------------------------------------------------------------------- 

# WARNING: Script lines 29-33 will install required packages onto pc if not installed already.  

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------  

rm(list=ls(all=TRUE))

# Ensuring the correct packages are installed before executing the script

if("neuralnet" %in% rownames(installed.packages())==FALSE) {install.packages("neuralnet")}
if("caret" %in% rownames(installed.packages())==FALSE) {install.packages("caret")}
if("rvest" %in% rownames(installed.packages())==FALSE) {install.packages("rvest")}
if("leaps" %in% rownames(installed.packages())==FALSE) {install.packages("leaps")}
if("car" %in% rownames(installed.packages())==FALSE) {install.packages("car")}
if("e1071" %in% rownames(installed.packages())==FALSE) {install.packages("e1071")}


library(caret)
library(e1071)
library(car)
library(leaps)
library(neuralnet)
library(rvest)






shopdata <- read.csv("shopping-mall-catchment.csv", header = TRUE)
str(shopdata)

shopdata$catchment_area <- as.factor(shopdata$catchment_area)   #Converting catchment area into a factor


## Creating a new dataframe which totals the lettable area in each catchment area as well as provides the income, population and unemployment rate ##

catchdata <- matrix(nrow = length(unique(shopdata$catchment_area)), ncol = 8)
colnames(catchdata) <- c("postcode", "total_let_area", "income", "pop", "prop_unemp", "MedianAge", "Prop_BDegree", "Prop_vehicle" )
catchdata <- as.data.frame(catchdata)
catchdata$postcode <- as.vector(unique(shopdata$poa))
catchdata$total_let_area <- aggregate(shopdata$lettable, by = list(catchment_area = shopdata$catchment_area), FUN = sum) [,2]
catchdata$income <- unique(shopdata$income)
catchdata$pop <- aggregate(shopdata$population, by = list(catchment_area = shopdata$catchment_area), FUN = sum) [,2]
catchdata$prop_unemp <- aggregate(shopdata$pr_unemployment, by = list(catchment_area = shopdata$catchment_area), FUN = sum) [,2]

############################################################################################################################################################
# Scraping script to extract data directly from census website
# This Scraper was created in collaboration with Seamus O'leary
# Library(rvest)

Prop_BDegree <- c()
for (i in catchdata$postcode){
  url <- read_html(paste("http://quickstats.censusdata.abs.gov.au/census_services/getproduct/census/2016/quickstat/POA", i,  "?opendocument", sep = ""))
  rbind(url)
  BDegree <- url %>% html_nodes("#peopleContent table:nth-child(20) tr:nth-child(2) td:nth-child(3)") %>% html_text()
  Prop_BDegree <- rbind(Prop_BDegree, BDegree)
}

rownames(Prop_BDegree) <- NULL


MedianAge <- c()
for (i in catchdata$postcode){
  url <- read_html(paste("http://quickstats.censusdata.abs.gov.au/census_services/getproduct/census/2016/quickstat/POA", i,  "?opendocument", sep = ""))
  rbind(url)
  mage <- url %>% html_nodes(".qsPeople tr:nth-child(4) .summaryData") %>% html_text()
  MedianAge <- rbind(MedianAge, mage)
}

rownames(MedianAge) <- NULL

Prop_vehicle <- c()
for (i in catchdata$postcode){
  url <- read_html(paste("http://quickstats.censusdata.abs.gov.au/census_services/getproduct/census/2016/quickstat/POA", i,  "?opendocument", sep = ""))
  rbind(url)
  PV <- url %>% html_nodes("table:nth-child(39) tr:nth-child(3) td:nth-child(3)") %>% html_text()
  Prop_vehicle <- rbind(Prop_vehicle, PV)
}

rownames(Prop_vehicle) <- NULL

#Changing variables to integer and numeric
catchdata$MedianAge <- as.integer(MedianAge)
catchdata$Prop_BDegree <- as.numeric(Prop_BDegree)/100
catchdata$Prop_vehicle <- as.numeric(Prop_vehicle)/100


# Duplicate dataframe  for  ANN model without post code variable
anndata <- catchdata[,-1]


# create the function for data Standardisation
datanorm <- function(x) {(x-min(x))/(max(x)-min(x))}
# create function for data Standardisation
denorm <- function(x){(x)*(max(catchdata$total_let_area)-min(catchdata$total_let_area))+min(catchdata$total_let_area)}

# normalise the data
data <- lapply(anndata, datanorm)
data <- as.data.frame(data)

# spilt the dataset into train, validate, test data
set.seed(1)
size1 <- floor(0.5*nrow(data))    
train_ind <- sample(seq(nrow(data)),size=size1)

trainset <- data[train_ind,]
restofdata <- data[-train_ind,]


size2 <- floor(0.5*nrow(restofdata))
test_ind <- sample(seq(nrow(restofdata)),size=size2)

testset <- restofdata[test_ind,]
valiset <- restofdata[-test_ind,]



################################################################################################################
################################################################################################################
################################################################################################################
# Create ANN Model


numberofnodes <- c(seq(from=2,to=60,by=5))
MSE=vector(length = 5)
i=0
for (nh in numberofnodes) {
  set.seed(7)
  nnmodel<- neuralnet(total_let_area ~ income + pop + prop_unemp + MedianAge + Prop_BDegree + Prop_vehicle, data=trainset, hidden = nh, threshold = 0.01, act.fct = 'tanh', stepmax = 1e+09)
  net.results<-compute(nnmodel,valiset[,2:7])
  cleanoutput<-cbind(valiset$total_let_area,as.data.frame(net.results$net.result))
  colnames(cleanoutput)<- c("Actual Output", "Neural Net Output")
  
  i=i+1
  MSE[i]<- mean((cleanoutput$"Neural Net Output"-cleanoutput$"Actual Output")^2)
}
plot(numberofnodes,MSE, xlab = "Number of Nodes", ylab = "MSE", type = "b" )

#plot with optimal number of nodes

nnmodel<- neuralnet(total_let_area ~ income + pop + prop_unemp + MedianAge + Prop_BDegree + Prop_vehicle, data=trainset, hidden = 7, threshold = 0.01, act.fct = 'tanh', stepmax = 1e+09)

net.results<-compute(nnmodel,testset[,2:7])
cleanoutput<-cbind(testset$total_let_area,as.data.frame(net.results$net.result))
colnames(cleanoutput)<- c("Actual Output", "Neural Net Output")
RMSE <- sqrt(mean(denorm((cleanoutput$"Neural Net Output"-cleanoutput$"Actual Output"))^2))
print(RMSE)


#################################################################################################################
#################################################################################################################
#################################################################################################################
###################################################SVM Model#####################################################



#Checking for correlation in the dataset
cor(catchdata)
ifelse(abs(cor(catchdata))>0.4,cor(catchdata),0)
str(catchdata)

#Variable Selection using stepwise
lmp <- lm(total_let_area ~ ., data = catchdata) 
lm0 <- lm(total_let_area ~ 1, data = catchdata) 

fwd <- step(lm0, scope = list(lower = lm0, upper = lmp), direction = "forward", k=2, trace=0) 
bwd <- step(lmp, scope = list(lower = lm0, upper = lmp), direction = "backward", k=2, trace = 0) 
hyb <- step(lmp, scope = list(lower = lm0, upper = lmp), direction = "both", k = 2, trace = 0)

summary(fwd)
summary(bwd)
summary(hyb)

# Stepwise variable selection using forward, backward and bybrid provide same results. 
# lm(formula = total_let_area ~ prop_unemp + pop + Prop_BDegree,data = catchdata)

#Create new dataset from original dataset with only selected variables for SVM model

modeldata <- catchdata[,-c(1,3,6,8)]

#################################################################################################################
#Build and Tune SVM Model

#################################################################################################################
#################################################################################################################
#################################################################################################################
# Split data into train and test sets
set.seed(1)
size1 <- floor(0.8*nrow(modeldata))    
train_ind2 <- sample(seq(nrow(modeldata)),size=size1)

trainset2 <- modeldata[train_ind2,]
testset2 <- modeldata[-train_ind2,]

#SVM model - untuned 
svmmodel <- svm(total_let_area ~ prop_unemp + pop + Prop_BDegree, data=trainset2, scale = FALSE)
summary(svmmodel)
predy <-predict(svmmodel, trainset2)
error <-trainset$total_let_area-predy
svmpredrmse <- sqrt(mean((error)^2))

x <- subset(trainset2, select = -total_let_area)

y <- trainset2$total_let_area

tunemodel <- tune(svm, train.x = x, train.y = y, kernel = 'radial',  ranges = list(cost=10^(-1:2), gamma = c(0.5,1,2))) 
print(tunemodel)
plot(tunemodel)

#Using the best model
tunedSVM <- tunemodel$best.model
print(tunedSVM)
tunedSVMY <- predict(tunedSVM, x)
errortune <- y-tunedSVMY
tunedsvmrmse <-sqrt(mean((errortune)^2))
#using test data
x1 <-subset(testset2,select= -total_let_area)
testtunedsvmy <-predict(tunedSVM,x1)
testerrortune <-testset2$total_let_area-testtunedsvmy
testtunedsvmrmse <-sqrt(mean((testerrortune)^2))
testtunedsvmrmse


#################################################PLOT#####################################
xx <- testset$total_let_area
yy <- testset$total_let_area

lm23 <- lm(yy~xx)

plot(testset2$total_let_area,testtunedsvmy, xlab = "Total Let Area", ylab = "Potential Growth"  )
abline(lm23, col = "green", lwd = 2)
############################################################################################

errormode <- as.data.frame(nnmodel$net.result)
colnames(errormode) <- "ANN Results"
errormode <- denorm(errormode) 
error <- errormode - testset$total_let_area
xlabz <- 1:length(error[,1])
plot(xlabz, error$`ANN Results`, xlab = "Observations", ylab = "ERROR")

##############################################################################################


#Funciton to calcualte confidence intervals to check for statistical significance
lower.ci <- function(RMSE, n, alpha) {sqrt(n/qchisq(1-alpha/2, n))*RMSE}
upper.ci <- function(RMSE, n, alpha) {sqrt(n/qchisq(alpha/2, n))*RMSE}


#Calculate CI's for ANN
lowerRMSEANN <- lower.ci(RMSE,92 , 0.05)
upperRMSEANN <- upper.ci(RMSE, 92, 0.05)

#Results
CIANN <- cbind(lowerRMSEANN, RMSE, upperRMSEANN)
colnames(CIANN) <- c("Lower CI", "ANN RMSE", "Upper CI")
CIANN


#Calculate CI's for SVM
lowerRMSESVM <- lower.ci(testtunedsvmrmse, 74 , 0.05)
upperRMSESVM <- upper.ci(testtunedsvmrmse, 74, 0.05)

#Results
CISVM <- cbind(lowerRMSESVM, testtunedsvmrmse, upperRMSESVM)
colnames(CISVM) <- c("Lower CI", "SVM RMSE", "Upper CI")
CISVM

########################################################END###########################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
############################################################################################################################################################################################################################################
######################################################################################################################

















