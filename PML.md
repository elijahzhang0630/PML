---
title: "PML"
author: "Shengbai Zhang"
date: "8/9/2021"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Synopsis

"Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways." - Coursera

The aim of this project is to predict the manner in which they did the exercise ("classe" variable in the training set). This report describes the building and testing of the different models. The best model will then be used to predict 20 different test cases. 

**Data used in this project were provided by Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.**

# R libraries
```{r}
library("caret")
library("rattle")
library("randomForest")
library("corrplot")
library("gbm")
```

# Preparation of the data

## Loading
```{r}
#Training set
trainData <- read.csv("C:/Users/celin/Desktop/MachineLearning/pml-training.csv", header=T)
#Testing set
testData <- read.csv("C:/Users/celin/Desktop/MachineLearning/pml-testing.csv", header=T)
#Dimensions of training set
dim(trainData)
```
```{r}
#Dimensions of testing set
dim(testData)
```
The training set is composed of 19622 observations and 160 variables whereas the testing set contains 20 observations.

## Cleaning

We remove variables that contains missing values (NA or ""):
```{r}
#Training set
trainData <- trainData[,colSums(is.na(trainData)) == 0]
trainData <- trainData[,colSums(trainData == "") == 0]
#Testing set
testData <- testData[,colSums(is.na(testData)) == 0]
testData <- testData[,colSums(testData == "") == 0]
#New dimensions of training set
dim(trainData)
```
```{r}
#New dimensions of testing set
dim(testData)
```
We also remove the first column (X = ID) as this will not be informative
```{r}
trainData$X <- NULL
testData$X <- NULL
```

## Preparation
We split the training data such as 70% will be used as training set and 30% will be used as test set: it will be used to compute the out-of-samples errors.
The loaded testing set (pml-testing.csv) is renamed as validTestData and will be used later on for the 20 cases.
```{r}
#Rename the loaded testing set testData -> validTestData & remove the testData 
validTestData <- testData
rm(testData)
set.seed(42) 
partitionTrainData <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
trainData <- trainData[partitionTrainData, ]
testData <- trainData[-partitionTrainData, ]
#Dimension of the training set (obtained after partitioning)
dim(trainData)
```
```{r}
#Dimensions of the testing set (obtained after partitioning)
dim(testData)
```
Remove the variables having a near zero variance before performing the training & testing steps
```{r}
nzv <- nearZeroVar(trainData)
trainData <- trainData[, -nzv]
testData  <- testData[, -nzv]
#Dimensions of the training set ready
dim(trainData)
```
```{r}
#Dimensions of the testing set ready
dim(testData)
```
After all these QC steps, we now have 58 variables remaining. The correlation between numerical variables can be observed in the following corrplot:
```{r}
#Remove non numerical variables
corMtx <- cor(trainData[,-c(1,2,3,4,5,58)])
#Plot
corrplot(corMtx, method = "color", type = "upper", order = "hclust", tl.cex = 0.5,tl.col="black")
```
We can observe that several variables are highly correlated (dark red = strong positive correlation, dark blue = strong negative correlation).

# Models

We will build the model for predictions using three different algorithms studied during the course: (1) decision tree, (2) random forest and (3) generalized boosted model.

## Decision tree

### Building
```{r}
set.seed(1234)
#3-fold cross validation
ctrlTreeModel <- trainControl(method="cv", number=3)
decisionTreeModel <- train(classe ~ ., data=trainData, method="rpart", trControl=ctrlTreeModel)
#Visualisation of the constructed decision tree
fancyRpartPlot(decisionTreeModel$finalModel)
```

### Predictions
```{r}
predTreeModel <- predict(decisionTreeModel, testData)
confMatTree <- confusionMatrix(predTreeModel, as.factor(testData$classe))
print(confMatTree)
```
The **accuracy of the model** is poor and equal to **0.5643** and hence a **out-of-sample error** of 1-0.5643 = **0.4357**.

## Random forest

### Building
```{r}
set.seed(1234)
#3-fold cross validation
rfCtrl <- trainControl(method="cv", number=3)
rfModel <- train(classe ~ ., data=trainData, method="rf", trControl=rfCtrl)
```

### Predictions
```{r}
predRf <- predict(rfModel,testData)
confMatRf <- confusionMatrix(predRf, as.factor(testData$classe))
print(confMatRf)
```
The **accuracy of the RF model is 1 and hence the out-of-samples errors is 0**. However, this value is most likely due to an over-fitting of the model.
```{r}
#Plot of the model performances
plot(confMatRf$table, col = confMatRf$byClass, main = "Random Forest Confusion Matrix")
```


## Generalized boosted model

### Building
```{r}
set.seed(1432)
#3-fold cross validation
gbmCtrl <- trainControl(method = "cv", number = 3)
gbmModel  <- train(classe ~ ., data=trainData, method = "gbm", trControl = gbmCtrl, verbose = FALSE)
print(gbmModel)
```

### Predictions
```{r}
predGbm <- predict(gbmModel,testData)
confMatGbm <- confusionMatrix(predGbm, as.factor(testData$classe))
print(confMatGbm)
```
The GBM model has an **accuracy of 0.9985 and hence an out-of-samples errors of 0.0025**.

## Selection of the best model for the assignement

The random forest showed an accuracy of 1 that might be due to over-fitting but it is followed closely by GBM with an accuracy of almost 1. We will then **select the random forest model** as the best one to use for the assignment on the 20 cases.

We finally apply the RF model on the original testing set (validTestData) to get the predictions:
```{r}
finalPredictions <- predict(rfModel,validTestData)
print(finalPredictions)
```