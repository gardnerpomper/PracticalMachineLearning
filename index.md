# Prediction of exercise type from instrumentation readings

writeup for Coursera's Practical Machine Learning Final Project

# Data

Data was provided for the exercise from the following url: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

It consists of timestamped measurements from varying instruments,
along with a user name and the type of exercise they were doing at the
time. This will provide both the training and test data for this
analysis.

## Loading

The data is loaded from the file, which was copied to the current directory:


```r
library(caret)
library(doMC)

registerDoMC(3) # allow 2 cores for processing
set.seed(12345)
raw.df <- read.csv("pml-training.csv",na.strings=c("NA",""))
```

## Preprocessing

There are two types of events in the file. One is the raw collection
of instrumentation readings. The other is the statistical information
for a 1 second time period of the raw data. This statistical
information is marked with a field named "new_window" and a value of
"yes". The analysis is done on only the raw data, since the
statistical information (mean, max, etc) is of a different type that
the bulk of the information. This code restricts to just the raw
information:


```r
whole.window <- raw.df$new_window != "yes"
window.df <- raw.df[whole.window,c(160,8:159)]
##
## make sure all the values are numeric, not factors
##
window.df[,-c(1)] <- sapply(window.df[,-c(1)],function(x){ as.numeric(as.character(x))})
##
## figure out which columns have values in them and eliminate the rest
##
nsv <- nearZeroVar(window.df,saveMetrics=TRUE)
window.df <- window.df[,nsv$nzv == FALSE]
```

## Splitting into training and testing datasets

In order to do cross validation, the data is split into 2 sets; one
for training the model and one for testing the accuracy. Because there
are many observations and the model chosen (random forest) is
computationally expensive, the training set is restricted to 3000
observations.


```r
inTrain <- sample(nrow(window.df),3000)

training.raw <- window.df[inTrain,]
testing.raw <- window.df[-inTrain,]
```

## Imputing missing values

The observations have occasional missing values, which messed up the learning algorithsm. The missing values are imputed using the knn preprocess method:


```r
predictors <- subset(training.raw,select=-c(classe))
preObj <- preProcess(predictors,method="knnImpute")
training <- predict(preObj,predictors)
training$classe <- training.raw$classe

testing <- predict(preObj,subset(testing.raw,select=-c(classe)))
testing$classe <- testing.raw$classe
```

# Modelling

Several modelling approaches were taken. For each approach, the model
is trained with the training data set and tested against the test data
set. The matrix of predicted versus actual values, along with the
confusion matrix is printed. The accuracy and out of sample error
stats are saved and reported at the end, by model.

## Tree classifier


```r
modFit <- train(classe ~ ., method="rpart",data=training)
pred <- predict(modFit,testing)
tree.cm <- confusionMatrix(testing$classe,pred)
print( tree.cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2839  458  808  475   60
##          B  517 1802  648  178    0
##          C   76  520 2227   17    0
##          D  134  866  722  878    0
##          E   41  925  481  127 1417
## 
## Overall Statistics
##                                         
##                Accuracy : 0.565         
##                  95% CI : (0.557, 0.573)
##     No Information Rate : 0.301         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.453         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.787    0.394    0.456   0.5242   0.9594
## Specificity             0.857    0.885    0.946   0.8816   0.8932
## Pos Pred Value          0.612    0.573    0.784   0.3377   0.4738
## Neg Pred Value          0.934    0.788    0.801   0.9415   0.9955
## Prevalence              0.222    0.282    0.301   0.1033   0.0911
## Detection Rate          0.175    0.111    0.137   0.0541   0.0874
## Detection Prevalence    0.286    0.194    0.175   0.1603   0.1844
## Balanced Accuracy       0.822    0.639    0.701   0.7029   0.9263
```

## Linear Discriminate Analysis


```r
modFit <- train(classe ~ ., data=training,method="lda")
pred <- predict(modFit,testing)
lda.cm <- confusionMatrix(testing$classe,pred)
print( lda.cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3798  107  327  382   26
##          B  495 1876  397  171  206
##          C  346  231 1790  407   66
##          D  167   81  254 1984  114
##          E  115  481  291  299 1805
## 
## Overall Statistics
##                                         
##                Accuracy : 0.694         
##                  95% CI : (0.687, 0.701)
##     No Information Rate : 0.303         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.612         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.772    0.676    0.585    0.612    0.814
## Specificity             0.925    0.906    0.920    0.953    0.915
## Pos Pred Value          0.819    0.597    0.630    0.763    0.603
## Neg Pred Value          0.903    0.931    0.905    0.908    0.969
## Prevalence              0.303    0.171    0.189    0.200    0.137
## Detection Rate          0.234    0.116    0.110    0.122    0.111
## Detection Prevalence    0.286    0.194    0.175    0.160    0.184
## Balanced Accuracy       0.849    0.791    0.753    0.782    0.865
```

## Boosting


```r
modFit <- train(classe ~ ., method="gbm",data=training,verbose=FALSE)
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loaded gbm 2.1
## Loading required package: plyr
```

```r
pred <- predict(modFit,testing)
boost.cm <- confusionMatrix(testing$classe,pred)
print( boost.cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4480   88   23   32   17
##          B  175 2845  100    6   19
##          C    7  130 2635   64    4
##          D    2   13   89 2467   29
##          E    6   78   47   48 2812
## 
## Overall Statistics
##                                         
##                Accuracy : 0.94          
##                  95% CI : (0.936, 0.943)
##     No Information Rate : 0.288         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.924         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.959    0.902    0.911    0.943    0.976
## Specificity             0.986    0.977    0.985    0.990    0.987
## Pos Pred Value          0.966    0.905    0.928    0.949    0.940
## Neg Pred Value          0.984    0.976    0.981    0.989    0.995
## Prevalence              0.288    0.194    0.178    0.161    0.178
## Detection Rate          0.276    0.175    0.162    0.152    0.173
## Detection Prevalence    0.286    0.194    0.175    0.160    0.184
## Balanced Accuracy       0.973    0.940    0.948    0.966    0.981
```

## Random Forest

ran out of time for this run to make submission deadline
##```{r randomForest, cache=TRUE}
##modFit <- train(classe ~ ., data=training, method="rf",prox=TRUE)
##tree2 <- getTree(modFit$finalModel,k=2)
##pred <- predict(modFit,testing)
##rf.cm <- confusionMatrix(testing$classe,pred)
##print( rf.cm)
##```

# Results

Here are the prediction results, by model, with both the accuracy and out of sample error:


```r
model <- c("tree","lda","boost")
acc <- c(tree.cm$overall["Accuracy"],
         lda.cm$overall["Accuracy"],
         boost.cm$overall["Accuracy"])

oos <- c(1-tree.cm$overall["Accuracy"],
         1-lda.cm$overall["Accuracy"],
         1-boost.cm$overall["Accuracy"])

results <- data.frame( model,acc,oos)
names(results) <- c("Model","Accuracy","Out of sample error")
print(results)
```

```
##   Model Accuracy Out of sample error
## 1  tree   0.5651             0.43494
## 2   lda   0.6939             0.30606
## 3 boost   0.9398             0.06025
```

