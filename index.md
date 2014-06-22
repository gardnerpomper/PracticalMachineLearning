# Prediction of exercise type from instrumentation readings

writeup for Coursera's Practical Machine Learning Final Project

# Data

Data was provided for the exercise from the following url: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

It consists of timestamped measurements from varying instruments, along with a user name and the type of exercise they were doing at the time. This will provide both the training and test data for this analysis.

## Loading

The data is loaded from the file, which was copied to the current directory:


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoMC(3) # allow 2 cores for processing
set.seed(12345)
raw.df <- read.csv("pml-training.csv",na.strings=c("NA",""))
```

## Preprocessing

There are two types of events in the file. One is the raw collection of instrumentation readings. The other is the statistical information for a 1 second time period of the raw data. This statistical information is marked with a field named "new_window" and a value of "yes". The analysis is done on only the raw data, since the statistical information (mean, max, etc) is of a different type that the bulk of the information. This code restricts to just the raw information:

```r
whole.window <- raw.df$new_window != "yes"
window.df <- raw.df[whole.window,c(160,8:159)]
window.df[,-c(1)] <- sapply(window.df[,-c(1)],function(x){ as.numeric(as.character(x))})
nsv <- nearZeroVar(window.df,saveMetrics=TRUE)
window.df <- window.df[,nsv$nzv == FALSE]
```
