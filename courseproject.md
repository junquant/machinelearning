# Predicting the Manner which Unilateral Dumbbell Biceps Curls (Bar Bell Lifts) Were Performed

## **Summary** 
The objective of the project is to construct a model to predict the manner which Unilateral Dumbbell Biceps Curls (barbell lifts) were performed. The set of data being used to contruct the model was based on data collected from accelerometers on the belt, forearm, arm and dumbbells of the 6 participants who were asked to perform the barbell lifts. 

Attributes used for book keeping purposes as well as attributes with more than 95% NAs were removed from the training of the model. Next, the Random Forests method was used to used to train the model and a data set was held out using random subsampling to be used to cross validate the final model built. 

As the Random Forests model constructed gave an estimated out of sample accuracy of over 99%, the model was chosen to predict the outcome of the 20 test cases. 

The model was able to accurately predict 20 times out of the 20 test cases. 

## **Loading Required Libraries**

```r
library(caret)
```

## **Data Cleansing and Exploration**

**Read and Explore Data**

We will first read the data from the training set (pml-training.csv) and look at a summary of the data. 

```r
# Read the CSV file in the working directory
data <- read.csv("pml-training.csv", na.strings=c("#DIV/0","#DIV/0!","NA"))
```

```r
summary(data)
```

Refer to Appendix 1 - Summary of Data for the detailed output. 

**Cleanse Data**

We notice that there are attributes used for book keeping purposes such as user_name, raw_timestamp_part_1. We will proceed to remove these attributes from the data set to exclude them from the model. 


```r
# Remove columns timestamps and identifiers (eg. ID) columns
data <- data[,c(-1:-7)]
```

We also notice that some attributes such as kurtosis_roll_belt and min_roll_belt contains a large number of NAs. We will create a function to calculate the percentage of NAs in an attribute and proceed to remove the attributes with more than 95% NAs from the data set. 


```r
# Create a function to check for % NA values in the respective columns
# Returns a data frame of the number of NA rows, number of rows, and % NA
CalcNA <- function(data){
        percentageNA <- apply(data,2,FUN=function(x) sum(is.na(x))/length(x))
        countNA <- apply(data,2,FUN=function(x) sum(is.na(x)))
        countRow <- apply(data,2,FUN=function(x) length(x))
        result <- data.frame(cbind(countNA,cbind(countRow,percentageNA)))
        return(result)
}

result <- CalcNA(data)
includedCol = rownames(result[result$percentageNA<=0.95,])

data <- data[,includedCol]
```

## **Training the Model**
The seed was set so that the results of the model can be reproduced. The training data set was further split into *train* and *traintest* by random subsampling the training data set. *traintest* will later be used to evaluate the final model. The model is trained using the Random Forests method and using all other attributes to predict the attribute classe. 5-fold Cross Validation was also used in the training of the model by specifying it in the trainControl function. 


```r
# Setting the seed for reproducibility
set.seed(123)

# Split the data set into 60 / 40 for the training and test data set
inTrain <- createDataPartition(y=data$classe, p=0.6, list=FALSE)
train <- data[inTrain,]
traintest <- data[-inTrain,]

# Specifing trainControl and training the model
trainCtrl <- trainControl(method="cv", number=5)
set.seed(456)
model <- train(classe~.,method="rf", data=train, trControl=trainCtrl)

# (Optional) - Save the model onto disk and be able to load the model in future without training the model again. 
# saveRDS(model,"model_rf.rds")

# (Optional) - If the model was saved, read the saved model using readRDS. 
# model <- readRDS("model_rf.rds")
```

## **Cross Validation and Out of Sample Error Rate Estimation**
As a test set (random subsampled from the training set, *traintest*) had been held out for the final evaluation, we will be using model to predict the outcome. The result is stored in the *outcome* variable. This ensures that the samples used to evaluate the model were not used for training the model. The Confusion Matrix is shown in the Figure 1 - Confusion Matrix.

```r
# Use the model to predict the classe variable
outcome <- predict(model, traintest)

# Use a Confusion Matrix to assess the performance of the model
cm <- confusionMatrix(outcome,traintest$classe)
```

**Figure 1 - Confusion Matrix**

```r
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228   12    0    0    0
##          B    4 1504    8    0    0
##          C    0    2 1347   17    5
##          D    0    0   13 1268    5
##          E    0    0    0    1 1432
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9915          
##                  95% CI : (0.9892, 0.9934)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9892          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9908   0.9846   0.9860   0.9931
## Specificity            0.9979   0.9981   0.9963   0.9973   0.9998
## Pos Pred Value         0.9946   0.9921   0.9825   0.9860   0.9993
## Neg Pred Value         0.9993   0.9978   0.9968   0.9973   0.9984
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1917   0.1717   0.1616   0.1825
## Detection Prevalence   0.2855   0.1932   0.1747   0.1639   0.1826
## Balanced Accuracy      0.9980   0.9944   0.9905   0.9916   0.9965
```

**The Out of Sample Error Rate estimate is given by:**

```r
errorRateEstimate <- as.vector(1 - confusionMatrix(outcome,traintest$classe)$overall[1])
errorRateEstimate
```

```
## [1] 0.008539383
```

As the model constructed has an out of sample error estimate of less than 1%, fitting other models or tuning the existing model will not result in substantial improvement in accuracy. As such, the model will be used to predict the test cases. 

## **Predicting the Test Cases**
Here, we first read in the test data and predict the outcome using the model constructed above. The predicted outcomes are shown in Figure 2 - Predicted Outcomes of Test Cases. 

```r
# Read the test data for submission
submissionData <- read.csv("pml-testing.csv", na.strings=c("#DIV/0","#DIV/0!","NA"))

# Predict the outcome
finalOutcome <- predict(model, submissionData)
finalOutcome <- as.vector(finalOutcome)
```

**Figure 2 - Predicted Outcomes of Test Cases**

```r
finalOutcome
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

## **Output the Predictions for the 20 Test Cases**

```r
# Write the outcome to files for submission
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("outcome/problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

pml_write_files(finalOutcome)
```

***
## **Appendix 1 - Summary of Data**

**Figure 3 - Summary of Data**


```r
summary(data)
```

```
##    roll_belt        pitch_belt          yaw_belt       total_accel_belt
##  Min.   :-28.90   Min.   :-55.8000   Min.   :-180.00   Min.   : 0.00   
##  1st Qu.:  1.10   1st Qu.:  1.7600   1st Qu.: -88.30   1st Qu.: 3.00   
##  Median :113.00   Median :  5.2800   Median : -13.00   Median :17.00   
##  Mean   : 64.41   Mean   :  0.3053   Mean   : -11.21   Mean   :11.31   
##  3rd Qu.:123.00   3rd Qu.: 14.9000   3rd Qu.:  12.90   3rd Qu.:18.00   
##  Max.   :162.00   Max.   : 60.3000   Max.   : 179.00   Max.   :29.00   
##   gyros_belt_x        gyros_belt_y       gyros_belt_z    
##  Min.   :-1.040000   Min.   :-0.64000   Min.   :-1.4600  
##  1st Qu.:-0.030000   1st Qu.: 0.00000   1st Qu.:-0.2000  
##  Median : 0.030000   Median : 0.02000   Median :-0.1000  
##  Mean   :-0.005592   Mean   : 0.03959   Mean   :-0.1305  
##  3rd Qu.: 0.110000   3rd Qu.: 0.11000   3rd Qu.:-0.0200  
##  Max.   : 2.220000   Max.   : 0.64000   Max.   : 1.6200  
##   accel_belt_x       accel_belt_y     accel_belt_z     magnet_belt_x  
##  Min.   :-120.000   Min.   :-69.00   Min.   :-275.00   Min.   :-52.0  
##  1st Qu.: -21.000   1st Qu.:  3.00   1st Qu.:-162.00   1st Qu.:  9.0  
##  Median : -15.000   Median : 35.00   Median :-152.00   Median : 35.0  
##  Mean   :  -5.595   Mean   : 30.15   Mean   : -72.59   Mean   : 55.6  
##  3rd Qu.:  -5.000   3rd Qu.: 61.00   3rd Qu.:  27.00   3rd Qu.: 59.0  
##  Max.   :  85.000   Max.   :164.00   Max.   : 105.00   Max.   :485.0  
##  magnet_belt_y   magnet_belt_z       roll_arm         pitch_arm      
##  Min.   :354.0   Min.   :-623.0   Min.   :-180.00   Min.   :-88.800  
##  1st Qu.:581.0   1st Qu.:-375.0   1st Qu.: -31.77   1st Qu.:-25.900  
##  Median :601.0   Median :-320.0   Median :   0.00   Median :  0.000  
##  Mean   :593.7   Mean   :-345.5   Mean   :  17.83   Mean   : -4.612  
##  3rd Qu.:610.0   3rd Qu.:-306.0   3rd Qu.:  77.30   3rd Qu.: 11.200  
##  Max.   :673.0   Max.   : 293.0   Max.   : 180.00   Max.   : 88.500  
##     yaw_arm          total_accel_arm  gyros_arm_x        gyros_arm_y     
##  Min.   :-180.0000   Min.   : 1.00   Min.   :-6.37000   Min.   :-3.4400  
##  1st Qu.: -43.1000   1st Qu.:17.00   1st Qu.:-1.33000   1st Qu.:-0.8000  
##  Median :   0.0000   Median :27.00   Median : 0.08000   Median :-0.2400  
##  Mean   :  -0.6188   Mean   :25.51   Mean   : 0.04277   Mean   :-0.2571  
##  3rd Qu.:  45.8750   3rd Qu.:33.00   3rd Qu.: 1.57000   3rd Qu.: 0.1400  
##  Max.   : 180.0000   Max.   :66.00   Max.   : 4.87000   Max.   : 2.8400  
##   gyros_arm_z       accel_arm_x       accel_arm_y      accel_arm_z     
##  Min.   :-2.3300   Min.   :-404.00   Min.   :-318.0   Min.   :-636.00  
##  1st Qu.:-0.0700   1st Qu.:-242.00   1st Qu.: -54.0   1st Qu.:-143.00  
##  Median : 0.2300   Median : -44.00   Median :  14.0   Median : -47.00  
##  Mean   : 0.2695   Mean   : -60.24   Mean   :  32.6   Mean   : -71.25  
##  3rd Qu.: 0.7200   3rd Qu.:  84.00   3rd Qu.: 139.0   3rd Qu.:  23.00  
##  Max.   : 3.0200   Max.   : 437.00   Max.   : 308.0   Max.   : 292.00  
##   magnet_arm_x     magnet_arm_y     magnet_arm_z    roll_dumbbell    
##  Min.   :-584.0   Min.   :-392.0   Min.   :-597.0   Min.   :-153.71  
##  1st Qu.:-300.0   1st Qu.:  -9.0   1st Qu.: 131.2   1st Qu.: -18.49  
##  Median : 289.0   Median : 202.0   Median : 444.0   Median :  48.17  
##  Mean   : 191.7   Mean   : 156.6   Mean   : 306.5   Mean   :  23.84  
##  3rd Qu.: 637.0   3rd Qu.: 323.0   3rd Qu.: 545.0   3rd Qu.:  67.61  
##  Max.   : 782.0   Max.   : 583.0   Max.   : 694.0   Max.   : 153.55  
##  pitch_dumbbell     yaw_dumbbell      total_accel_dumbbell
##  Min.   :-149.59   Min.   :-150.871   Min.   : 0.00       
##  1st Qu.: -40.89   1st Qu.: -77.644   1st Qu.: 4.00       
##  Median : -20.96   Median :  -3.324   Median :10.00       
##  Mean   : -10.78   Mean   :   1.674   Mean   :13.72       
##  3rd Qu.:  17.50   3rd Qu.:  79.643   3rd Qu.:19.00       
##  Max.   : 149.40   Max.   : 154.952   Max.   :58.00       
##  gyros_dumbbell_x    gyros_dumbbell_y   gyros_dumbbell_z 
##  Min.   :-204.0000   Min.   :-2.10000   Min.   : -2.380  
##  1st Qu.:  -0.0300   1st Qu.:-0.14000   1st Qu.: -0.310  
##  Median :   0.1300   Median : 0.03000   Median : -0.130  
##  Mean   :   0.1611   Mean   : 0.04606   Mean   : -0.129  
##  3rd Qu.:   0.3500   3rd Qu.: 0.21000   3rd Qu.:  0.030  
##  Max.   :   2.2200   Max.   :52.00000   Max.   :317.000  
##  accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z  magnet_dumbbell_x
##  Min.   :-419.00   Min.   :-189.00   Min.   :-334.00   Min.   :-643.0   
##  1st Qu.: -50.00   1st Qu.:  -8.00   1st Qu.:-142.00   1st Qu.:-535.0   
##  Median :  -8.00   Median :  41.50   Median :  -1.00   Median :-479.0   
##  Mean   : -28.62   Mean   :  52.63   Mean   : -38.32   Mean   :-328.5   
##  3rd Qu.:  11.00   3rd Qu.: 111.00   3rd Qu.:  38.00   3rd Qu.:-304.0   
##  Max.   : 235.00   Max.   : 315.00   Max.   : 318.00   Max.   : 592.0   
##  magnet_dumbbell_y magnet_dumbbell_z  roll_forearm       pitch_forearm   
##  Min.   :-3600     Min.   :-262.00   Min.   :-180.0000   Min.   :-72.50  
##  1st Qu.:  231     1st Qu.: -45.00   1st Qu.:  -0.7375   1st Qu.:  0.00  
##  Median :  311     Median :  13.00   Median :  21.7000   Median :  9.24  
##  Mean   :  221     Mean   :  46.05   Mean   :  33.8265   Mean   : 10.71  
##  3rd Qu.:  390     3rd Qu.:  95.00   3rd Qu.: 140.0000   3rd Qu.: 28.40  
##  Max.   :  633     Max.   : 452.00   Max.   : 180.0000   Max.   : 89.80  
##   yaw_forearm      total_accel_forearm gyros_forearm_x  
##  Min.   :-180.00   Min.   :  0.00      Min.   :-22.000  
##  1st Qu.: -68.60   1st Qu.: 29.00      1st Qu.: -0.220  
##  Median :   0.00   Median : 36.00      Median :  0.050  
##  Mean   :  19.21   Mean   : 34.72      Mean   :  0.158  
##  3rd Qu.: 110.00   3rd Qu.: 41.00      3rd Qu.:  0.560  
##  Max.   : 180.00   Max.   :108.00      Max.   :  3.970  
##  gyros_forearm_y     gyros_forearm_z    accel_forearm_x   accel_forearm_y 
##  Min.   : -7.02000   Min.   : -8.0900   Min.   :-498.00   Min.   :-632.0  
##  1st Qu.: -1.46000   1st Qu.: -0.1800   1st Qu.:-178.00   1st Qu.:  57.0  
##  Median :  0.03000   Median :  0.0800   Median : -57.00   Median : 201.0  
##  Mean   :  0.07517   Mean   :  0.1512   Mean   : -61.65   Mean   : 163.7  
##  3rd Qu.:  1.62000   3rd Qu.:  0.4900   3rd Qu.:  76.00   3rd Qu.: 312.0  
##  Max.   :311.00000   Max.   :231.0000   Max.   : 477.00   Max.   : 923.0  
##  accel_forearm_z   magnet_forearm_x  magnet_forearm_y magnet_forearm_z
##  Min.   :-446.00   Min.   :-1280.0   Min.   :-896.0   Min.   :-973.0  
##  1st Qu.:-182.00   1st Qu.: -616.0   1st Qu.:   2.0   1st Qu.: 191.0  
##  Median : -39.00   Median : -378.0   Median : 591.0   Median : 511.0  
##  Mean   : -55.29   Mean   : -312.6   Mean   : 380.1   Mean   : 393.6  
##  3rd Qu.:  26.00   3rd Qu.:  -73.0   3rd Qu.: 737.0   3rd Qu.: 653.0  
##  Max.   : 291.00   Max.   :  672.0   Max.   :1480.0   Max.   :1090.0  
##  classe  
##  A:5580  
##  B:3797  
##  C:3422  
##  D:3216  
##  E:3607  
## 
```
