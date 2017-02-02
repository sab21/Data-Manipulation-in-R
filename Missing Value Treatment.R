############     Missing Value Treatment - 19.12.16


data ("BostonHousing", package="mlbench")  # initialize the data  # load the data
original <- BostonHousing  # backup original data
# Introduce missing values
set.seed(100)
BostonHousing[sample(1:nrow(BostonHousing), 40), "rad"] <- NA
BostonHousing[sample(1:nrow(BostonHousing), 40), "ptratio"] <- NA
head(BostonHousing)


library(mice)
md.pattern(BostonHousing)  # pattern or missing values in data.


##            1. Deleting the observations
  #Ensure below before deleting datapoints
    #Have sufficent data points, so the model doesn't lose power.
    #Not to introduce bias
# Example
lm(medv ~ ptratio + rad, data=BostonHousing, na.action=na.omit)  
# though na.omit is default in lm()


######         2. Deleting the variable


#########      3. Imputation with mean / median / mode

library(Hmisc)
impute(BostonHousing$ptratio, mean)  # replace with mean
impute(BostonHousing$ptratio, median)  # median
impute(BostonHousing$ptratio, 20)  # replace specific number
# or if you want to impute manually
BostonHousing$ptratio[is.na(BostonHousing$ptratio)] <- mean(BostonHousing$ptratio, na.rm = T)
# not run



#Lets compute the accuracy.

library(DMwR)
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- rep(mean(BostonHousing$ptratio, na.rm=T), length(actuals))
regr.eval(actuals, predicteds)
#>        mae        mse       rmse       mape 
#> 1.62324034 4.19306071 2.04769644 0.09545664




########           4. Prediction
###########        4.1. kNN Imputation
#DMwR::knnImputation uses k-Nearest Neighbours approach to impute missing values. 
#What kNN imputation does in simpler terms is as follows: For every observation 
#to be imputed, it identifies 'k' closest observations based on the euclidean distance
#and computes the weighted average (weighted based on distance) of these 'k' obs.

#The advantage is that you could impute all the missing values in all variables 
#with one call to the function. It takes the whole data frame as the argument 
#and you don't even have to specify which variabe you want to impute. 
#But be cautious not to include the response variable while imputing, because, 
#when imputing in test/production environment, if your data contains missing values,
#you won't be able to use the unknown response variable at that time.

library(DMwR)
knnOutput <- knnImputation(BostonHousing[, !names(BostonHousing) %in% "medv"])  # perform knn imputation.
anyNA(knnOutput)
#> FALSE

#Lets compute the accuracy.
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- knnOutput[is.na(BostonHousing$ptratio), "ptratio"]
regr.eval(actuals, predicteds)
#>        mae        mse       rmse       mape 
#> 1.00188715 1.97910183 1.40680554 0.05859526 


########    4.2 rpart
#The limitation with DMwR::knnImputation is that it sometimes may not be 
#appropriate to use when the missing value comes from a factor variable. 
#Both rpart and mice has flexibility to handle that scenario. 
#The advantage with rpart is that you just need only one of the 
#variables to be non NA in the predictor fields.

#The idea here is we are going to use rpart to predict the missing values 
#instead of kNN. To handle factor variable, we can set the method=class 
#while calling rpart(). For numerics, we use, method=anova. Here again,
#we need to make sure not to train rpart on response variable (medv).


library(rpart)
class_mod <- rpart(rad ~ . - medv, data=BostonHousing[!is.na(BostonHousing$rad), ], 
                   method="class", na.action=na.omit)  # since rad is a factor
anova_mod <- rpart(ptratio ~ . - medv, data=BostonHousing[!is.na(BostonHousing$ptratio), ],
                   method="anova", na.action=na.omit)  # since ptratio is numeric.
rad_pred <- predict(class_mod, BostonHousing[is.na(BostonHousing$rad), ])
ptratio_pred <- predict(anova_mod, BostonHousing[is.na(BostonHousing$ptratio), ])

#Lets compute the accuracy.
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- ptratio_pred
regr.eval(actuals, predicteds)
#>        mae        mse       rmse       mape 
#> 0.71061673 0.99693845 0.99846805 0.04099908 

#The mean absolute percentage error (mape) has improved additionally by 
#another ~ 30% compared to the knnImputation. Very Good.

#Accuracy for rad
actuals <- original$rad[is.na(BostonHousing$rad)]
predicteds <- as.numeric(colnames(rad_pred)[apply(rad_pred, 1, which.max)])
mean(actuals != predicteds)  # compute misclass error.
#> 0.25 This yields a mis-classification error of 25%. Not bad for a factor variable!


##           4.3 mice

#mice short for Multivariate Imputation by Chained Equations is an R package 
#that provides advanced features for missing value treatment. It uses a slightly 
#uncommon way of implementing the imputation in 2-steps, using mice() to build 
#the model and complete() to generate the completed data. The mice(df) function 
#produces multiple complete copies of df, each with different imputations of the
#missing data. The complete() function returns one or several of these data sets,
#with the default being the first. Lets see how to impute 'rad' and 'ptratio':
library(mice)
miceMod <- mice(BostonHousing[, !names(BostonHousing) %in% "medv"], method="rf")  # perform mice imputation, based on random forests.
miceOutput <- complete(miceMod)  # generate the completed data.
anyNA(miceOutput)
#> FALSE

#Lets compute the accuracy of ptratio.
actuals <- original$ptratio[is.na(BostonHousing$ptratio)]
predicteds <- miceOutput[is.na(BostonHousing$ptratio), "ptratio"]
regr.eval(actuals, predicteds)
#>        mae        mse       rmse       mape 
#> 0.36500000 0.78100000 0.88374204 
#The mean absolute percentage error (mape) has improved additionally 
#by ~ 48% compared to the rpart. Excellent!.

#Lets compute the accuracy of rad
actuals <- original$rad[is.na(BostonHousing$rad)]
predicteds <- miceOutput[is.na(BostonHousing$rad), "rad"]
mean(actuals != predicteds)  # compute misclass error.
#> 0.15
#The mis-classification error reduced to 15%, which is 6 out of 40 observations. 
#This is a good improvement compared to rpart's 25%.