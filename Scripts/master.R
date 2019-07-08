###############################################################################
#Libraries/Cores:--------------

pacman::p_load(doParallel, readr, rstudioapi, dplyr, plotly, corrplot, caret,
               rpart, rpart.plot, C50, randomForest, MASS, e1071, kknn, tidyverse, dummies, zoo, kernlab)
detectCores()

###############################################################################
# Github setup --------------

current_path <- getActiveDocumentContext()$path

setwd(dirname(dirname(current_path)))
rm(current_path)
###############################################################################
#Cluster --------------

cl <- makeCluster(6) #creating a cluster
registerDoParallel(cl)
getDoParWorkers()

###############################################################################
# Importing Data:--------

training_features <- read_csv("Datasets/Training_data_features.csv")
str(training_features)

training_labels <- read_csv("Datasets/Training_data_labels.csv")
str(training_labels)
training_labels[,c("city", "year", "weekofyear")] <- NULL


train_set <- cbind(training_features, training_labels)

test_features <- read_csv("Datasets/Test_set_features.csv")


###############################################################################
#Preprocessing:------------------
########
# Missing values: (train set)
sum(is.na(train_set))
which(is.na(train_set))
na <- colnames(train_set)[colSums(is.na(train_set)) > 0] #columns that have missing values

for (i in na){
  train_set[,i] <- as.numeric(train_set[,i])
}


for(i in na){
  train_set[is.na(train_set[,i]), i] <- mean(train_set[,i], na.rm = TRUE) #replacing NA with mean of the column
}

########
# Missing values: (test set)
test_features$ndvi_ne<-na.locf(test_features$ndvi_ne)
test_features$ndvi_nw<-na.locf(test_features$ndvi_nw)
test_features$ndvi_se<-na.locf(test_features$ndvi_se)
test_features$ndvi_sw<-na.locf(test_features$ndvi_sw)
test_features$precipitation_amt_mm<-na.locf(test_features$precipitation_amt_mm)
test_features$reanalysis_air_temp_k<-na.locf(test_features$reanalysis_air_temp_k)
test_features$reanalysis_avg_temp_k<-na.locf(test_features$reanalysis_avg_temp_k)
test_features$reanalysis_dew_point_temp_k<-na.locf(test_features$reanalysis_dew_point_temp_k)
test_features$reanalysis_max_air_temp_k<-na.locf(test_features$reanalysis_max_air_temp_k)
test_features$reanalysis_min_air_temp_k<-na.locf(test_features$reanalysis_min_air_temp_k)
test_features$reanalysis_precip_amt_kg_per_m2<-na.locf(test_features$reanalysis_precip_amt_kg_per_m2)
test_features$reanalysis_relative_humidity_percent<-na.locf(test_features$reanalysis_relative_humidity_percent)
test_features$reanalysis_sat_precip_amt_mm<-na.locf(test_features$reanalysis_sat_precip_amt_mm)
test_features$reanalysis_specific_humidity_g_per_kg<-na.locf(test_features$reanalysis_specific_humidity_g_per_kg)
test_features$reanalysis_tdtr_k<-na.locf(test_features$reanalysis_tdtr_k)
test_features$station_avg_temp_c<-na.locf(test_features$station_avg_temp_c)
test_features$station_diur_temp_rng_c<-na.locf(test_features$station_diur_temp_rng_c)
test_features$station_max_temp_c<-na.locf(test_features$station_max_temp_c)
test_features$station_min_temp_c<-na.locf(test_features$station_min_temp_c)
test_features$station_precip_mm<-na.locf(test_features$station_precip_mm)

########
#Multicollinearity: (deleting multicollinear variables)
train_set$reanalysis_avg_temp_k <- NULL
train_set$reanalysis_sat_precip_amt_mm <- NULL
train_set$reanalysis_specific_humidity_g_per_kg <- NULL
train_set$reanalysis_max_air_temp_k <- NULL

test_features$reanalysis_avg_temp_k <- NULL
test_features$reanalysis_sat_precip_amt_mm <- NULL
test_features$reanalysis_specific_humidity_g_per_kg <- NULL
test_features$reanalysis_max_air_temp_k <- NULL


########

# Deleting unneeded columns in train set:
train_set$city <- as.factor(train_set$city)
train_set$year <- NULL
train_set$weekofyear <- as.factor(train_set$weekofyear)
train_set$week_start_date <- NULL

# Deleting unneeded columns in test set:
test_features$city <- as.factor(test_features$city)
test_features$year <- NULL
test_features$weekofyear <- as.factor(test_features$weekofyear)
test_features$week_start_date <- NULL

########

# Creating dummy variables:
train <- dummy.data.frame(train_set)
test <- dummy.data.frame(test_features)


###############################################################################
# Train/Test sets:

random<-sample(1:dim(train)[1])
pct2o3<-floor(dim(train)[1]*2/3)
train_df<-train[random[1:pct2o3],]
validation_df<-train[random[(pct2o3+1):dim(train)[1]],]

###############################################################################

###############################################################################
#Modeling:

# Random Forest: 
set.seed(432)
rf_mod1 <- randomForest(total_cases ~ ., data = train_df, importance = TRUE,
                       ntree = 500, cross = 3, method = "rf")

predictions_rf1 <- predict(rf_mod1, validation_df)
MAE_rf1 <- mean(abs(predictions_rf1 - validation_df$total_cases))
MAE_rf1

# SVM:
set.seed(321)
SVM1 <- ksvm(total_cases ~., data = train_df, kernel = "rbfdot", kpar = "automatic",
             C = 0.3, cross = 3, prob.model = TRUE)

predictions_svm1 <-predict(SVM1, validation_df, type="votes")
MAE_svm1<-mean(abs(predictions_svm1 - validation_df$total_cases))
MAE_svm1

###############################################################################
# Applying model:

test_features$total_cases <- NA 

test_rf <- test_features
test_rf_1 <- dummy.data.frame(test_rf)

final_predictions_rf <- predict(rf_mod1, test_rf_1)
test_rf$total_cases <- final_predictions_rf
write_csv(test_rf, "random_forest_predicted.csv")

test_svm <- test_features

final_predictions_svm <- predict(SVM1, dummy.data.frame(test_svm_1))
test_svm$total_cases <- final_predictions_svm
write_csv(test_svm, "SVM_predicted.csv")

###############################################################################
#Submission:
test_features <- read_csv("Datasets/Test_set_features.csv")
test_rf$year <- test_features$year

rf_submission <- dplyr::select(.data = test_rf, city, year, weekofyear, total_cases )
write_csv(rf_submission, "rf_submission.csv")

test_svm$year <- test_features$year
svm_submission <- dplyr::select(.data = test_svm, city, year, weekofyear, total_cases )
write_csv(svm_submission, "svm_submission.csv")

###############################################################################
stopCluster(cl) #stopping the cluster
