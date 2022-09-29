library(mlflow)
library(carrier)
library(glmnet)
library(optparse)

# Loading azureml_utils.R. This is needed to use AML as MLflow backend tracking store.
source('azureml_utils.R')

# Setting MLflow related env vars
# https://www.mlflow.org/docs/latest/R-api.html#details
Sys.setenv(MLFLOW_BIN=system("which mlflow", intern=TRUE))
Sys.setenv(MLFLOW_PYTHON_BIN=system("which python", intern=TRUE))
#Sys.setenv(MLFLOW_BIN="/home/krbock/miniconda3/envs/r-mlflow-1.27.0/bin/mlflow")
#Sys.setenv(MLFLOW_PYTHON_BIN="/home/krbock/miniconda3/envs/r-mlflow-1.27.0/bin/python")

#mlflow_set_tracking_uri("http://127.0.0.1:5000")
#mlflow_get_tracking_uri()

options <- list(
  make_option(c("-d", "--data"), default="../data")
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

paste(opt$data_folder)

# Read the wine-quality csv file
data <- read.csv(file.path(opt$data, "wine-quality.csv"))

# Split the data into training and test sets. (0.75, 0.25) split.
sampled <- sample(1:nrow(data), 0.75 * nrow(data))
train <- data[sampled, ]
test <- data[-sampled, ]

# The predicted column is "quality" which is a scalar from [3, 9]
train_x <- as.matrix(train[, !(names(train) == "quality")])
test_x <- as.matrix(test[, !(names(train) == "quality")])
train_y <- train[, "quality"]
test_y <- test[, "quality"]

alpha <- mlflow_param("alpha", 0.8, "numeric")
lambda <- mlflow_param("lambda", 0.5, "numeric")

with(run <- mlflow_start_run(), {
  print("Training the model")  
  model <- glmnet(train_x, train_y,
                  alpha = alpha, lambda = lambda, family = "gaussian")
  predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), model)
  predicted <- predictor(test_x)

  rmse <- sqrt(mean((predicted - test_y) ^ 2))
  mae <- mean(abs(predicted - test_y))
  r2 <- as.numeric(cor(predicted, test_y) ^ 2)

  message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
  message("  RMSE: ", rmse)
  message("  MAE: ", mae)
  message("  R2: ", r2)

  mlflow_log_param("alpha", alpha)
  mlflow_log_param("lambda", lambda)
  mlflow_log_metric("rmse", rmse)
  mlflow_log_metric("r2", r2)
  mlflow_log_metric("mae", mae)

  mlflow_log_model(predictor, "model")
})