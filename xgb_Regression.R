#####Boosting

train_label <- train[,"Avg_temp"]
train_matrix <- xgb.DMatrix(data = as.matrix(train[,-5]), label = train_label)

test_label <- test[,"Avg_temp"]
test_matrix <- xgb.DMatrix(data = as.matrix(test[,-5]), label = test_label)

# Parameters

xgb_params <- list("objective" = "reg:linear",
                   "eval_metric" = "rmse"
)
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 1300,
                       watchlist = watchlist,
                       eta = 0.01)

# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_rmse, col = 'blue')
lines(e$iter, e$test_rmse, col = 'red')

min(e$test_rmse)
e[e$test_rmse == 10.54353,] # To view the iteration with minimum rmse eg: 10.54 is the minimum test rmse.

# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)


#Accuracy check

rmse(test$Avg_temp, p) # 10.54353

actuals_preds_xgboost<- data.frame(cbind(actuals=test$Avg_temp, predictedval=p))  # make actuals_predicteds dataframe.
correlation_accuracy_xgboost <- cor(actuals_preds_xgboost)  # 0.9991
head(actuals_preds_xgboost)

postResample(p, test$Avg_temp)
#RMSE   Rsquared        MAE 
#10.5435319  0.9983167  3.3863969 