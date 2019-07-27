rm(list=ls());

library(randomForest)
library(caret)
library(ROCR)
library(DMwR)
library(data.table)
library(zoo)

df <- fread("creditcard.csv")

#### Exploratory analysis
summary(df)
sum(is.na(df))

set.seed(1003)

ggplot(df, aes(x=V3)) + 
    geom_density(aes(group=Class, colour=Class, fill=Class), alpha=0.3)

#### my check
#exploration
summary(df)
str(df)
table(df$Class)
sum(is.na(df))
boxplot(df$Amount[df$Class == "1"]) #check fraud amount boxplot
# density check
plot(density(df$V1))
boxplot(df$V1~df$Class) # V1跟Y的关系， boxplot
ggplot(data = df, aes(x = V1)) + # V1跟Y的关系， density
    geom_density(aes(group = Class, fill = Class, alpha = .3))
ggplot(data = df, aes(x = Amount)) + # Amount跟Y的关系， density
    geom_density(aes(group = Class, fill = Class, alpha = .3))
# 重点检查amount
boxplot(df$Amount~df$Class) # amount跟Y的关系， boxplot
boxplot(df[df$Class == "1"]$Amount~
            df[df$Class == "1"]$Class) # amount跟Y的关系， boxplot
# fraud amount is usually no more than USD2000

#### Data pre-processing
## 'normalize' the data
transform_columns <- c("V","Amount")
transformed_column<- df[ ,grepl(paste(transform_columns, collapse = "|"),names(df)),with = FALSE]
transformed_column2 <-df[,2:30]
#normalize
transformed_column_processed <- predict(preProcess(transformed_column, method = c("BoxCox","scale")),
                                        transformed_column)

df_new <- data.table(cbind(transformed_column_processed,Class = df$Class))
class(df_new$Class)
df_new[,Class:=as.factor(Class)] #:= data.table method
class(df_new$Class)


#### Training and Test dataset
set.seed(1003)
training_index <- createDataPartition(df_new$Class, p=0.7,list=F)
set.seed(1003)
training_index2 <- sample(nrow(df_new), 0.7*nrow(df_new))

training <- df_new[training_index,]
test<- df_new[-training_index,]

### Logistic regression
logit <- glm(Class ~ ., data = training, family = "binomial")
logit_pred <- predict(logit, test, type = "response")

logit_prediction <- prediction(predictions = logit_pred,
                               labels = test$Class)
#library(ROCR)
#data(ROCR.simple)
#pred <- prediction(ROCR.simple$predictions,ROCR.simple$labels)
logit_recall <- performance(logit_prediction,"prec","rec")
logit_roc <- performance(logit_prediction,"tpr","fpr")
plot(logit_recall)
plot(logit_roc)
logit_auc <- performance(logit_prediction,"auc")

### Random forest
rf.model <- randomForest(Class ~ ., data = training, ntree = 2000, nodesize = 20)
rf_pred <- predict(rf.model, test,type="prob")

rf_prediction <- prediction(rf_pred[,2],test$Class)
rf_recall <- performance(rf_prediction,"prec","rec")
rf_roc <- performance(rf_prediction,"tpr","fpr")
rf_auc <- performance(rf_prediction,"auc")

### Bagging Trees
ctrl <- trainControl(method = "cv", number = 10)

tb_model <- train(Class ~ ., data = train_smote, method = "treebag",
                 trControl = ctrl)

tb_pred <- predict(tb_model$finalModel, test, type = "prob")

tb_prediction <- prediction(tb_pred[,2],test$Class)
tb_recall <- performance(logit_prediction,"prec","rec")
tb_roc <- performance(logit_prediction,"tpr","fpr")
tb_auc <- performance(logit_prediction,"auc")

plot(logit_recall,col='red')
plot(rf_recall, add = TRUE, col = 'blue')
plot(tb_recall, add = TRUE, col = 'green')

#### Functions to calculate 'area under the pr curve'
auprc <- function(pr_curve) {
 x <- as.numeric(unlist(pr_curve@x.values))
 y <- as.numeric(unlist(pr_curve@y.values))
 y[is.nan(y)] <- 1
 id <- order(x)
 result <- sum(diff(x[id])*rollmean(y[id],2))
 return(result)
}

auprc_results <- data.frame(logit=auprc(logit_recall)
                            , rf = auprc(rf_recall)
                            , tb = auprc(tb_recall))
