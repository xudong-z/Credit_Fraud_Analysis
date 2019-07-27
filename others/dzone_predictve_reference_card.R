# https://dzone.com/refcardz/machine-learning-predictive

summary(iris)
head(iris)

# pre: Prepare training and testing data 
testidx <- which(1:length(iris[,1])%%5 == 0) # 如何做到每五个数取一个？
# 1:150 %% 5 ==0
iristrain <- iris[-testidx,] 
iristest <- iris[testidx,]

# Modeling
library(car) 
summary(Prestige)
head(Prestige)
sum(is.na(Prestige))

testidx <- which(1:nrow(Prestige)%%4==0) 
testidx2 <- sample(1:nrow(Prestige), nrow(Prestige)/4)
prestige_train <- Prestige[-testidx,] 
prestige_test <- Prestige[testidx,]

# LINEAR REGRESSION
model <- lm(prestige~., data=prestige_train) 
# Use the model to predict the output of test data 
prediction <- predict(model, newdata=prestige_test) 
# Check for the correlation with actual result 
cor(prediction, prestige_test$prestige) 
summary(model)
# The goal of minimizing the square error makes linear regression
# very sensitive to outliers that greatly deviate in the output.  
# It is a common practice to identify those outliers, remove them, 
# and then rerun the training.

# LOGISTIC REGRESSION
newcol = data.frame(isSetosa=(iristrain$Species == 'setosa')) 
traindata <- cbind(iristrain, newcol) 
head(traindata)
# or 
# traindata$new_col = ifelse(iristrain$Species == 'setosa', 'yes','no')

formula <- isSetosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
logisticModel <- glm(formula, data=traindata, family='binomial')
prob <- predict(logisticModel, newdata=iristest, type='response')
round(prob, 3)

# REGRESSION WITH REGULARIZATION
library(glmnet) 
cv.ft <- cv.glmnet(as.matrix(prestige_train[,c(-4, -6)]),  # as.matrix
                   as.vector(prestige_train[,4]), 
                   nlambda=100, alpha=0.7, family='gaussian') # a=0.7  接近L1 lasso
# This is the cross-validation plot. 
# It shows the best lambda with minimal-root,mean-square error.
plot(cv.ft) 
coef(cv.ft)

prediction <- predict(cv.ft, newx=as.matrix(prestige_test[,c(-4, -6)])) 
cor(prediction, as.vector(prestige_test[,4]))


# NEURAL NETWORK  (NON-LINEAR) 适合多个Y的预测
library(neuralnet) 
nnet_iristrain <- iristrain 
#Binarize the categorical output   dummy每个output class
nnet_iristrain <- cbind(nnet_iristrain, iristrain$Species == 'setosa')
nnet_iristrain <- cbind(nnet_iristrain, iristrain$Species == 'versicolor') 
nnet_iristrain <- cbind(nnet_iristrain, iristrain$Species == 'virginica') 
names(nnet_iristrain)[6] <- 'setosa'
names(nnet_iristrain)[7] <- 'versicolor'
names(nnet_iristrain)[8] <- 'virginica'

# modeling   多个X预测多个Y
nn_formula <- setosa + versicolor + virginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
nn <- neuralnet(nn_formula, data=nnet_iristrain, hidden=c(3)) 
plot(nn)
mypredict <- compute(nn, iristest[-5])$net.result 

# Consolidate multiple binary output back to categorical output
maxidx <- function(arr) {
    return(which(arr == max(arr)))
    } 
idx <- apply(mypredict, 1 , maxidx) # 1 row
prediction <- c('setosa', 'versicolor', 'virginica')[idx] 
table(prediction, iristest$Species)
# Neural networks are very good at learning non-linear functions. 
# They can even learn multiple outputs simultaneously, 
# though the training time is relatively long

# SVM  也可以用于多个Y的预测
library(e1071) 
tune <- tune.svm(Species~., 
                 data=iristrain, gamma=10^(-6:-1), cost=10^(1:6))
summary(tune) 

model <- svm(Species~., data=iristrain, 
             method='C-classifcation', kernel='radial', 
             probability=T, gamma=0.001, cost=10000)
prediction <- predict(model, iristest, probability=T)
table(iristest$Species, prediction)
# Although it is a binary classifer, it can be easily extended to a multi-class classifcation 
# by training a group of binary classifers and using “one vs all” or “one vs one” as predictors.

# BAYESIAN NETWORK AND NAÏVE BAYES  （OUTPUT必须为categorical
library(e1071) 
model <- naiveBayes(Species~., data=iristrain) 
prediction <- predict(model, iristest[,-5])
table(prediction, iristest[,5])

# KNN
library(class)
train_input <- as.matrix(iristrain[,-5]) 
train_output <- as.vector(iristrain[,5]) 
test_input <- as.matrix(iristest[,-5]) 

prediction <- knn(train_input, test_input, train_output, k=5) 
table(prediction, iristest$Species)

# PRO: The strength of K-nearest neighbor is its simplicity.  No model needs to be trained
# CON: Tt doesn’t handle high numbers of dimensions well.


# DECISION TREE
library(rpart) 
#Train the decision tree 
treemodel <- rpart(Species~., data=iristrain) 
plot(treemodel) 
text(treemodel, use.n=T) 
#Predict using the decision tree 
prediction <- predict(treemodel, newdata=iristest, type='class') 
#Use contingency table to see how accurate it is 
table(prediction, iristest$Species) 

# TREE ENSEMBLES (RandomForest)
library(randomForest) 
#Train 100 trees, random selected attributes 
model <- randomForest(Species~., data=iristrain, nTree=500) 
#Predict using the forest 
prediction <- predict(model, newdata=iristest, type='class') 
table(prediction, iristest$Species) 
importance(model)

# Gradient-based models
library(gbm) 
iris2 <- iris 
newcol = data.frame(isVersicolor=(iris2$Species=='versicolor')) 
iris2 <- cbind(iris2, newcol) 
# or iris2$isVersicolor2 = ifelse(iris2$Species=="versicolor",'Yes','No')
iris2[45:55,]

formula <- isVersicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width 
model <- gbm(formula, data=iris2,
             n.trees=1000, interaction.depth=2, distribution='bernoulli')
prediction <- predict.gbm(model, iris2[45:55,], 
                          type='response', n.trees=1000) 
round(prediction, 3) 
summary(model)
