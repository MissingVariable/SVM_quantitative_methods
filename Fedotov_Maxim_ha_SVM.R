library(caret)
library(ggplot2)
library(tictoc)
library(kernlab)
library(dplyr)
library(e1071)

setwd("C:/Users/Максим/Downloads")

df <- read.csv("winequality-red.csv", header = TRUE)

str(df)
summary <- summary(df)
C_implied <- 20
gamma_implied <- 0.5
sigma_implied <- 1/sqrt(gamma_implied) #Это нигде не используется, т.к. в пакетах R e1071 и kernlab gamma = sigma, но, по идее, между ними должно быть такое соотношение, если гамма и сигма рассматриваются как канонические параметры ядерных функций
k <- 5

ggplot(df, aes(x=quality)) + labs(x = 'Качество', y = 'Число видов', title = "Распределение видов вина по качеству") + geom_histogram(bins = 16) 

count(df, quality)

threshold <- 5

df$class <- as.integer(df$quality > threshold)
df$class <- as.factor(df$class)
levels(df$class) <- list(class2 = "0", class1 = "1")

hist(as.integer(df$quality > threshold))

set.seed(123)
train_ratio <- 0.75
train.index <- createDataPartition(y = df$class, p = train_ratio, list = FALSE)
train.df <- df[train.index, ]
test.df <- df[-train.index, ]

#Package kernlab svm 
tic()
m1 <- ksvm(class ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + 
             total.sulfur.dioxide + density + pH + sulphates + alcohol, data = train.df, kernel = "rbfdot", kpar = list(sigma = gamma_implied), 
           C = C_implied)
toc()

#Package e1071 svm 
tic()
m1_e <- svm(class ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + 
            total.sulfur.dioxide + density + pH + sulphates + alcohol, data = train.df, kernel = "radial", gamma = gamma_implied, 
          cost = C_implied)
toc()

calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}

test.df$class.pred <- predict(m1, test.df)
table(test.df$class.pred, test.df$class)

calc_class_err(test.df$class, test.df$class.pred)

test.df$class.pred <- predict(m1_e, test.df)
table(test.df$class.pred, test.df$class)

calc_class_err(test.df$class, test.df$class.pred)

index<-m1@alphaindex 

SV_class2 <- sum(train.df[m1@alphaindex[[1]],]$class == "class2")
SV_class1 <- sum(train.df[m1@alphaindex[[1]],]$class == "class1")

m2 <- ksvm(class ~ residual.sugar + alcohol, data = train.df, kernel = "rbfdot", kpar = list(sigma = gamma_implied), 
           C = C_implied, cross = k)
m2

m2_e <- svm(class ~ residual.sugar + alcohol, data = train.df, kernel = "radial", gamma = gamma_implied, 
            cost = C_implied, cross = k)


plot(m2_e, residual.sugar ~ alcohol, data = train.df)
plot(m2_e, residual.sugar ~ alcohol, data = test.df)

ctrl <- tune.control(sampling = "cross", cross = k)

x <- subset(train.df, select = -c(class, quality) )
x_test <- subset(test.df, select = -c(class, quality, class.pred))

set.seed(223)
svm_tune <- tune(svm, train.x = x, train.y = train.df$class, kernel="radial", 
                 ranges = list(cost=seq(from = 1, to = 30, by = 1), gamma=seq(from = 0.1, to = 3, by = 0.1)),
                 tunecontrol = ctrl)
print(svm_tune)
tunedModel <- svm_tune$best.model
summary(tunedModel)

test.df$class.pred <- predict(tunedModel, x_test)
table(test.df$class.pred, test.df$class)

calc_class_err(test.df$class, test.df$class.pred)

train.df$class.pred <- predict(tunedModel, x)
table(train.df$class.pred, train.df$class)

calc_class_err(train.df$class, train.df$class.pred)

#Подбор quality_threshold
df_threshold <- df

err_k <- numeric(6)
for (i in seq(3, 8, 1)) {
  quality_threshold = i
  df_threshold$class <- as.integer(df_threshold$quality > quality_threshold)
  df_threshold$class <- as.factor(df_threshold$class)
  levels(df_threshold$class) <- list(class2 = "0", class1 = "1")
  
  set.seed(123)
  train.index_threshold <- createDataPartition(y = df_threshold$class, p = train_ratio, list = FALSE)
  train.df_threshold <- df_threshold[train.index_threshold, ]
  test.df_threshold <- df_threshold[-train.index_threshold, ]
  
  ctrl <- trainControl(method = "repeatedcv", number = k, repeats = k, classProbs = TRUE, summaryFunction = twoClassSummary, search = "random")
  modelControl <- train(class ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + 
                          total.sulfur.dioxide + density + pH + sulphates + alcohol, data = train.df, method = "svmRadial", trControl = ctrl, tuneLength = 3)
  
  test.df_threshold$class.pred <- predict(modelControl, test.df_threshold)
  err_k[i-2] = calc_class_err(test.df_threshold$class.pred, test.df_threshold$class)
}
err_k

