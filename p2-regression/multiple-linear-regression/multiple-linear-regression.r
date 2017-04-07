# multiple linear regression
# F1 to see docs

# import dataset
dataset = read.csv('50_Startups.csv')
# dataset = dataset[, 2:3]

# encode categorical data
dataset$State = factor(dataset$State,
                         level = c('New York', 'California', 'Florida'),
                         labels = c(1, 2 ,3))

# split dataset into training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
# 20% goes into test
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# # feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# fit multiple linear regression on training set
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
# data = training_set)
# . means all independent variables
regressor = lm(formula = Profit ~ .,
               data = training_set)

# predict test set results
y_pred = predict(regressor, newdata=test_set)

# build optimal model using backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

# remove predictor - State
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

# remove Administration
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

# remove Marketing.Spend because > 0.05 p-value
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)