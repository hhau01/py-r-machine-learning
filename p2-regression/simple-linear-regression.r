# simple linear regression
# F1 to see docs

# import dataset
dataset = read.csv('Salary_Data.csv')

# split dataset into training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
# 20% goes into test
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# # feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# fit simple linear regression to training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# summary(regressor)

# predict test set results
y_pred = predict(regressor, newdata = test_set)

# visualize training set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Year of Experience') +
  ylab('Salary')

# visualize test set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Year of Experience') +
  ylab('Salary')