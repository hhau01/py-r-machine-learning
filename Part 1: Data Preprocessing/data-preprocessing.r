# data preprocessing
# F1 to see docs

# import dataset
dataset = read.csv('Data.csv')

# take care of missing data
# return true if data in column Age is missing, false if not
dataset$Age = ifelse(is.na(dataset$Age),
                    ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                    dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# encode categorical data
dataset$Country = factor(dataset$Country,
                         level = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2 ,3))
dataset$Purchased = factor(dataset$Purchased,
                         level = c('No', 'Yes'),
                         labels = c(0, 1))

# split dataset into training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
# 20% goes into test
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)