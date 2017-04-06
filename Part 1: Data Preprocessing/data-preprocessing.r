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