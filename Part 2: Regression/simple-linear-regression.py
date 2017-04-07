# simple linear regression

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# print full numpy array without truncation
np.set_printoptions(threshold=np.inf)

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# split dataset into training set and test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# 20% goes into test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

'''
# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# fit simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict test set results
y_pred = regressor.predict(X_test)

# visualize training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualize test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()