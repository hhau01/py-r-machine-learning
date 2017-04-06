# data preprocessing
# ctrl+I to see docs

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# print full numpy array without truncation
np.set_printoptions(threshold=np.inf)

# import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
                
# replace missing data with mean
from sklearn.preprocessing import Imputer
# missing_values: search for all NaN, strategy: mean,
# axis: 0 for col, 1 for row
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# column 1-2
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# encode categorical data using dummy variables
# don't want to just use labelencoder because python will think
# certain countries rank higher than others
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# split dataset into training set and test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# 20% goes into test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)