import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

"""NOTE: Feature Scaling will be taken care of by the linear regression model.
So, there is no need to do it explicitly. But, this will not always be the case. Most ML models do 
have in-built automatic feature scaling, but there are also some models which do not."""


#Fitting Simple Linear Regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting test set results
Y_pred = regressor.predict(X_test)
