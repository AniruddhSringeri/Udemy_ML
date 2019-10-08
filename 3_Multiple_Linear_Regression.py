#Multiple Linear Regression
#Running the whole code is not the point. Different sections of the code do different tasks, and some sections can replace some previous ones.
#Therefore, this code shows how my sequential learning of ML went.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Here, X will be a matrix of 4 independent variables 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #LabelEncoder is just for encoding into numbers, while OneHotEncoder is for dummy encoding.
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap => removing one column from the three dummy variable columns
X = X[:, 1:]

#Feature Scaling is taken care of by the multiple linear regression library.

#Splitting of dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Prediction of Test result
Y_pred = regressor.predict(X_test)

#But, this may not be the optimal solution. Because, our model may have some independent variables which may not be very statistically significant and may not affect the result much.
#And, there may be some variables which have a high effect on our result. 
#So, we can agree that making use of all independent variables is not optimal.

#Building the optimal model using Backward Elimination
import statsmodels.api as sm
#NOTE: The formula for multiple linear regression is y = b0 + b1*x1 + b2*x2 + ... + bn*xn, and b0 is a constant. But since the statsmodel.formula.api library that we imported does not support a constant in the equation, we need to rewrite the equation as y = b0*x0 + ... where x0 = 1.              
#NOTE: The constant b0 feature is present in the LinearRegression library from above, but just not in this particular library.
#Therefore, we need to add an extra column of 1's to the X array.
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#Next step is going to be to create an optimal matrix instead of X, which contains only a team of statistically significant independent variables or features.
#Backward elimination consists of initializing the optimal feature array with all the independent variables, and then removing the insignificant features one-by-one in steps.
X_opt = X[:, [0,1,2,3,4,5]]
#We will use a class of sm called OLS(Ordinary Least Square) to create a new regressor to fit to our model.
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#Note that we are going through all the steps of backward elimination. Refer the steps for clarity.
#Next, we have to remove the highest p-valued predictor(independent variable) if its p-value exceeds the level of significance of the problem(LS = 0.05). Otherwise, just finalize the model. There is no need to remove any predictor. All of them are statistically significant.
regressor_OLS.summary()
#Seeing the summary, we know that x2(the New York dummy variable column) has a p-value > LS(=0.05) and is not really statistically significant for our purposes. Hence, we ought to remove it from the X_opt array and make a new model, check for insignificant predictors again, and the cycle is repeated until there is no insignificant variable left in the most recently created model.
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#From summary, we see that x1 => Florida dummy variable is not really significant(i.e, its p-value > 0.05). Therefore, remove it.
X_opt = X_opt[:, [0, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#Remove column 2
X_opt = X_opt[:, [0, 1, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#Remove marketing column
X_opt = X_opt[:, [0, 1]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()