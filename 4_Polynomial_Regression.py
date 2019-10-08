#Polynomial Regression
#Here, an interview for a company is taking place. The interviewee has said that he was between level 6 and 7 in his previous company and he was receiving a salary of 160k.
#The HR team has to check whether he is telling the truth by building a polynomial regression model based on the previous company's salary dataset and predict the salary for level 6.5 and compare the two numbers.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#NOTE: Since here the no. of data points is 10, splitting into training and test sets will be ridiculous.
#So, no splitting.

#Feature scaling will be taken care of by the library.

#Let's build both linear and polynomial regression models to compare between the two.

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



#Fitting Polynomial Regression to dataset
#This library is useful for adding polynomial terms to the linear regression equation.
from sklearn.preprocessing import PolynomialFeatures
#The poly_reg object should be able to add square, cube, ... terms to the X array.
poly_reg = PolynomialFeatures(degree = 4) #Here, we are adding only one column, that of square terms.
#Also, note that this particular library has the automatic feature of adding the constant term.(x0 = 1 for b0)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Predicting the result with Linear Regression
print(lin_reg.predict([[6.5]]))

#Predicting the result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))


