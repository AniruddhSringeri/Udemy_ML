#DATA PREPROCESSING


#IMPORTING THE LIBRARIES


#A library is a tool that you can use to do a specific job.
#numpy is a mathematical library.
#matplotlib is a plotting library.
#pandas is really useful to import and manage datasets.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET

#First, you need to have a working directory, where you have the .py file and the dataset, i.e, .csv file.
dataset = pd.read_csv('Data.csv') #The dataset is now created.
#Next step is to create a matrix of independent variables(those which decide the value of the dependent variable, i.e, the Purchased variable in this example)
X = dataset.iloc[:, :-1].values #this command selects and stores all rows in all columns except the last column in dataset.(because the last column is the dependent variable).
#iloc[rows, columns]
Y = dataset.iloc[:, -1].values



#WHEN THERE IS MISSING DATA
#We can just remove the lines having missing data. But this can result in loss of crucial data.
#Better, we can just fill in the mean of the remaining data points of the same column for a missing data.
#We can do this using scikit-learn.

from sklearn.preprocessing import Imputer
#creating an object of Imputer class
#cmd+i => info about any class/...
#axis = 0 => takes mean of column. axis = 1 => mean of row.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#We need to fit these imputer object to the X matrix.
imputer = imputer.fit(X[:, 1:3]) #we are fitting imputer to the 2nd and 3rd columns of X(which have missing data)
X[:, 1:3] = imputer.transform(X[:, 1:3]) #This is the method which will fill in the mean values.

#Encoding categorical data
#Categorical data contains data classified into categories, ex: countries France, Spain, Germany in the .csv file.
#Since Machine learning primarily works on numbers, working on such categories will be tricky.
#Therefore, we have to encode such categories into numbers.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #LabelEncoder is just for encoding into numbers, while OneHotEncoder is for dummy encoding.
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
#But since each country gets a different number, like 0, 1, 2, it seems to imply that one country is 'greater' than the other in some way.
#To prevent this, we are going to make use of dummy variables(dummy encoding).
#Here, each country column data point is transformed into three columns, i.e, if country1 = 1, the other country columns will be zero.
#Hence, in each row, one of the country columns will be 1 and the others will be zero.
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)


#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


#FEATURE SCALING
#This is needed to make sure that no one variable dominates the other. This often happens when one variable has values very much larger than the other variable.
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)




