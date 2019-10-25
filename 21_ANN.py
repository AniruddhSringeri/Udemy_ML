#ARTIFICIAL NEURAL NETWORKS

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #avoiding dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and first hidden layer
#Here, for the no. of layers of hidden layer, it is generally good to have no. of nodes equal to average of no. of nodes in first and last layers.
#(11 + 1)/2 == 6
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))   #relu -> rectifier function as activation function

#Adding second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) #input_dim is not needed because it is only needed for the first hidden layer, to let it know what to expect from the input layer.

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))  #NOTE: For a dependent variable having two or more categories, like two or more countries, activation function should be softmax function instead of sigmoid.

#Compiling the ANN
#adam is a Stochastic Gradient Descent algorithm.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  #threshold is 0.5 to predict as 0 or 1. Removes probabilities.

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
