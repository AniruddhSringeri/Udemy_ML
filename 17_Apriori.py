#ASSOCIATED LEARNING
#People who bought... also bought...

#Apriori Algorithm
#Example: By placing cereals and milk close together in a store, the store can increase profits and induce customers to buy both of those things even if they started out thinking of buying only one of them.
#apyori.py is the file to be imported and it is the Python Software Foundation's implementation(module) for the apriori algorithm.

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the mall dataset with pandas
#Python will think that the first entries are the headings. So, header = None
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) #each transaction has to be a string in Apriori.
    
    
#Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


#Visualising the results
results = list(rules)