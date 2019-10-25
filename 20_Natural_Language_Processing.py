#Natural Language Processing

#Here, we predict whether a restaurant review is positive or negative.

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
#Here, it is a tsv file(tab separated) because commas would be common in regular sentences and a csv would mess up the algorithm.
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the text

#Cleaning the text (removing the non-useful words) is necessary because we will be using a bag of words model.
#the, numbers, and punctuation marks will be got rid of.
#Get rid of capitals
#Stemming: different forms of the same word will be removed and will be associated to the original word.
#For ex: loved, loving = love.

import re
import nltk
nltk.download('stopwords') #this downloads the irrelevent and insignificant words like the, an, a,...
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) #this removes everything except letters of the alphabet
review = review.lower()         #making everything lowercase
review = review.split()  
ps = PorterStemmer()   #Stemming = simplifying derived words into root words.(loved = love)
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   #removing insignificant words(like, a, an, ...)

#making review a string again
review = ' '.join(review)