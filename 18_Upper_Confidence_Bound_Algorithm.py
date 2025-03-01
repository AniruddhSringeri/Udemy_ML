#REINFORCEMENT LEARNING
#Upper Confidence Bound Algorithm
#The given dataset

#Importing the libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#Implementing UCB (There is no package. So, the algorithm has to be implemented from scratch.)
N = 10000 #size of dataset
d = 10 #no of ads
ads_selected = []
numbers_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sum_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward
    
#Visualising the result
plt.hist(ads_selected)
plt.title("Histogram of ad selection")
plt.xlabel("Ads")
plt.ylabel("No. of selections")
plt.show()
