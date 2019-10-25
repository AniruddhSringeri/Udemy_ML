#Thompson Sampling

#Importing the libraries
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#Implementing UCB (There is no package. So, the algorithm has to be implemented from scratch.)
N = 10000 #size of dataset
d = 10 #no of ads
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
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
