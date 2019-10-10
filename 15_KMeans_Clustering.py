#K-Means Clustering Algorithm
#This dataset contains the information about the customers of a mall. The task is to cluster the different customers into different groups according to the annual income and the spending score given to them by the mall.

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values  #the rows of annual income and spending score

#To cluster the data, first we need to know the number of clusters to be made(K must be known).
#To find out the optimal number of clusters for our problem, we will use the elbow method.
from sklearn.cluster import KMeans
#Now, we will plot the elbow graph for iterations of the clusters.
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #wcss is also called inertia
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


#Optimal number of clusters turns out to be 5

#Applying kmeans to the mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
#fit_predict method tells us which cluster a client belongs to 
y_kmeans = kmeans.fit_predict(X)


#Visulising the results
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c = 'red', s = 100, label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'green', s = 100, label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c = 'magenta', s = 100, label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], c = 'blue', s = 100, label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], c = 'cyan', s = 100, label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], c = 'yellow', s = 300, label = 'Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
