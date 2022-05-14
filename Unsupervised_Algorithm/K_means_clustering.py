# Importing the required libraries 
import matplotlib.pyplot as plt  
from matplotlib import style  
style.use('ggplot')  
import numpy as np  
  
#Dummy data  
X = np.array([  
              [1, 2 ],  
              [5, 3],
              [1, 4],  
              [2, 3]] 
              )  
  
#Plot of the Dummy Data  
plt.scatter(X[:,0], X[:,1], s=150)  
plt.show()  
  
#Defining the colors that can be used in plots  
colors =["clus1","clus2"]  
  
#Defination of the class K_Means  
#Using this class we are implementing all the steps required  
#to be performed to generate a k-means model  
class K_Means:  
    def __init__(self, k=2, tol=0.001, max_iter=300):  
        self.k = k  
        self.tol = tol  
        self.max_iter = max_iter  
  
    #Function to fit the dummy mode on to the model  
    def fit(self,data):  
  
        self.centroids = {}  
  
        for i in range(self.k):  
            self.centroids[i] = data[i]  
  
        for i in range(self.max_iter):  
            self.classifications = {}  
  
            for i in range(self.k):  
                self.classifications[i] = []  
  
            for featureset in data:  
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]  
                classification = distances.index(min(distances))  
                self.classifications[classification].append(featureset)  
  
            prev_centroids = dict(self.centroids)  
  
            for classification in self.classifications:  
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)  
  
            optimized = True  
  
            for c in self.centroids:  
                original_centroid = prev_centroids[c]  
                current_centroid = self.centroids[c]  
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:  
                    np.sum((current_centroid-original_centroid)/original_centroid*100.0)  
                    optimized = False  
  
            if optimized:  
                break  
  
    #Function to generate prediction result  
    def predict(self,data):  
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]  
        classification = distances.index(min(distances))  
        return classification  
clf = K_Means()  
clf.fit(X)  
  
#putting the centroid values on to the plot  
for centroid in clf.centroids:  
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],  
                marker="o", color="k", s=150, linewidths=5)  
for classification in clf.classifications:  
    color = colors[classification]  
    for featureset in clf.classifications[classification]:  
        #plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5) 
        print(featureset[0], featureset[1],color)
  
#Priting the centroid values  
for centroid in clf.centroids:  
    print(clf.centroids[centroid][0],clf.centroids[centroid][1])  
