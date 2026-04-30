
import numpy as np 
import  pandas as pd 
import random
from numpy import argmin 

def init_centroids(data,k):

    rows=len(data)
    indices=random.sample(range(0,rows),k) 
    return data[indices] 
    

def get_distance_for_k_centroids(data,centroids):
    dists=[]

    for centroid in centroids:
         distance=[np.sqrt(np.sum((centroid-i)**2)) for i in data]
         dists.append(distance)
    
    return np.transpose(dists)

def calculate_new_centroids(data,k,labels):

     new_centroids = np.zeros((k, data.shape[1]))

     for cluster_id in range(k):

        points=data[labels==cluster_id]

        if len(points)>0:
            new_centroids[cluster_id]=(np.mean(points,axis=0)) 

        else:
            new_centroids[cluster_id]=(np.random.rand(data.shape[1])) 
    
     return new_centroids

def get_centroid_shift(old_centroids,new_centroids,k):

    return max([np.sqrt(np.sum(old_centroids[i]-new_centroids[i])**2)  for i in range(k)])

class KMeansClustering:
    def __init__(self,k,n_iters,tol):
        self.k=k 
        self.tol=tol 
        self.n_iters=n_iters
        self.centroids=[]
        self.labels=[]
    
    def fit(self,data):

        self.centroids=init_centroids(data,self.k)
        
        for _ in range(self.n_iters):
            old_centroids=np.copy(self.centroids)
            distances=get_distance_for_k_centroids(data,self.centroids)

            self.labels=argmin(distances,axis=1)

            self.centroids=calculate_new_centroids(data,self.k,self.labels)
        
            if get_centroid_shift(old_centroids,self.centroids,self.k) < self.tol:
                break

    
    def predict(self,data):
        
        dist=get_distance_for_k_centroids(data,self.centroids)

        labels=np.argmin(dist,axis=1)
        
        return labels 





