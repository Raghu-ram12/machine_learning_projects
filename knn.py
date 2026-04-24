import pandas as pd
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris =datasets.load_iris()

X=iris.data
Y=iris.target 


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)



def get_distance(x1, x2):

    return np.sqrt(np.sum((x1 - x2) ** 2))


class Knn:
    def __init__(self, k=3):
        self.k = k

    def fit(self,X, Y):

        self.x_train = X

        self.y_train = Y

    def predict(self, X):

        predicted_labels = [self.predict_one(x) for x in X]

        return np.array(predicted_labels)

    def predict_one(self, x):

        distances = [get_distance(x, x_train) for x_train in self.x_train]

        k_indices = np.argsort(distances)[: self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common_class = Counter(k_nearest_labels).most_common(1)

        # counter returns as list of tuples with [(1,2)]

        return most_common_class[0][0]


clf=Knn()

clf.fit(x_train,y_train)

y_predictions=clf.predict(x_test)

accuracy = np.mean(y_predictions == y_test)
print(f"Test accuracy: {accuracy:.3f}")
