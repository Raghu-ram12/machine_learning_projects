import numpy as np
from numpy import where
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.datasets import make_classification


class SupportVectorMachine:

    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000):

        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):

            for idx, x_i in enumerate(X):

                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (
                        2 * self.lambda_param * self.w - (y[idx] * x_i)
                    )
                    self.b += self.learning_rate * y[idx]

    def predict(self, X):

        result = np.dot(X, self.w) + self.b

        return np.sign(result)

    def plot(self, X, y):  # Add X, y params 
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", s=60, edgecolors="k", alpha=0.8)

        # Data range
        xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        xx = np.linspace(xmin, xmax, 200)
        yy = -(self.w[0] * xx + self.b) / self.w[1]
        plt.plot(xx, yy, "k-", linewidth=3, label="Decision boundary")

        # Margins (fixed equations)
        yy1 = -(self.w[0] * xx + self.b + 1) / self.w[1]  # w·x + b = -1
        yy2 = -(self.w[0] * xx + self.b - 1) / self.w[1]  # w·x + b = +1
        plt.plot(xx, yy1, "r--", linewidth=2, alpha=0.7, label="Support -1")
        plt.plot(xx, yy2, "g--", linewidth=2, alpha=0.7, label="Support +1")

        plt.xlim(xmin, xmax)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.legend()

        plt.grid(alpha=0.3)
        plt.show()


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
cancer = load_breast_cancer()
X = cancer.data[:,[0,5]]       # 2 features 
y = cancer.target      # 0 and 1

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train
clf = SupportVectorMachine(learning_rate=0.001, lambda_param=0.001, epochs=2000)
clf.fit(x_train, y_train)

# Predict
y_predict = clf.predict(x_test)
y_predict = np.where(y_predict <= 0, 0, 1)  

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_predict)*100:.2f}")
clf.plot(x_test,y_test) 