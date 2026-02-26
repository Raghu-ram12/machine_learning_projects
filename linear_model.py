import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class linear_regression:
    def __init__(self,lr=0.01,n_iters=1000):
        
        self.lr=lr
        self.n_iters=n_iters
        self.bias=None
        self.weights=None
    
    def fit(self,x,y):
        costs=[]
        n_samples,n_features=x.shape
        self.weights=np.zeros(n_features)
        self.bias=1
        prev_cost=0
        for i in range(self.n_iters):

            predictions=x.dot(self.weights)
           
            cost=(1/n_samples)*np.sum(predictions-y)**2

            if(abs(prev_cost-cost)<1e-7):
                 print(f"converged at the iteration {i}")
                 break
            
            costs.append(cost)

            dw=(1/n_samples)*(x.T.dot(predictions-y))

            db=(1/n_samples)*(np.sum(predictions-y))
            
            self.weights-=self.lr*dw

            self.bias-=self.lr*db
        return costs
    

    def predict(self,X):
        
        predicted=X.dot(self.weights)+self.bias

        return predicted
        
    


model=linear_regression(lr=0.001,n_iters=5000)

data=pd.read_csv("Student_Performance.csv")
data["Extracurricular Activities"]=data["Extracurricular Activities"].map({"Yes":1,"No":0})

df_x=data.iloc[:,[0,1,2,3,4]]
df_y=data.iloc[:,[5]]

x = df_x.to_numpy()
y = df_y.to_numpy().flatten()

x_min=x.min(axis=0)
x_max=x.max(axis=0)
y_min=y.min()
y_max=y.max()

x_norm=(x-x_min)/((x_max-x_min))
y_norm=(y-y_min)/((y_max-y_min))

costs=model.fit(x=x_norm,y=y_norm)

print(model.weights)

plt.plot(costs)

plt.show()
