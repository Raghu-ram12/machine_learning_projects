import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(X, Y, learning_rate=0.01, iters=1000, tolerance=1e-6):
    """
    Batch Gradient Descent for Multiple Linear Regression
    """
    m = len(Y)
    X_bias = np.c_[np.ones((m, 1)), X]  
    weights = np.random.randn(X_bias.shape[1], 1) * 0.01  
    costs = []
    prev_cost = float('inf')
    
    Y = Y.reshape(-1, 1)  
    
    for i in range(iters):
        
        predictions = X_bias.dot(weights)
        
        cost = (1/m) * np.sum((predictions - Y) ** 2)
        costs.append(cost)
        
      
        if abs(prev_cost - cost) < tolerance:
            print(f"Converged at iteration {i}")
            break
            
        prev_cost = cost
        
     
        gradients = (2/m) * X_bias.T.dot(predictions - Y)
        weights -= learning_rate * gradients
    
    return weights, costs


print("Loading insurance dataset...")
dataframe = pd.read_csv('insurance.csv')
df = dataframe.loc[:, ["age", "bmi", "smoker", "charges"]]

df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

print("Dataset shape:", df.shape)
print("\nDataset preview:")
print(df.head())

X_df = df[["age", "bmi", "smoker"]]
y_df = df["charges"]

X = X_df.to_numpy(dtype=float)
y = y_df.to_numpy(dtype=float)


X_min, X_max = X.min(axis=0), X.max(axis=0)
y_min, y_max = y.min(), y.max()

X_normalized = (X - X_min) / (X_max - X_min)
y_normalized = (y - y_min) / (y_max - y_min)

print(f"\nNormalization ranges:")
print(f"X min: {X_min}, X max: {X_max}")
print(f"y min: ${y_min:.2f}, y max: ${y_max:.2f}")


print("\nTraining model...")
theta, costs = gradient_descent(X_normalized, y_normalized, learning_rate=0.1, iters=3000)

print(f"Final cost: {costs[-1]:.6f}")
print(f"Learned weights: {theta.flatten()}")


plt.figure(figsize=(10, 6))
plt.plot(costs)
plt.title("Gradient Descent: Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.grid(True, alpha=0.3)
plt.show()

def predict_charges(age, bmi, smoker, theta, X_min, X_max, y_min, y_max):
    """
    Predict insurance charges for new patient
    """
 
    age_norm = (age - X_min[0]) / (X_max[0] - X_min[0])
    bmi_norm = (bmi - X_min[1]) / (X_max[1] - X_min[1])
    

    X_new_norm = np.array([[1, age_norm, bmi_norm, smoker]])
 
    predicted_norm = X_new_norm.dot(theta)[0, 0]
    predicted_charges = y_min + predicted_norm * (y_max - y_min)
    
    return predicted_charges

# Interactive prediction
print("\n" + "="*50)
print("INSURANCE CHARGES PREDICTOR")
print("="*50)

while True:
    try:
        print("\nEnter patient details (or 'quit' to exit):")
        age = float(input("Age: "))
        bmi = float(input("BMI: "))
        smoker = int(input("Smoker (0=No, 1=Yes): "))
        
        predicted = predict_charges(age, bmi, smoker, theta, X_min, X_max, y_min, y_max)
        print(f"\nPredicted annual charges: ${predicted:.2f}")
        
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        break
    except KeyboardInterrupt:
        break

