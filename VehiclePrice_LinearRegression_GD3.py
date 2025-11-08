# Introduction to Machine Learning
# Vehicle Price dataset
# Linear regression solved through gradient descent in PyTorch
# Version 3: plot training and test MSE history
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import matplotlib.pyplot as plt

def collapse_small_categories(df, col, min_count=10, other_label="others"):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df

# Load and prepare the data
df = pd.read_csv("vehicles_clean2.csv", header=0)
df = collapse_small_categories(df, "manufacturer", min_count=100)
df = pd.get_dummies(df, prefix_sep="_", drop_first=True, dtype=int)
labels = df["price"]
df = df.drop(columns="price")
train_data, test_data, train_labels, test_labels = \
            sklearn.model_selection.train_test_split(df, labels,
            test_size=0.2, shuffle=True, random_state=2025)

# Standardize scale for all columns
train_means = train_data.mean()
train_stds = train_data.std()
train_data = (train_data - train_means) / train_stds
test_data  = (test_data  - train_means) / train_stds

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# ============================================================
# Training constants
# ============================================================
learning_rate = 0.01
n_iterations = 2000
print_step = 100

# ============================================================
# Convert data to PyTorch tensors
# ============================================================
X = torch.tensor(train_data.values, dtype=torch.float32)
Y = torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32)

X_test = torch.tensor(test_data.values, dtype=torch.float32)
Y_test = torch.tensor(test_labels.values.reshape(-1, 1), dtype=torch.float32)

# ============================================================
# Create and initialize weights and bias
# ============================================================

# Create a vector of coefficients with random values between -1 and 1
W = torch.rand((ncoeffs, 1)) * 2 - 1

# Create a bias variable initialized to zero
B = torch.zeros((1, 1), dtype=torch.float32)

# Start tracking gradients on W
W.requires_grad_(True)
B.requires_grad_(True)

# ============================================================
# Training loop
# ============================================================

# History lists
train_cost_hist = []
test_cost_hist = []
eval_step = 100     # evaluate and record MSE every eval_step iterations

for iteration in range(n_iterations):

    # Forward pass: predictions
    Y_pred = X @ W + B

    # Mean squared error 
    mse = torch.mean((Y_pred - Y) ** 2) 

    # Compute gradients of MSE with respect to W & B
    # Will be stored in W.grad & B.grad
    mse.backward()

    # Gradient descent step: W = W - lr * dW
    with torch.no_grad():
        W -= learning_rate * W.grad
        B -= learning_rate * B.grad

    # Zero gradients before next step
    W.grad.zero_()
    B.grad.zero_()

    # Evaluate and record cost every eval_step iterations
    if iteration % eval_step == 0:
        # Training MSE
        mse_train = mse.item()
        train_cost_hist.append(mse_train)
        
        # Test MSE
        with torch.no_grad():
            Y_pred_test = X_test @ W + B
            mse_test = torch.mean((Y_pred_test - Y_test) ** 2).item()
            test_cost_hist.append(mse_test)
        
        print(f"Iteration {iteration:4d}: Train MSE: {mse_train:.1f} Test MSE: {mse_test:.1f}")

# Stop tracking gradients on W & B
W.requires_grad_(False)
B.requires_grad_(False)

# Print the final MSEs
train_rmse = (train_cost_hist[-1]) ** 0.5
test_rmse = (test_cost_hist[-1]) ** 0.5
print(f"Training RMSE: {train_rmse:.1f}")
print(f"Test RMSE: {test_rmse:.1f}")

# Plot MSE history
iterations_hist = [i for i in range(0, n_iterations, eval_step)]
plt.plot(iterations_hist, train_cost_hist, "b", label="Train MSE")
plt.plot(iterations_hist, test_cost_hist, "r", label="Test MSE")
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Cost evolution")
plt.legend()
plt.show()

