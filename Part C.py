# Introduction to Artificial Intelligence
# Homework 6 - Part A
# Based on class example, code by Juan Carlos Rojas
# Adrian Quiros, Mariela Venegas

# second-order polynomial regression model 


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

# Standardize only numeric features only
numeric_cols = train_data.select_dtypes(include=np.number).columns
train_means = train_data[numeric_cols].mean()
train_stds = train_data[numeric_cols].std()
train_data[numeric_cols] = (train_data[numeric_cols] - train_means) / train_stds
test_data[numeric_cols] = (test_data[numeric_cols] - train_means) / train_stds

# -------------------------------------------------------------
# Separate numeric vs categorical (dummy) columns / Converting data to PyTorch tensors
# -------------------------------------------------------------
num_cols = [c for c in train_data.columns if train_data[c].nunique() > 10]  # numeric
cat_cols = [c for c in train_data.columns if c not in num_cols]             # dummy

X_num = torch.tensor(train_data[num_cols].values, dtype=torch.float32)
X_cat = torch.tensor(train_data[cat_cols].values, dtype=torch.float32)
Y = torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32)

X_num_test = torch.tensor(test_data[num_cols].values, dtype=torch.float32)
X_cat_test = torch.tensor(test_data[cat_cols].values, dtype=torch.float32)
Y_test = torch.tensor(test_labels.values.reshape(-1, 1), dtype=torch.float32)

n_num = X_num.shape[1]
n_cat = X_cat.shape[1]


# -------------------------------------------------------------
# Create and initialize weights and bias
# Model parameters: 2nd-order for numeric, 1st-order for categorical
# -------------------------------------------------------------
W1 = torch.randn((n_num, 1), dtype=torch.float32, requires_grad=True)   # linear terms for numeric
W2 = torch.randn((n_num, 1), dtype=torch.float32, requires_grad=True)   # quadratic terms for numeric
Wc = torch.randn((n_cat, 1), dtype=torch.float32, requires_grad=True)   # categorical weights
B = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)



# History lists
train_cost_hist = []
test_cost_hist = []
 


# ============================================================
# Training constants
# ============================================================
learning_rate = 0.01
n_iterations = 2000
eval_step = 100     # evaluate and record MSE every eval_step iterations


# ============================================================
# Training loop
# ============================================================

for iteration in range(n_iterations):

    # Forward pass: predictions
    Y_pred = B + X_num @ W1 + (X_num ** 2) @ W2 + X_cat @ Wc

    # Mean squared error 
    mse = torch.mean((Y_pred - Y) ** 2) 

    # Compute gradients of MSE with respect to W & B
    # Will be stored in W.grad & B.grad
    mse.backward()

    # Gradient descent step: W = W - lr * dW
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        W2 -= learning_rate * W2.grad
        Wc -= learning_rate * Wc.grad
        B -= learning_rate * B.grad

        # Zero gradients before next step
        W1.grad.zero_()
        W2.grad.zero_()
        Wc.grad.zero_()
        B.grad.zero_()

    # Evaluate and record cost every eval_step iterations
    if iteration % eval_step == 0:
        # Evaluate and record training and test MSE (once per eval step)
        with torch.no_grad():
            # Training MSE (current)
            mse_train = mse.item()

            # Test MSE (on held-out test set)
            Y_pred_test = B + X_num_test @ W1 + (X_num_test ** 2) @ W2 + X_cat_test @ Wc
            mse_test = torch.mean((Y_pred_test - Y_test) ** 2).item()

            # Record
            train_cost_hist.append(mse_train)
            test_cost_hist.append(mse_test)

        print(f"Iteration {iteration:4d}: Train MSE: {mse_train:.1f} Test MSE: {mse_test:.1f}")

# Stop tracking gradients on W1, W2, Wc & B
W1.requires_grad_(False)
W2.requires_grad_(False)
Wc.requires_grad_(False)
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