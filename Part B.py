# Introduction to Artificial Intelligence
# Homework 6 - Part B & C
# Adrian Quiros, Mariela Venegas

import time
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import matplotlib.pyplot as plt

# -----------------------------
# Helper to collapse rare categories
# -----------------------------
def collapse_small_categories(df, col, min_count=10, other_label="others"):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df

# -----------------------------
# Load and prepare the data
# -----------------------------
df = pd.read_csv("/Users/marielavenegas/Downloads/Unit14_Examples/vehicles_clean2.csv", header=0)
df = collapse_small_categories(df, "manufacturer", min_count=100)
df = pd.get_dummies(df, prefix_sep="_", drop_first=True, dtype=int)

labels = df["price"]
df = df.drop(columns="price")

train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(
    df, labels, test_size=0.2, shuffle=True, random_state=2025
)

# Standardize ONLY numeric (continuous) features
numeric_cols = train_data.select_dtypes(include=np.number).columns
train_means = train_data[numeric_cols].mean()
train_stds = train_data[numeric_cols].std()
train_data[numeric_cols] = (train_data[numeric_cols] - train_means) / train_stds
test_data[numeric_cols]  = (test_data[numeric_cols]  - train_means) / train_stds

# Separate numeric vs. categorical (dummy) columns
# (same heuristic as your Part A)
num_cols = [c for c in train_data.columns if train_data[c].nunique() > 10]  # numeric
cat_cols = [c for c in train_data.columns if c not in num_cols]             # dummy

# -----------------------------
# Convert to PyTorch tensors
# -----------------------------
X_num       = torch.tensor(train_data[num_cols].values, dtype=torch.float32)
X_cat       = torch.tensor(train_data[cat_cols].values, dtype=torch.float32)
Y           = torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32)

X_num_test  = torch.tensor(test_data[num_cols].values, dtype=torch.float32)
X_cat_test  = torch.tensor(test_data[cat_cols].values, dtype=torch.float32)
Y_test      = torch.tensor(test_labels.values.reshape(-1, 1), dtype=torch.float32)

n_num = X_num.shape[1]
n_cat = X_cat.shape[1]
nsamples = X_num.shape[0]

# -----------------------------
# Model parameters
#   - 2nd-order for numeric
#   - 1st-order for categorical
# -----------------------------
torch.manual_seed(2025)
W1 = torch.randn((n_num, 1), dtype=torch.float32, requires_grad=True)   
W2 = torch.randn((n_num, 1), dtype=torch.float32, requires_grad=True)   
Wc = torch.randn((n_cat, 1), dtype=torch.float32, requires_grad=True)   
B  = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)

# -----------------------------
# Training constants (mini-batch)
# -----------------------------
batch_size   = 1024        
num_epochs   = 30         
learning_rate= 0.02       
eval_every_k_batches = 1 

print(f"Batch size: {batch_size} | Epochs: {num_epochs} | LR: {learning_rate}")

num_batches = int(np.ceil(nsamples / batch_size))
total_iterations = num_epochs * num_batches
print(f"Batches/epoch: {num_batches} | Total iters: {total_iterations}")

# -----------------------------
# Training loop (mini-batches)
# -----------------------------
train_cost_hist = []
test_cost_hist  = []
iters_hist      = []

start_time = time.time()
iteration = 0

for epoch in range(num_epochs):
   
    perm = torch.randperm(nsamples)

    for b in range(num_batches):
        idx = perm[b * batch_size : min((b + 1) * batch_size, nsamples)]
        Xn_b = X_num[idx]
        Xc_b = X_cat[idx]
        Y_b  = Y[idx]

        # Forward: y_hat = B + X_num@W1 + (X_num**2)@W2 + X_cat@Wc
        Y_pred_b = B + Xn_b @ W1 + (Xn_b ** 2) @ W2 + Xc_b @ Wc

        # Loss (MSE)
        mse = torch.mean((Y_pred_b - Y_b) ** 2)

        # Backprop
        mse.backward()

        # Parameter update
        with torch.no_grad():
            W1 -= learning_rate * W1.grad
            W2 -= learning_rate * W2.grad
            Wc -= learning_rate * Wc.grad
            B  -= learning_rate * B.grad

            # Zero grads
            W1.grad.zero_(); W2.grad.zero_(); Wc.grad.zero_(); B.grad.zero_()

        # Periodic eval
        iteration += 1
        if iteration % eval_every_k_batches == 0:
            with torch.no_grad():
                # Current training MSE (on the last mini-batch loss)
                train_cost_hist.append(mse.item())
                iters_hist.append(iteration)

                # Test MSE on full held-out set
                Y_pred_test = B + X_num_test @ W1 + (X_num_test ** 2) @ W2 + X_cat_test @ Wc
                mse_test = torch.mean((Y_pred_test - Y_test) ** 2).item()
                test_cost_hist.append(mse_test)

            
            if iteration % max(1, total_iterations // 10) == 0:
                print(f"[Epoch {epoch+1:2d}/{num_epochs}] "
                      f"Iter {iteration:5d}/{total_iterations}: "
                      f"Train MSE {train_cost_hist[-1]:.1f} | Test MSE {mse_test:.1f}")

training_time = time.time() - start_time
print(f"\nTraining time (mini-batch): {training_time:.2f} seconds")

# -----------------------------
# Final metrics
# -----------------------------
final_train_mse = train_cost_hist[-1]
final_test_mse  = test_cost_hist[-1]
print(f"Final Train RMSE: {final_train_mse ** 0.5:.1f}")
print(f"Final Test  RMSE: {final_test_mse  ** 0.5:.1f}")

# -----------------------------
# Plots
# -----------------------------
plt.figure()
plt.plot(iters_hist, train_cost_hist, label="Train MSE")
plt.plot(iters_hist, test_cost_hist,  label="Test MSE")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Cost evolution (mini-batch)")
plt.legend()
plt.tight_layout()
plt.show()
