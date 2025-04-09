import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

def get_X_y_from_dataset_without_handmade_features(df):
    X = df[[col for col in df.columns if col.startswith("dim_")]]
    y = df["greenwash_score"].values
    return X.values, y

def get_X_y_from_dataset_with_handmade_features(df):
    X_bert = df[[col for col in df.columns if col.startswith("dim_")]].values
    X_hand = df[[col for col in df.columns if col.startswith("hm_")]].values
    y = df["greenwash_score"].values
    return X_bert, X_hand, y

def train_one_fold(model, X_train, y_train, X_val, y_val, 
                   lr=0.005, max_epochs=500, early_stop_rounds=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    best_val_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_rounds:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)

    # Record training and testing sample sizes
    metrics = {}
    metrics = record_train_test_samples(metrics, X_train, X_val)
    return model, metrics

def train_one_fold_with_hm(model, X_bert_train, X_hand_train, y_train, 
                   X_bert_val, X_hand_val, y_val,
                   lr=0.005, max_epochs=500, early_stop_rounds=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 轉成 tensor 並搬到裝置上
    Xb_train = torch.tensor(X_bert_train, dtype=torch.float32).to(device)
    Xh_train = torch.tensor(X_hand_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    Xb_val = torch.tensor(X_bert_val, dtype=torch.float32).to(device)
    Xh_val = torch.tensor(X_hand_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    best_val_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(Xb_train, Xh_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xb_val, Xh_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_rounds:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)

    # Record training and testing sample sizes
    metrics = {}
    metrics = record_train_test_samples(metrics, Xb_train, Xb_val, Xh_train, Xh_val)
    return model, metrics


def evaluate_model(y_true, y_pred): 
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model_class, path, device="cpu"):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_predictions(y_true, y_pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(path, index=False)

def save_metrics_log(metrics_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=4)

def predict(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).cpu().numpy().flatten()

    return y_pred

def predict_with_hm(model, X_bert, X_hand):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        Xb = torch.tensor(X_bert, dtype=torch.float32).to(device)
        Xh = torch.tensor(X_hand, dtype=torch.float32).to(device)
        y_pred = model(Xb, Xh).cpu().numpy().flatten()

    return y_pred

def record_train_test_samples(metrics, X_train, X_test, X_hand_train=None, X_hand_test=None):
    """
    Record the number of training and test samples for both BERT and handmade features.

    Args:
    - metrics (dict): Dictionary to store metrics information.
    - X_train (array): Training data for BERT features.
    - X_test (array): Test data for BERT features.
    - X_hand_train (array, optional): Training data for handmade features.
    - X_hand_test (array, optional): Test data for handmade features.

    Returns:
    - metrics (dict): Updated dictionary with training and test sample counts.
    """

    # Record BERT features (X_train, X_test)
    metrics["train_samples"] = len(X_train)  # Training samples count (BERT features)
    metrics["test_samples"] = len(X_test)   # Test samples count (BERT features)

    # If handmade features exist, record their samples too
    if X_hand_train is not None and X_hand_test is not None:
        metrics["handmade_train_samples"] = len(X_hand_train)  # Handmade features training count
        metrics["handmade_test_samples"] = len(X_hand_test)    # Handmade features testing count

    return metrics
