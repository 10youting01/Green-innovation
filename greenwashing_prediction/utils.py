import os
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

def get_X_y_from_dataset(df):
    X = df[[col for col in df.columns if col.startswith("dim_")]]
    y = df["greenwash_score"].values
    return X.values, y

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
    joblib.dump(model, path)

def save_predictions(y_true, y_pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(path, index=False)

def save_metrics_log(metrics_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=4)