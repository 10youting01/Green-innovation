import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_X_y_from_dataset_base, evaluate_model, save_model, save_predictions, save_metrics_log
from models.mlp import build_mlp_model

## Read and check data
handmade_features_score = pd.read_csv("../data/handmade_features_all_v1_corrected_txt_divided_with_score.csv")
handmade_features_score.dropna(inplace=True)
handmade_features_score = handmade_features_score[handmade_features_score["greenwash_score"] != 0]

## Sort the data by year
handmade_features_score.sort_values(
    by=["year1"],
    ascending=True,
    inplace=True,
)
X, y = get_X_y_from_dataset_base(handmade_features_score)

tscv = TimeSeriesSplit(n_splits=5)

models = {
    'LR': LinearRegression(),
    'SVR': SVR(),
    'DT': DecisionTreeRegressor(random_state=42),
    'RF': RandomForestRegressor(random_state=42),
    'XGB': XGBRegressor(random_state=42)
}

# 儲存每個模型的 5 fold 結果
metrics_summary = {name: {'MAE': [], 'MSE': [], 'MAPE': [], 'RMSE': []} for name in models}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Standardization 僅用在 SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        if name == 'SVR':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        metrics_summary[name]['MAE'].append(mae)
        metrics_summary[name]['MSE'].append(mse)
        metrics_summary[name]['RMSE'].append(rmse)
        metrics_summary[name]['MAPE'].append(mape)

# Step 5: 彙總平均表現
print("\n📊 Baseline Model Performance Summary (5-Fold CV):")
for name, scores in metrics_summary.items():
    print(f"\n{name}:")
    print(f"  MAE  : {np.mean(scores['MAE']):.4f}")
    print(f"  MSE  : {np.mean(scores['MSE']):.4f}")
    print(f"  MAPE : {np.mean(scores['MAPE']):.4f}")
    print(f"  RMSE : {np.mean(scores['RMSE']):.4f}")
