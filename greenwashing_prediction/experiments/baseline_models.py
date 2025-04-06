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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_X_y_from_dataset_with_handmade_features, evaluate_model

# 讀取資料
df = pd.read_csv("../data/handmade_features_all_v1_corrected_txt_divided_with_score.csv")
df.dropna(inplace=True)
df = df[df["greenwash_score"] != 0]
df.sort_values(by=["year1"], ascending=True, inplace=True)

X_bert, X_hand, y = get_X_y_from_dataset_with_handmade_features(df)
X = X_hand  # 轉成 NumPy，避免某些模型報錯
y = y         # 已是 NumPy

tscv = TimeSeriesSplit(n_splits=5)

# baseline 模型
models = {
    'LR': LinearRegression(),
    'SVR': SVR(),
    'DT': DecisionTreeRegressor(random_state=42),
    'RF': RandomForestRegressor(random_state=42),
    'XGB': XGBRegressor(random_state=42)
}

# 儲存每個模型每折的評估結果
metrics_summary = {name: {'MAE': [], 'MSE': [], 'MAPE': [], 'RMSE': []} for name in models}

# 執行每折
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\n Fold {fold + 1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 標準化（僅對 SVR 使用）
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

        # 使用模組化的 evaluate_model
        metrics = evaluate_model(y_test, y_pred)
        for metric_name in metrics:
            metrics_summary[name][metric_name].append(metrics[metric_name])

        print(f"{name} => MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}")

# 輸出平均表現
print("\n Baseline Model Performance Summary (5-Fold CV):")
for name, scores in metrics_summary.items():
    print(f"\n{name}:")
    print(f"  MAE  : {np.mean(scores['MAE']):.4f}")
    print(f"  MSE  : {np.mean(scores['MSE']):.4f}")
    print(f"  MAPE : {np.mean(scores['MAPE']):.4f}")
    print(f"  RMSE : {np.mean(scores['RMSE']):.4f}")


#  Fold 1
# LR => MAE: 90.97, RMSE: 172.20, MAPE: 11.40
# SVR => MAE: 8.02, RMSE: 10.46, MAPE: 1.48
# DT => MAE: 10.39, RMSE: 13.05, MAPE: 1.66
# RF => MAE: 9.11, RMSE: 11.46, MAPE: 1.54
# XGB => MAE: 9.60, RMSE: 12.15, MAPE: 1.56

#  Fold 2
# LR => MAE: 13.15, RMSE: 17.25, MAPE: 4.91
# SVR => MAE: 11.54, RMSE: 14.10, MAPE: 4.34
# DT => MAE: 12.65, RMSE: 16.03, MAPE: 4.73
# RF => MAE: 11.44, RMSE: 14.28, MAPE: 4.21
# XGB => MAE: 11.34, RMSE: 14.60, MAPE: 4.15

#  Fold 3
# LR => MAE: 9.83, RMSE: 12.82, MAPE: 10.11
# SVR => MAE: 9.62, RMSE: 12.17, MAPE: 50.41
# DT => MAE: 11.38, RMSE: 15.53, MAPE: 7.00
# RF => MAE: 9.40, RMSE: 11.63, MAPE: 48.27
# XGB => MAE: 10.63, RMSE: 13.15, MAPE: 22.75

#  Fold 4
# LR => MAE: 9.86, RMSE: 11.69, MAPE: 3.05
# SVR => MAE: 7.58, RMSE: 9.68, MAPE: 2.32
# DT => MAE: 11.91, RMSE: 14.59, MAPE: 2.00
# RF => MAE: 8.07, RMSE: 9.52, MAPE: 2.55
# XGB => MAE: 9.56, RMSE: 10.91, MAPE: 2.21

#  Fold 5
# LR => MAE: 8.39, RMSE: 10.52, MAPE: 1.32
# SVR => MAE: 5.80, RMSE: 7.75, MAPE: 0.93
# DT => MAE: 11.26, RMSE: 14.10, MAPE: 1.61
# RF => MAE: 6.67, RMSE: 8.74, MAPE: 1.15
# XGB => MAE: 7.56, RMSE: 9.25, MAPE: 1.07

#  Baseline Model Performance Summary (5-Fold CV):

# LR:
#   MAE  : 26.4408
#   MSE  : 6072.3719
#   MAPE : 6.1582
#   RMSE : 44.8958

# SVR:
#   MAE  : 8.5111
#   MSE  : 122.0304
#   MAPE : 11.8955
#   RMSE : 10.8328

# DT:
#   MAE  : 11.5205
#   MSE  : 215.9816
#   MAPE : 3.3991
#   RMSE : 14.6587

# RF:
#   MAE  : 8.9380
#   MSE  : 127.5099
#   MAPE : 11.5434
#   RMSE : 11.1263

# XGB:
#   MAE  : 9.7388
#   MSE  : 147.6857
#   MAPE : 6.3485
#   RMSE : 12.0130