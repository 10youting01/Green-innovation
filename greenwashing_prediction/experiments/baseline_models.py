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
fold_sizes = []  # 新增的變數，用來儲存每折的訓練集和測試集大小

# 執行每折
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\n Fold {fold + 1}")
    
    # 訓練集與測試集
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 記錄訓練集與測試集的大小
    train_size = len(train_idx)
    test_size = len(test_idx)
    fold_sizes.append((train_size, test_size))  # 儲存每折的訓練集與測試集大小
    
    print(f"Train size: {train_size}, Test size: {test_size}")

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

        print(f"MSE: {metrics['MSE']: .2f}, RMSE: {metrics['RMSE']:.2f}, {name} => MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}")

# 輸出平均表現
print("\n Baseline Model Performance Summary (5-Fold CV):")
for name, scores in metrics_summary.items():
    print(f"\n{name}:")
    print(f"  MSE  : {np.mean(scores['MSE']):.4f}")
    print(f"  RMSE : {np.mean(scores['RMSE']):.4f}")
    print(f"  MAE  : {np.mean(scores['MAE']):.4f}")
    print(f"  MAPE : {np.mean(scores['MAPE']):.4f}")

# 輸出每個fold的訓練集與測試集大小
print("\n Fold-wise Train and Test Sizes:")
for fold_num, (train_size, test_size) in enumerate(fold_sizes, 1):
    print(f"  Fold {fold_num}: Train size = {train_size}, Test size = {test_size}")


#  Fold 1
# Train size: 29, Test size: 27
# MSE:  29652.59, RMSE: 172.20, LR => MAE: 90.97, MAPE: 11.40
# MSE:  109.38, RMSE: 10.46, SVR => MAE: 8.02, MAPE: 1.48
# MSE:  170.37, RMSE: 13.05, DT => MAE: 10.39, MAPE: 1.66
# MSE:  131.33, RMSE: 11.46, RF => MAE: 9.11, MAPE: 1.54
# MSE:  147.66, RMSE: 12.15, XGB => MAE: 9.60, MAPE: 1.56

#  Fold 2
# Train size: 56, Test size: 27
# MSE:  297.64, RMSE: 17.25, LR => MAE: 13.15, MAPE: 4.91
# MSE:  198.75, RMSE: 14.10, SVR => MAE: 11.54, MAPE: 4.34
# MSE:  256.86, RMSE: 16.03, DT => MAE: 12.65, MAPE: 4.73
# MSE:  203.95, RMSE: 14.28, RF => MAE: 11.44, MAPE: 4.21
# MSE:  213.19, RMSE: 14.60, XGB => MAE: 11.34, MAPE: 4.15

#  Fold 3
# Train size: 83, Test size: 27
# MSE:  164.40, RMSE: 12.82, LR => MAE: 9.83, MAPE: 10.11
# MSE:  148.16, RMSE: 12.17, SVR => MAE: 9.62, MAPE: 50.41
# MSE:  241.13, RMSE: 15.53, DT => MAE: 11.38, MAPE: 7.00
# MSE:  135.17, RMSE: 11.63, RF => MAE: 9.40, MAPE: 48.27
# MSE:  172.99, RMSE: 13.15, XGB => MAE: 10.63, MAPE: 22.75

#  Fold 4
# Train size: 110, Test size: 27
# MSE:  136.63, RMSE: 11.69, LR => MAE: 9.86, MAPE: 3.05
# MSE:  93.75, RMSE: 9.68, SVR => MAE: 7.58, MAPE: 2.32
# MSE:  212.74, RMSE: 14.59, DT => MAE: 11.91, MAPE: 2.00
# MSE:  90.70, RMSE: 9.52, RF => MAE: 8.07, MAPE: 2.55
# MSE:  119.01, RMSE: 10.91, XGB => MAE: 9.56, MAPE: 2.21

#  Fold 5
# Train size: 137, Test size: 27
# MSE:  110.60, RMSE: 10.52, LR => MAE: 8.39, MAPE: 1.32
# MSE:  60.11, RMSE: 7.75, SVR => MAE: 5.80, MAPE: 0.93
# MSE:  198.81, RMSE: 14.10, DT => MAE: 11.26, MAPE: 1.61
# MSE:  76.40, RMSE: 8.74, RF => MAE: 6.67, MAPE: 1.15
# MSE:  85.57, RMSE: 9.25, XGB => MAE: 7.56, MAPE: 1.07

#  Baseline Model Performance Summary (5-Fold CV):

# LR:
#   MSE  : 6072.3719
#   RMSE : 44.8958
#   MAE  : 26.4408
#   MAPE : 6.1582

# SVR:
#   MSE  : 122.0304
#   RMSE : 10.8328
#   MAE  : 8.5111
#   MAPE : 11.8955

# DT:
#   MSE  : 215.9816
#   RMSE : 14.6587
#   MAE  : 11.5205
#   MAPE : 3.3991

# RF:
#   MSE  : 127.5099
#   RMSE : 11.1263
#   MAE  : 8.9380
#   MAPE : 11.5434

# XGB:
#   MSE  : 147.6857
#   RMSE : 12.0130
#   MAE  : 9.7388
#   MAPE : 6.3485

#  Fold-wise Train and Test Sizes:
#   Fold 1: Train size = 29, Test size = 27
#   Fold 2: Train size = 56, Test size = 27
#   Fold 3: Train size = 83, Test size = 27
#   Fold 4: Train size = 110, Test size = 27
#   Fold 5: Train size = 137, Test size = 27