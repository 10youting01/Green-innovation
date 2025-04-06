import os
import sys
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_X_y_from_dataset_with_handmade_features, train_one_fold_with_hm, evaluate_model, predict_with_hm, save_model, save_predictions, save_metrics_log
from models.mlp import build_mlp_model

## Read and check data
csr_embeddings_hm_score = pd.read_csv("../data/CSR_embeddings_handmade_score.csv")
csr_embeddings_hm_score.dropna(inplace=True)
csr_embeddings_hm_score = csr_embeddings_hm_score[csr_embeddings_hm_score["greenwash_score"] != 0]

## Sort the data by year
csr_embeddings_hm_score.sort_values(
    by=["year1"],
    ascending=True,
    inplace=True,
)
X_bert, X_hand, y = get_X_y_from_dataset_with_handmade_features(csr_embeddings_hm_score)
## Check the shape of the data
# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}")

## Time Series K-Fold
tscv = TimeSeriesSplit(n_splits=5)
metrics_all_folds = []
for fold, (train_index, test_index) in enumerate(tscv.split(X_bert)):
    Xb_train, Xb_test = X_bert[train_index], X_bert[test_index]
    Xh_train, Xh_test = X_hand[train_index], X_hand[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ## Build the model
    model = build_mlp_model("mlp2")   
    ## Train the model
    model = train_one_fold_with_hm(model, Xb_train, Xh_train, y_train, Xb_test, Xh_test, y_test,
                           lr=0.005, max_epochs=500, early_stop_rounds=10)
    ## Predict the model
    y_pred = predict_with_hm(model, Xb_test, Xh_test)
    ## Evaluate the model
    metrics = evaluate_model(y_test, y_pred)
    print(f"Fold {fold+1} Metrics: {metrics}")
    metrics["fold"] = fold + 1
    metrics_all_folds.append(metrics)
    ## Save the model
    metrics = evaluate_model(y_test, y_pred)
    metrics["fold"] = fold + 1
    metrics_all_folds.append(metrics)

#     # 儲存
#     save_model(model, f"../models/DNN_model2_with_handmade/fold{fold+1}_model.pt")
#     save_predictions(y_test, y_pred, f"../results/DNN_model2_with_handmade/fold{fold+1}_predictions.csv")

# # 儲存所有 fold 的結果
# save_metrics_log(metrics_all_folds, "../results/DNN_model2_with_handmade/metrics_log.json")

# Early stopping at epoch 29
# Fold 1 Metrics: {'MSE': 115.37635110883261, 'RMSE': 10.741338422600444, 'MAE': 8.687929257539649, 'MAPE': 1.4719267942177092}
# Early stopping at epoch 14
# Fold 2 Metrics: {'MSE': 362.689008612925, 'RMSE': 19.044395727166695, 'MAE': 13.230844565772047, 'MAPE': 3.467415263118048}
# Early stopping at epoch 30
# Fold 3 Metrics: {'MSE': 108.56603245361033, 'RMSE': 10.419502505091609, 'MAE': 7.651665720015634, 'MAPE': 55.78205597335014}
# Early stopping at epoch 22
# Fold 4 Metrics: {'MSE': 218.2030878153413, 'RMSE': 14.77169888047212, 'MAE': 11.213510902189988, 'MAPE': 2.030771857413181}
# Early stopping at epoch 14
# Fold 5 Metrics: {'MSE': 399.02509285374714, 'RMSE': 19.97561245253189, 'MAE': 16.262410826806267, 'MAPE': 5.559103025472601}