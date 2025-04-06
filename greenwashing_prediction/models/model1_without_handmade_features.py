import os
import sys
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_X_y_from_dataset_without_handmade_features, train_one_fold, evaluate_model, predict, save_model, save_predictions, save_metrics_log
from models.mlp import build_mlp1_model

## Read and check data
csr_embeddings_score = pd.read_csv("../data/csr_embeddings_score.csv")
csr_embeddings_score.dropna(inplace=True)
csr_embeddings_score = csr_embeddings_score[csr_embeddings_score["greenwash_score"] != 0]

## Sort the data by year
csr_embeddings_score.sort_values(
    by=["year1"],
    ascending=True,
    inplace=True,
)
X, y = get_X_y_from_dataset_without_handmade_features(csr_embeddings_score)
## Check the shape of the data
# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}")

## Time Series K-Fold
tscv = TimeSeriesSplit(n_splits=5)
metrics_all_folds = []
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ## Build the model
    model = build_mlp1_model()   
    ## Train the model
    model = train_one_fold(model, X_train, y_train, X_test, y_test, lr=0.005, max_epochs=500, early_stop_rounds=10)
    ## Predict the model
    y_pred = predict(model, X_test)
    ## Evaluate the model
    metrics = evaluate_model(y_test, y_pred)
    # print(f"Fold {fold+1} Metrics: {metrics}")
    metrics["fold"] = fold + 1
    metrics_all_folds.append(metrics)
    ## Print the metrics
    print(f"Metrics: {metrics}")
    ## Save the model

        # Save model
#     save_model(model, f"../models/DNN_model1/fold{fold+1}_model.pkl")

#     # Save predictions
#     save_predictions(y_test, y_pred, f"../results/DNN_model1/fold{fold+1}_predictions.csv")

# # Save metrics log (after all folds)
# save_metrics_log(metrics_all_folds, "../results/DNN_model1/metrics_log.json")

# Early stopping at epoch 22
# Metrics: {'MSE': 253.17392560138035, 'RMSE': 15.911440085717583, 'MAE': 11.946565503480114, 'MAPE': 4.649468814641202, 'fold': 1}
# Early stopping at epoch 18
# Metrics: {'MSE': 237.55195562254144, 'RMSE': 15.412720578228278, 'MAE': 11.381758628320124, 'MAPE': 5.238755290973708, 'fold': 2}
# Early stopping at epoch 22
# Metrics: {'MSE': 223.11486633941763, 'RMSE': 14.9370300374411, 'MAE': 11.042352206132934, 'MAPE': 4.54240239615341, 'fold': 3}
# Early stopping at epoch 24
# Metrics: {'MSE': 238.83837390079728, 'RMSE': 15.454396588052129, 'MAE': 11.128429970971146, 'MAPE': 2.805332797283217, 'fold': 4}
# Early stopping at epoch 31
# Metrics: {'MSE': 222.67165184184634, 'RMSE': 14.922186563699247, 'MAE': 10.904291063073458, 'MAPE': 3.6309857519490154, 'fold': 5}