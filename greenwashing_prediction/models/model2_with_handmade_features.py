import os
import sys
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_X_y_from_dataset_with_handmade_features, train_one_fold_with_hm, evaluate_model, predict_with_hm, save_model, save_predictions, save_metrics_log, record_train_test_samples
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

## Define version
version = "v0"  # Adjust this for each new version of the model

# Create directories for the current version
model_folder = f"../models/model2/model2_{version}"
result_folder = f"../results/model2_{version}"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

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
    model, metrics = train_one_fold_with_hm(model, Xb_train, Xh_train, y_train, Xb_test, Xh_test, y_test,
                           lr=0.005, max_epochs=500, early_stop_rounds=10)
    
    ## Predict the model
    y_pred = predict_with_hm(model, Xb_test, Xh_test)
    ## Evaluate the model
    fold_metrics = evaluate_model(y_test, y_pred)
    metrics.update(fold_metrics)  # Update the metrics with the evaluation results

    # Record training and testing sample sizes
    metrics = record_train_test_samples(metrics, Xb_train, Xb_test, Xh_train, Xh_test)

    print(f"Training samples: {len(Xb_train)}, Test samples: {len(Xb_test)}")
    print(f"Fold {fold+1} Metrics: {metrics}")
    metrics["fold"] = fold + 1
    metrics_all_folds.append(metrics)

    # Save model for this version
    model_path = os.path.join(model_folder, f"fold{fold+1}_model_{version}.pkl")
    save_model(model, model_path)

    # Save predictions for this version
    predictions_path = os.path.join(result_folder, f"fold{fold+1}_predictions_{version}.csv")
    save_predictions(y_test, y_pred, predictions_path)

# Save metrics log (after all folds) for this version
save_metrics_log(metrics_all_folds, f"../results/model2_{version}/metrics_log_{version}.json")

# Early stopping at epoch 13
# Training samples: 26, Test samples: 26
# Fold 1 Metrics: {'MSE': 134.86513843540882, 'RMSE': 11.613145070798385, 'MAE': 9.90597926331009, 'MAPE': 1.205449236241226}
# Early stopping at epoch 19
# Training samples: 52, Test samples: 26
# Fold 2 Metrics: {'MSE': 187.381323056044, 'RMSE': 13.688729782417505, 'MAE': 10.378823789102679, 'MAPE': 2.068477367663991}
# Early stopping at epoch 21
# Training samples: 78, Test samples: 26
# Fold 3 Metrics: {'MSE': 119.14448503302083, 'RMSE': 10.91533256630419, 'MAE': 8.46781229650372, 'MAPE': 62.7178334554338}
# Early stopping at epoch 20
# Training samples: 104, Test samples: 26
# Fold 4 Metrics: {'MSE': 226.10166203033447, 'RMSE': 15.036677227045026, 'MAE': 11.830944416067705, 'MAPE': 1.2801884651916902}
# Early stopping at epoch 26
# Training samples: 130, Test samples: 26
# Fold 5 Metrics: {'MSE': 101.80840818298063, 'RMSE': 10.09001527169214, 'MAE': 7.778197445319591, 'MAPE': 2.321087381602918}