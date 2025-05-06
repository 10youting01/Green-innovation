import os
import sys
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_X_y_from_dataset_without_handmade_features, train_one_fold, evaluate_model, predict, save_model, save_predictions, save_metrics_log, record_train_test_samples
from models.mlp import build_mlp_model

## Define version
version = "v0"  # Adjust this for each new version of the model

# Create directories for the current version
model_folder = f"../models/model1/model1_{version}"
result_folder = f"../results/model1_{version}"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

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

## Time Series K-Fold
tscv = TimeSeriesSplit(n_splits=5)
metrics_all_folds = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ## Build the model
    model = build_mlp_model("mlp1")   
    ## Train the model
    model, metrics = train_one_fold(model, X_train, y_train, X_test, y_test, lr=0.005, max_epochs=500, early_stop_rounds=10)
    
    ## Predict the model
    y_pred = predict(model, X_test)
    
    ## Evaluate the model
    fold_metrics = evaluate_model(y_test, y_pred)
    metrics.update(fold_metrics)  # Update the metrics with the evaluation results

    # Record training and testing sample sizes
    metrics = record_train_test_samples(metrics, X_train, X_test)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
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
save_metrics_log(metrics_all_folds, f"../results/model1_{version}/metrics_log_{version}.json")


# Early stopping at epoch 21
# Fold 1 Metrics: {'MSE': 242.69813545673793, 'RMSE': 15.578771949570926, 'MAE': 11.710377014265582, 'MAPE': 4.390193185648197}
# Training samples: 823, Test samples: 823
# Early stopping at epoch 23
# Fold 2 Metrics: {'MSE': 256.0272353143279, 'RMSE': 16.00085108093716, 'MAE': 11.685437623310131, 'MAPE': 6.375378498762015}
# Training samples: 1646, Test samples: 823
# Early stopping at epoch 19
# Fold 3 Metrics: {'MSE': 217.2693649877037, 'RMSE': 14.740059870560353, 'MAE': 10.92326680849885, 'MAPE': 4.057159371210535}
# Training samples: 2469, Test samples: 823
# Early stopping at epoch 25
# Fold 4 Metrics: {'MSE': 243.21495943496464, 'RMSE': 15.595350571082545, 'MAE': 11.193176979844317, 'MAPE': 2.9576815121995814}
# Training samples: 3292, Test samples: 823
# Early stopping at epoch 18
# Fold 5 Metrics: {'MSE': 229.7444763057485, 'RMSE': 15.157324180268379, 'MAE': 11.074404233412936, 'MAPE': 4.053625243786816}
# Training samples: 4115, Test samples: 823