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
    model = train_one_fold_with_hm(model, Xb_train, Xh_train, y_train, Xb_test, Xh_test, y_test,
                           lr=0.005, max_epochs=500, early_stop_rounds=10)
    ## Predict the model
    y_pred = predict_with_hm(model, Xb_test, Xh_test)
    ## Evaluate the model
    metrics = evaluate_model(y_test, y_pred)
    print(f"Training samples: {len(Xb_train)}, Test samples: {len(Xb_test)}")
    # print(f"Handmade features - Training samples: {len(Xh_train)}, Test samples: {len(Xh_test)}")
    # The number of samples in the handmade features is the same as in the BERT features

    print(f"Fold {fold+1} Metrics: {metrics}")
    metrics["fold"] = fold + 1
    metrics_all_folds.append(metrics)
    print(f"Training samples: {len(Xb_train)}, Test samples: {len(Xb_test)}")

    ## Save model for this version
    model_path = os.path.join(model_folder, f"fold{fold+1}_model_{version}.pkl")
    save_model(model, model_path)

    ## Save predictions for this version
    predictions_path = os.path.join(result_folder, f"fold{fold+1}_predictions_{version}.csv")
    save_predictions(y_test, y_pred, predictions_path)

# Save metrics log (after all folds) for this version
save_metrics_log(metrics_all_folds, f"../results/model1_{version}/metrics_log_{version}.json")
