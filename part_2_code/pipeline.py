import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve,average_precision_score, accuracy_score
from models import build_1d_cnn_autoencoder, build_lstm_autoencoder, determine_ae_threshold_max_division
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import prepare_2d_input,prepare_sequence_input,compute_threshold,reconstruction_error
from tensorflow.keras.models import save_model
import joblib




def save_models(models_dict, output_folder):
    model_folder = os.path.join(output_folder, "models")
    os.makedirs(model_folder, exist_ok=True)
    for name, model in models_dict.items():
        #path = os.path.join(model_folder, name.replace(" ", "_"))
        file_path = os.path.join(model_folder, f"{name}.keras")
        model.save(file_path)
        print(f"Saved model {name} to {file_path}")

def save_thresholds(thresholds_dict, folder_path):
    """
    Save thresholds or any metrics to a JSON file.
    Automatically converts all NumPy/TensorFlow types and nested structures to Python types.
    """
    os.makedirs(folder_path, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.ndarray, list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    thresholds_serializable = make_serializable(thresholds_dict)

    file_path = os.path.join(folder_path, "thresholds.json")
    with open(file_path, "w") as f:
        json.dump(thresholds_serializable, f, indent=4)

    print(f"Thresholds saved to {file_path}")

def train_and_evaluate(model, X_train, X_all, y_all, config):

    model.fit(
        X_train, X_train,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        validation_split=config["training"]["validation_split"],
        verbose=0
    )

    errors = reconstruction_error(model, X_all)

    threshold = compute_threshold(
        errors[y_all == 1],
        config["threshold"]
    )

    fpr, tpr, _ = roc_curve(y_all, errors)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_all, errors)
    pr_auc = average_precision_score(y_all, errors)

    return {
        "errors": errors,
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc
    }


def save_model_plots(results, y_train=None, y_val=None, output_folder="outputs"):
    """
    Save histograms per model, combined ROC and PR curves for validation,
    and print validation accuracy for each model.
    """
    os.makedirs(output_folder, exist_ok=True)

    # ---------------------------
    # Histograms (one per model)
    # ---------------------------
    for name, r in results.items():
        # Determine if this is classifier (probabilities in [0,1]) or autoencoder
        max_val = np.max(r["errors_val"])
        if max_val <= 1.0:  # classifier
            bins = np.linspace(0,1,50)
        else:  # autoencoder
            bins = 30

        # Training histogram
        if y_train is not None:
            plt.figure(figsize=(10,6))
            plt.hist(r["errors_train"][y_train==0], bins=bins, alpha=0.5, label="Healthy")
            plt.hist(r["errors_train"][y_train==1], bins=bins, alpha=0.5, label="Faulty")
            plt.axvline(r.get("threshold", 0.5), color='red', linestyle='--', label="Threshold")
            plt.title(f"{name} Training Error / Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{name}_train_hist.png"), dpi=300)
            plt.close()

        # Validation histogram
        if y_val is not None:
            plt.figure(figsize=(10,6))
            plt.hist(r["errors_val"][y_val==0], bins=bins, alpha=0.5, label="Healthy")
            plt.hist(r["errors_val"][y_val==1], bins=bins, alpha=0.5, label="Faulty")
            plt.axvline(r.get("threshold", 0.5), color='red', linestyle='--', label="Threshold")
            plt.title(f"{name} Validation Error / Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{name}_validation_hist.png"), dpi=300)
            plt.close()

            # Compute and print validation accuracy
            if max_val <= 1.0:  # classifier
                # Threshold = 0.5 by default for classifier
                preds = (r["errors_val"] >= 0.5).astype(int)
            else:  # autoencoder
                preds = (r["errors_val"] > r.get("threshold", 0.5)).astype(int)
            val_acc = accuracy_score(y_val, preds)
            print(f"{name} Validation Accuracy: {val_acc:.3f}")

    # ---------------------------
    # Combined ROC Curves
    # ---------------------------
    if y_val is not None:
        plt.figure(figsize=(10,8))
        for name, r in results.items():
            plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")
        plt.plot([0,1],[0,1],'k--', label="Random")
        plt.title("Validation ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "combined_validation_roc.png"), dpi=300)
        plt.close()

        # Combined Precision-Recall Curves
        plt.figure(figsize=(10,8))
        for name, r in results.items():
            plt.plot(r["recall"], r["precision"], label=f"{name} (AP={r['pr_auc']:.3f})")
        plt.title("Validation Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "combined_validation_pr.png"), dpi=300)
        plt.close()
# -----------------------------
# Pipeline function
# -----------------------------

def run_pipeline(X_autoenc, X_ML, y, config):
    """
    Train models and evaluate on validation set only.
    Saves histograms + ROC + PR curves for TRAIN and VALIDATION.
    Also saves all models to the output folder from config.
    """

    results = {}

    # Ensure output folder exists
    output_folder = config["output"]["folder"]
    os.makedirs(output_folder, exist_ok=True)

    # Normalize per feature across dataset
    mean_ae = X_autoenc.mean(axis=(0,1), keepdims=True)
    std_ae  = X_autoenc.std(axis=(0,1), keepdims=True) + 1e-8
    X_autoenc = (X_autoenc - mean_ae) / std_ae

    # --------------------------------------------------
    # Train / Validation split
    # --------------------------------------------------
    val_split = config["dataset"].get("val_split", 0.2)

    X_autoenc_train, X_autoenc_val, X_ML_train, X_ML_val, y_train, y_val = train_test_split(
        X_autoenc,
        X_ML,
        y,
        test_size=val_split,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Normalize ML features
    # --------------------------------------------------
    scaler = StandardScaler()
    X_ML_train = scaler.fit_transform(X_ML_train)
    X_ML_val = scaler.transform(X_ML_val)

    # ==================================================
    # 1️⃣ 1D CNN Autoencoder
    # ==================================================
    healthy_idx = y_train == 0

    model_1d = build_1d_cnn_autoencoder(
        X_autoenc_train.shape[1:],
        optimizer=config["training"]["optimizer"]
    )

    model_1d.fit(
        X_autoenc_train[healthy_idx],
        X_autoenc_train[healthy_idx],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        verbose=1
    )

    # Train reconstruction
    X_pred_train = model_1d.predict(X_autoenc_train)
    errors_train = np.mean((X_pred_train - X_autoenc_train)**2, axis=(1,2))

    # Validation reconstruction
    X_pred_val = model_1d.predict(X_autoenc_val)
    errors_val = np.mean((X_pred_val - X_autoenc_val)**2, axis=(1,2))

    threshold = threshold = determine_ae_threshold_max_division(errors_train, y_train)#np.percentile(errors_train[y_train==1], 95)

    fpr, tpr, _ = roc_curve(y_val, errors_val)
    precision, recall, _ = precision_recall_curve(y_val, errors_val)

    results["1D_CNN_AE"] = {
        "errors_train": errors_train,
        "errors_val": errors_val,
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc(fpr, tpr),
        "pr_auc": auc(recall, precision),
        "model": model_1d
    }

    # Save 1D CNN model
    model_1d_path = os.path.join(output_folder, "1D_CNN_AE.keras")
    save_model(model_1d, model_1d_path)
    # Save threshold together with model
    threshold_path = os.path.join(output_folder, "1D_CNN_AE_threshold.npz")
    np.savez(threshold_path, threshold=threshold)

    # ==================================================
    # 2️⃣ LSTM Autoencoder
    # ==================================================
    model_lstm = build_lstm_autoencoder(
        X_autoenc_train.shape[1:],
        optimizer=config["training"]["optimizer"]
    )

    model_lstm.fit(
        X_autoenc_train[healthy_idx],
        X_autoenc_train[healthy_idx],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        verbose=1
    )

    X_pred_train = model_lstm.predict(X_autoenc_train)
    errors_train = np.mean((X_pred_train - X_autoenc_train)**2, axis=(1,2))

    X_pred_val = model_lstm.predict(X_autoenc_val)
    errors_val = np.mean((X_pred_val - X_autoenc_val)**2, axis=(1,2))

    threshold = threshold = determine_ae_threshold_max_division(errors_train, y_train)#np.percentile(errors_train[y_train==1], 95)

    fpr, tpr, _ = roc_curve(y_val, errors_val)
    precision, recall, _ = precision_recall_curve(y_val, errors_val)

    results["LSTM_AE"] = {
        "errors_train": errors_train,
        "errors_val": errors_val,
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc(fpr, tpr),
        "pr_auc": auc(recall, precision),
        "model": model_lstm
    }

    # Save LSTM model
    model_lstm_path = os.path.join(output_folder, "LSTM_AE.keras")
    save_model(model_lstm, model_lstm_path)
    threshold_path = os.path.join(output_folder, "LSTM_AE_threshold.npz")
    np.savez(threshold_path, threshold=threshold)

    # ==================================================
    # 3️⃣ Classical ML Models
    # ==================================================
    ml_models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500)
    }

    for name, clf in ml_models.items():

        clf.fit(X_ML_train, y_train)

        probs_train = clf.predict_proba(X_ML_train)[:, 1]
        probs_val = clf.predict_proba(X_ML_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, probs_val)
        precision, recall, _ = precision_recall_curve(y_val, probs_val)

        results[name] = {
            "errors_train": probs_train,
            "errors_val": probs_val,
            "threshold": 0.5,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "roc_auc": auc(fpr, tpr),
            "pr_auc": auc(recall, precision),
            "model": clf
        }

        # Save classical ML model
        model_path = os.path.join(output_folder, f"{name}.pkl")
        joblib.dump(clf, model_path)

    # --------------------------------------------------
    # Save plots (TRAIN + VALIDATION only)
    # --------------------------------------------------
    save_model_plots(
        results,
        y_train=y_train,
        y_val=y_val,
        output_folder=output_folder
    )

    return results