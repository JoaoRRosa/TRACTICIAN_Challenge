import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve,average_precision_score
from models import build_1d_cnn_autoencoder, build_lstm_autoencoder
import json


from utils import (
    prepare_2d_input,
    prepare_sequence_input,
    compute_threshold,
    reconstruction_error
)




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


def plot_results(results, y_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # -------------------------
    # 1️⃣ Plot individual reconstruction error histograms per model
    # -------------------------
    for name, r in results.items():
        err_healthy = r["errors"][y_data == 0]
        err_loose   = r["errors"][y_data == 1]
        threshold   = r["threshold"]

        plt.figure(figsize=(10,6))
        plt.hist(err_healthy, bins=30, alpha=0.5, label=f'Healthy {name}')
        plt.hist(err_loose, bins=30, alpha=0.5, label=f'Loose {name}')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold {name}')
        plt.title(f"{name} Reconstruction Error")
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(output_folder, f"{name}_reconstruction_error.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    # -------------------------
    # 2️⃣ ROC Curve (all models together)
    # -------------------------
    plt.figure(figsize=(8,6))
    for name, r in results.items():
        plt.plot(r["fpr"], r["tpr"], label=f"{name} AUC={r['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall / True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "roc_curve.png"), dpi=300)
    plt.close()

    # -------------------------
    # 3️⃣ Precision-Recall Curve (all models together)
    # -------------------------
    plt.figure(figsize=(8,6))
    for name, r in results.items():
        plt.plot(r["recall"], r["precision"], label=f"{name} AP={r['pr_auc']:.3f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "precision_recall_curve.png"), dpi=300)
    plt.close()

    print(f"All plots saved in {output_folder}")


# -----------------------------
# Pipeline function
# -----------------------------
def run_pipeline(X_autoenc, X_ML, y, config):
    output_folder = config["output"]["folder"]
    os.makedirs(output_folder, exist_ok=True)
    results = {}

    healthy_idx = y==0

    # Normalize per feature across dataset
    mean_ae = X_autoenc.mean(axis=(0,1), keepdims=True)
    std_ae  = X_autoenc.std(axis=(0,1), keepdims=True) + 1e-8
    X_autoenc = (X_autoenc - mean_ae) / std_ae

    # ---------------------------
    # 1D CNN Autoencoder
    # ---------------------------
    model_1d = build_1d_cnn_autoencoder(
        X_autoenc.shape[1:], filters=config["cnn1d"]["filters"],
        optimizer=config["training"]["optimizer"]
    )
    model_1d.fit(X_autoenc[healthy_idx], X_autoenc[healthy_idx],
                 epochs=config["training"]["epochs"],
                 batch_size=config["training"]["batch_size"],
                 verbose=1)
    
    X_pred = model_1d.predict(X_autoenc)
    errors = np.mean((X_pred - X_autoenc)**2, axis=(1,2))
    threshold = np.percentile(errors[healthy_idx], 95)
    fpr, tpr, _ = roc_curve(y, errors)
    precision, recall, _ = precision_recall_curve(y, errors)
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)

    results["1D CNN"] = {"errors": errors, "threshold": threshold, 
                         "fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
                         "pr_auc": pr_auc, "roc_auc": roc_auc,
                         "model": model_1d}

    # ---------------------------
    # LSTM Autoencoder
    # ---------------------------
    model_lstm = build_lstm_autoencoder(
        X_autoenc.shape[1:], latent_dim=config["lstm"]["latent_dim"],
        optimizer=config["training"]["optimizer"]
    )
    #model_lstm.fit(X_autoenc[healthy_idx], X_autoenc[healthy_idx],
                   #epochs=config["training"]["epochs"],
                   #batch_size=config["training"]["batch_size"],
                   #verbose=1)
    #X_pred = model_lstm.predict(X_autoenc)
    #errors = np.mean((X_pred - X_autoenc)**2, axis=(1,2))
    #threshold = np.percentile(errors[healthy_idx], 95)
    #fpr, tpr, _ = roc_curve(y, errors)
    #precision, recall, _ = precision_recall_curve(y, errors)
    #pr_auc = auc(recall, precision)
    #roc_auc = auc(fpr, tpr)

    #results["LSTM"] = {"errors": errors, "threshold": threshold, 
                       #"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
                       #"pr_auc": pr_auc, "roc_auc": roc_auc,
                       #"model": model_lstm}

    # ---------------------------
    # Classical ML models
    # ---------------------------
    ml_models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500)
    }
    for name, clf in ml_models.items():
        clf.fit(X_ML, y)
        probs = clf.predict_proba(X_ML)[:,1]
        fpr, tpr, _ = roc_curve(y, probs)
        precision, recall, _ = precision_recall_curve(y, probs)
        pr_auc = auc(recall, precision)
        roc_auc = auc(fpr, tpr)
        threshold = 0.5
        results[name] = {"errors": probs, "threshold": threshold,
                         "fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
                         "pr_auc": pr_auc, "roc_auc": roc_auc,
                         "model": clf}

    # ---------------------------
    # Plot results
    # ---------------------------
    for name, r in results.items():
        plt.figure(figsize=(10,6))
        plt.hist(r["errors"][y==0], bins=30, alpha=0.5, label='Healthy')
        plt.hist(r["errors"][y==1], bins=30, alpha=0.5, label='Loose')
        plt.axvline(r["threshold"], color='red', linestyle='--')
        plt.title(f"{name} Reconstruction Error / Probabilities")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{name}_errors.png"), dpi=300)
        plt.close()

    # Combined ROC + PR curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for name, r in results.items():
        plt.plot(r["fpr"], r["tpr"], label=f"{name} AUC={r['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR / Recall"); plt.title("ROC Curve"); plt.legend()
    plt.subplot(1,2,2)
    for name, r in results.items():
        plt.plot(r["recall"], r["precision"], label=f"{name} AP={r['pr_auc']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "roc_pr_curves.png"), dpi=300)
    plt.close()

    return results