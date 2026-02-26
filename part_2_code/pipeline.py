import matplotlib.pyplot as plt
import os
import json
import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score
)

from utils import (
    prepare_2d_input,
    prepare_sequence_input,
    compute_threshold,
    reconstruction_error
)

from models import (
    build_2d_cnn,
    build_1d_cnn,
    build_lstm_ae
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
        err_healthy = r["errors"][y_data == 1]
        err_loose   = r["errors"][y_data == 0]
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

def run_pipeline(X_data, y_data, config):

    results = {}

    X_2d = prepare_2d_input(X_data)
    X_seq = prepare_sequence_input(X_data)

    X_healthy_2d = X_2d[y_data == 1]
    X_healthy_seq = X_seq[y_data == 1]

    # 2D
    model_2d = build_2d_cnn(
        X_2d.shape[1:], config["cnn2d"]["filters"],
        config["training"]["optimizer"]
    )
    
    results["2D CNN"] = train_and_evaluate(
        model_2d, X_healthy_2d, X_2d, y_data, config
    )

    # 1D
    model_1d = build_1d_cnn(
        X_seq.shape[1:], config["cnn1d"]["filters"],
        config["training"]["optimizer"]
    )
    results["1D CNN"] = train_and_evaluate(
        model_1d, X_healthy_seq, X_seq, y_data, config
    )

    # LSTM
    model_lstm = build_lstm_ae(
        X_seq.shape[1:], config["lstm"]["latent_dim"],
        config["training"]["optimizer"]
    )
    results["LSTM"] = train_and_evaluate(
        model_lstm, X_healthy_seq, X_seq, y_data, config
    )

    plot_results(results, y_data, config["output"]["folder"])

    # After plotting
    save_models({"2D CNN": model_2d, "1D CNN": model_1d, "LSTM": model_lstm}, config["output"]["folder"])
    save_thresholds(results, config["output"]["folder"])

    return results