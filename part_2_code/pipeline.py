import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def save_models(models_dict, output_folder):
    model_folder = os.path.join(output_folder, "models")
    os.makedirs(model_folder, exist_ok=True)
    for name, model in models_dict.items():
        ext = ".pkl"
        file_path = os.path.join(model_folder, f"{name}{ext}")
        joblib.dump(model, file_path)
        print(f"Saved model {name} to {file_path}")


def save_model_plots(results, y_train=None, y_val=None, output_folder="outputs"):
    os.makedirs(output_folder, exist_ok=True)

    # ---------------------------
    # Histograms per model
    # ---------------------------
    for name, r in results.items():
        # Training histogram
        if y_train is not None:
            plt.figure(figsize=(10, 6))
            plt.hist(r["scores_train"][y_train == 0], bins=30, alpha=0.5, label="Healthy")
            plt.hist(r["scores_train"][y_train == 1], bins=30, alpha=0.5, label="Faulty")
            plt.axvline(r.get("threshold", 0.5), color='red', linestyle='--', label="Threshold")
            plt.title(f"{name} Training Scores")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{name}_train_hist.png"), dpi=300)
            plt.close()

        # Validation histogram
        if y_val is not None:
            plt.figure(figsize=(10, 6))
            plt.hist(r["scores_val"][y_val == 0], bins=30, alpha=0.5, label="Healthy")
            plt.hist(r["scores_val"][y_val == 1], bins=30, alpha=0.5, label="Faulty")
            plt.axvline(r.get("threshold", 0.5), color='red', linestyle='--', label="Threshold")
            plt.title(f"{name} Validation Scores")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{name}_val_hist.png"), dpi=300)
            plt.close()

            preds = (r["scores_val"] >= r.get("threshold", 0.5)).astype(int)
            val_acc = accuracy_score(y_val, preds)
            print(f"{name} Validation Accuracy: {val_acc:.3f}")

    # ---------------------------
    # Combined ROC Curves
    # ---------------------------
    if y_val is not None:
        plt.figure(figsize=(10, 8))
        for name, r in results.items():
            plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.title("Validation ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "combined_validation_roc.png"), dpi=300)
        plt.close()

        # Combined Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        for name, r in results.items():
            plt.plot(r["recall"], r["precision"], label=f"{name} (AP={r['pr_auc']:.3f})")
        plt.title("Validation Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "combined_validation_pr.png"), dpi=300)
        plt.close()


def run_pipeline(X, y, config):
    """
    Train, validate, and compare simple ML models.
    Saves histograms + ROC + PR curves and all trained models.
    """
    results = {}
    output_folder = config["output"]["folder"]
    os.makedirs(output_folder, exist_ok=True)

    # ---------------------------
    # Train/Validation split
    # ---------------------------
    val_split = config["dataset"].get("val_split", 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    model_path = os.path.join(output_folder, f"StandardScaler.pkl")
    joblib.dump(scaler, model_path)
    print(f"Saved model StandardScaler to {model_path}")

    # ---------------------------
    # Define models
    # ---------------------------
    ml_models_config = config.get("ml_models", {})
    ml_models = {}

    for name, params in ml_models_config.items():
        if name == "RandomForest":
            clf = RandomForestClassifier(**params)
        elif name == "GradientBoosting":
            clf = GradientBoostingClassifier(**params)
        elif name == "SVM":
            clf = SVC(**params)
        elif name == "LogisticRegression":
            clf = LogisticRegression(**params)
        else:
            print(f"Warning: Unknown model {name}, skipping.")
            continue
        ml_models[name] = clf

    for name, clf in ml_models.items():
        clf.fit(X_train_scaled, y_train)
        scores_train = clf.predict_proba(X_train_scaled)[:, 1]
        scores_val = clf.predict_proba(X_val_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, scores_val)
        precision, recall, _ = precision_recall_curve(y_val, scores_val)

        results[name] = {
            "scores_train": scores_train,
            "scores_val": scores_val,
            "threshold": 0.5,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "roc_auc": auc(fpr, tpr),
            "pr_auc": average_precision_score(y_val, scores_val),
            "model": clf
        }

        # Save model
        model_path = os.path.join(output_folder, f"{name}.pkl")
        joblib.dump(clf, model_path)
        print(f"Saved model {name} to {model_path}")

    # ---------------------------
    # Save plots
    # ---------------------------
    save_model_plots(results, y_train=y_train, y_val=y_val, output_folder=output_folder)

    return results