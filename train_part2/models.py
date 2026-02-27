import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# -------------------------
# ML Model Trainer
# -------------------------
class SimpleMLModels:
    def __init__(self, model_type="RandomForest", random_state=42):
        """
        model_type: str, one of ["RandomForest", "GradientBoosting", "SVM", "Logistic"]
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self):
        if self.model_type == "RandomForest":
            self.model = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        elif self.model_type == "GradientBoosting":
            self.model = GradientBoostingClassifier(n_estimators=200, random_state=self.random_state)
        elif self.model_type == "SVM":
            self.model = SVC(probability=True, kernel="rbf", random_state=self.random_state)
        elif self.model_type == "Logistic":
            self.model = LogisticRegression(max_iter=500, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X, y, test_size=0.2):
        """
        Fit the model with train/validation split
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, 
                                                          stratify=y, random_state=self.random_state)

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Build and fit model
        self.build_model()
        self.model.fit(X_train, y_train)

        # Predictions & metrics
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, "predict_proba") else None
        acc = accuracy_score(y_val, y_pred)
        roc = roc_auc_score(y_val, y_prob) if y_prob is not None else None

        print(f"Validation Accuracy: {acc:.3f}")
        if roc is not None:
            print(f"Validation ROC-AUC: {roc:.3f}")
        print(classification_report(y_val, y_pred))

        return acc, roc

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            # For SVM without probability, fallback to decision function
            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(X_scaled)
                return 1 / (1 + np.exp(-scores))  # sigmoid to convert to 0-1
            else:
                raise ValueError("Model does not support probability prediction.")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "model_type": self.model_type}, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.model_type = data["model_type"]
        print(f"Model loaded from {path}")