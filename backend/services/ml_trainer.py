import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import torch
import torch.nn as nn
from config import get_settings
import warnings
warnings.filterwarnings("ignore")

settings = get_settings()

# Store trained models per session
_trained_models: dict[str, dict] = {}


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes if num_classes > 2 else 1),
        )

    def forward(self, x):
        return self.net(x)


def preprocess(df: pd.DataFrame, target_col: str):
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    num_classes = len(le.classes_)

    # Separate numeric and categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute numeric
    if num_cols:
        num_imp = SimpleImputer(strategy="mean")
        X[num_cols] = num_imp.fit_transform(X[num_cols])

    # Encode + impute categorical
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("missing")
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_enc, le, scaler, num_classes, list(X.columns)


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    avg = "binary" if num_classes == 2 else "weighted"
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    try:
        if num_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    # ROC curve (binary only)
    roc_data = []
    if num_classes == 2:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            # Downsample to 50 points
            idx = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
            roc_data = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]
        except Exception:
            roc_data = []

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "roc_curve": roc_data,
    }


def train_all_models(session_id: str, df: pd.DataFrame, target_col: str) -> dict:
    X, y, le, scaler, num_classes, feature_names = preprocess(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y
    )

    results = {}

    # 1. Logistic Regression
    try:
        lr = LogisticRegression(max_iter=1000, random_state=settings.RANDOM_STATE)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)
        cv = cross_val_score(lr, X, y, cv=min(settings.CV_FOLDS, len(set(y))), scoring="accuracy")
        metrics = compute_metrics(y_test, y_pred, y_prob, num_classes)
        metrics["cv_mean"] = round(float(cv.mean()), 4)
        metrics["cv_std"] = round(float(cv.std()), 4)

        # Feature importance (coefficients for binary)
        if num_classes == 2 and hasattr(lr, "coef_"):
            fi = np.abs(lr.coef_[0])
        elif hasattr(lr, "coef_"):
            fi = np.abs(lr.coef_).mean(axis=0)
        else:
            fi = np.zeros(len(feature_names))
        fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)[:10]

        results["Logistic Regression"] = {
            "metrics": metrics,
            "model": lr,
            "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in fi_pairs],
        }
    except Exception as e:
        results["Logistic Regression"] = {"error": str(e)}

    # 2. Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=settings.RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)
        cv = cross_val_score(rf, X, y, cv=min(settings.CV_FOLDS, len(set(y))), scoring="accuracy")
        metrics = compute_metrics(y_test, y_pred, y_prob, num_classes)
        metrics["cv_mean"] = round(float(cv.mean()), 4)
        metrics["cv_std"] = round(float(cv.std()), 4)

        fi = rf.feature_importances_
        fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)[:10]

        results["Random Forest"] = {
            "metrics": metrics,
            "model": rf,
            "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in fi_pairs],
        }
    except Exception as e:
        results["Random Forest"] = {"error": str(e)}

    # 3. XGBoost
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=settings.RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        y_prob = xgb_model.predict_proba(X_test)
        cv = cross_val_score(xgb_model, X, y, cv=min(settings.CV_FOLDS, len(set(y))), scoring="accuracy")
        metrics = compute_metrics(y_test, y_pred, y_prob, num_classes)
        metrics["cv_mean"] = round(float(cv.mean()), 4)
        metrics["cv_std"] = round(float(cv.std()), 4)

        fi = xgb_model.feature_importances_
        fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)[:10]

        results["XGBoost"] = {
            "metrics": metrics,
            "model": xgb_model,
            "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in fi_pairs],
        }
    except Exception as e:
        results["XGBoost"] = {"error": str(e)}

    # 4. Neural Network (PyTorch)
    try:
        input_dim = X_train.shape[1]
        X_tr_t = torch.FloatTensor(X_train)
        y_tr_t = torch.LongTensor(y_train) if num_classes > 2 else torch.FloatTensor(y_train)
        X_te_t = torch.FloatTensor(X_test)

        model = SimpleNN(input_dim, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            out = model(X_tr_t)
            if num_classes == 2:
                loss = criterion(out.squeeze(), y_tr_t)
            else:
                loss = criterion(out, y_tr_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_te_t)
            if num_classes == 2:
                probs_pos = torch.sigmoid(logits.squeeze()).numpy()
                y_prob_nn = np.column_stack([1 - probs_pos, probs_pos])
                y_pred_nn = (probs_pos >= 0.5).astype(int)
            else:
                y_prob_nn = torch.softmax(logits, dim=1).numpy()
                y_pred_nn = np.argmax(y_prob_nn, axis=1)

        metrics = compute_metrics(y_test, y_pred_nn, y_prob_nn, num_classes)
        metrics["cv_mean"] = None
        metrics["cv_std"] = None

        results["Neural Network"] = {
            "metrics": metrics,
            "model": model,
            "feature_importance": [],
        }
    except Exception as e:
        results["Neural Network"] = {"error": str(e)}

    # Find best model by ROC-AUC
    best_model_name = None
    best_score = -1
    for name, res in results.items():
        if "error" not in res:
            score = res["metrics"].get("roc_auc", 0)
            if score > best_score:
                best_score = score
                best_model_name = name

    # Store everything
    _trained_models[session_id] = {
        "results": results,
        "best_model": best_model_name,
        "scaler": scaler,
        "le": le,
        "num_classes": num_classes,
        "feature_names": feature_names,
        "target_col": target_col,
        "label_classes": le.classes_.tolist(),
    }

    # Build response (strip model objects)
    response_models = []
    for name, res in results.items():
        if "error" in res:
            response_models.append({"name": name, "error": res["error"]})
        else:
            response_models.append({
                "name": name,
                "metrics": res["metrics"],
                "feature_importance": res.get("feature_importance", []),
                "is_best": name == best_model_name,
            })

    return {
        "models": response_models,
        "best_model": best_model_name,
        "target_col": target_col,
        "label_classes": le.classes_.tolist(),
        "num_classes": num_classes,
        "feature_names": feature_names,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }


def get_trained_session(session_id: str):
    return _trained_models.get(session_id)
