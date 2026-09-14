import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import torch
import torch.nn as nn
from config import get_settings
import warnings
warnings.filterwarnings("ignore")

from services.ml_trainer import clean_metrics, clean_float, _trained_models

settings = get_settings()

class SimpleNNReg(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

def preprocess_regression(df: pd.DataFrame, target_col: str):
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)
    
    if y.isnull().any():
        y = y.fillna(y.mean())

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_cols:
        num_imp = SimpleImputer(strategy="mean")
        X[num_cols] = num_imp.fit_transform(X[num_cols])

    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("missing")
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler, list(X.columns)

def compute_regression_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    
    # Residual plot data (downsample to 100 max)
    limit = min(100, len(y_true))
    idx = np.linspace(0, len(y_true) - 1, limit, dtype=int)
    y_true_samp = np.array(y_true)[idx]
    y_pred_samp = np.array(y_pred)[idx]
    
    residual_data = [{"actual": float(a), "predicted": float(p)} for a, p in zip(y_true_samp, y_pred_samp)]
    
    return clean_metrics({
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "mape": round(mape, 4),
        "residual_plot": residual_data
    })

def train_regression_models(session_id: str, df: pd.DataFrame, target_col: str, feature_columns: list[str] | None = None) -> dict:
    if feature_columns:
        df = df[feature_columns + [target_col]]
        
    X, y, scaler, feature_names = preprocess_regression(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    results = {}

    # 1. Linear Regression
    try:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        cv = cross_val_score(lr, X, y, cv=settings.CV_FOLDS, scoring="r2")
        metrics = compute_regression_metrics(y_test, y_pred)
        metrics["cv_mean"] = clean_float(round(float(cv.mean()), 4))
        metrics["cv_std"] = clean_float(round(float(cv.std()), 4))

        if hasattr(lr, "coef_"):
            fi = np.abs(lr.coef_)
        else:
            fi = np.zeros(len(feature_names))
        fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)[:10]

        results["Linear Regression"] = {
            "metrics": metrics,
            "model": lr,
            "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in fi_pairs],
        }
    except Exception as e:
        results["Linear Regression"] = {"error": str(e)}

    # 2. Random Forest Regressor
    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=settings.RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        cv = cross_val_score(rf, X, y, cv=settings.CV_FOLDS, scoring="r2")
        metrics = compute_regression_metrics(y_test, y_pred)
        metrics["cv_mean"] = clean_float(round(float(cv.mean()), 4))
        metrics["cv_std"] = clean_float(round(float(cv.std()), 4))

        fi = rf.feature_importances_
        fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)[:10]

        results["Random Forest"] = {
            "metrics": metrics,
            "model": rf,
            "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in fi_pairs],
        }
    except Exception as e:
        results["Random Forest"] = {"error": str(e)}

    # 3. XGBoost Regressor
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            random_state=settings.RANDOM_STATE,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        cv = cross_val_score(xgb_model, X, y, cv=settings.CV_FOLDS, scoring="r2")
        metrics = compute_regression_metrics(y_test, y_pred)
        metrics["cv_mean"] = clean_float(round(float(cv.mean()), 4))
        metrics["cv_std"] = clean_float(round(float(cv.std()), 4))

        fi = xgb_model.feature_importances_
        fi_pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)[:10]

        results["XGBoost"] = {
            "metrics": metrics,
            "model": xgb_model,
            "feature_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in fi_pairs],
        }
    except Exception as e:
        results["XGBoost"] = {"error": str(e)}

    # 4. Neural Network Regressor
    try:
        input_dim = X_train.shape[1]
        X_tr_t = torch.FloatTensor(X_train)
        y_tr_t = torch.FloatTensor(y_train).view(-1, 1)
        X_te_t = torch.FloatTensor(X_test)

        model = SimpleNNReg(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            out = model(X_tr_t)
            loss = criterion(out, y_tr_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_nn = model(X_te_t).numpy().flatten()

        metrics = compute_regression_metrics(y_test, y_pred_nn)
        metrics["cv_mean"] = None
        metrics["cv_std"] = None

        results["Neural Network"] = {
            "metrics": metrics,
            "model": model,
            "feature_importance": [],
        }
    except Exception as e:
        results["Neural Network"] = {"error": str(e)}

    best_model_name = None
    best_score = -float('inf')
    for name, res in results.items():
        if "error" not in res:
            score = res["metrics"].get("r2_score", -float('inf'))
            if score > best_score:
                best_score = score
                best_model_name = name

    _trained_models[session_id] = {
        "results": results,
        "best_model": best_model_name,
        "scaler": scaler,
        "le": None,
        "num_classes": 0,
        "feature_names": feature_names,
        "target_col": target_col,
        "label_classes": [],
        "task_type": "regression"
    }

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
        "label_classes": [],
        "num_classes": 0,
        "feature_names": feature_names,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "task_type": "regression"
    }
