import traceback
import numpy as np
import pandas as pd
import io
import torch
from fastapi import APIRouter, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.ml_trainer import get_trained_session

router = APIRouter(tags=["Predict"])


class PredictRequest(BaseModel):
    features: dict
    model_name: str = "best"


@router.post("/predict")
async def predict(
    req: PredictRequest,
    x_session_id: str = Header(alias="X-Session-ID", default="default"),
):
    trained = get_trained_session(x_session_id)
    if not trained:
        raise HTTPException(status_code=400, detail="No trained model found. Please train first.")

    model_name = trained["best_model"] if req.model_name == "best" else req.model_name
    results = trained["results"]

    if model_name not in results or "error" in results[model_name]:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available.")

    model = results[model_name]["model"]
    scaler = trained["scaler"]
    le = trained["le"]
    feature_names = trained["feature_names"]
    num_classes = trained["num_classes"]
    task_type = trained.get("task_type", "classification")

    try:
        # Build feature vector
        row_dict = {f: req.features.get(f, 0) for f in feature_names}
        df = pd.DataFrame([row_dict])
        from services.ml_trainer import preprocess_inference
        X_scaled = preprocess_inference(df, trained)

        if model_name == "Neural Network":
            import torch
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_scaled))
                if task_type == "regression":
                    pred = float(logits.squeeze().item())
                    probs = []
                elif num_classes == 2:
                    prob = float(torch.sigmoid(logits.squeeze()).item())
                    probs = [1 - prob, prob]
                    pred_idx = int(prob >= 0.5)
                else:
                    probs_t = torch.softmax(logits, dim=1).numpy()[0]
                    probs = probs_t.tolist()
                    pred_idx = int(np.argmax(probs))
        else:
            if task_type == "regression":
                pred = float(model.predict(X_scaled)[0])
                probs = []
            else:
                pred_idx = int(model.predict(X_scaled)[0])
                probs = model.predict_proba(X_scaled)[0].tolist()

        if task_type == "regression":
            return {
                "prediction": round(pred, 4),
                "confidence": None,
                "probabilities": None,
                "model_used": model_name,
            }
        else:
            pred_label = le.inverse_transform([pred_idx])[0]
            confidence = round(max(probs) * 100, 1)
            return {
                "prediction": str(pred_label),
                "confidence": confidence,
                "probabilities": {
                    str(le.inverse_transform([i])[0]): round(p * 100, 1)
                    for i, p in enumerate(probs)
                },
                "model_used": model_name,
            }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    model_name: str = Form("best"),
    x_session_id: str = Header(alias="X-Session-ID", default="default"),
):
    trained = get_trained_session(x_session_id)
    if not trained:
        raise HTTPException(status_code=400, detail="No trained model found. Please train first.")

    target_model_name = trained["best_model"] if model_name == "best" else model_name
    results = trained["results"]

    if target_model_name not in results or "error" in results[target_model_name]:
        raise HTTPException(status_code=400, detail=f"Model '{target_model_name}' not available.")

    model = results[target_model_name]["model"]
    scaler = trained["scaler"]
    le = trained["le"]
    feature_names = trained["feature_names"]
    num_classes = trained["num_classes"]
    task_type = trained.get("task_type", "classification")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        from services.ml_trainer import preprocess_inference
        X_scaled = preprocess_inference(df, trained)
        
        if target_model_name == "Neural Network":
            import torch
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_scaled))
                if task_type == "regression":
                    preds = logits.squeeze().numpy()
                    if preds.ndim == 0:
                        preds = np.array([preds])
                    confidences = [None] * len(preds)
                elif num_classes == 2:
                    probs = torch.sigmoid(logits.squeeze()).numpy()
                    if probs.ndim == 0:
                        probs = np.array([probs])
                    preds = (probs >= 0.5).astype(int)
                    confidences = np.maximum(probs, 1 - probs) * 100
                else:
                    probs = torch.softmax(logits, dim=1).numpy()
                    preds = np.argmax(probs, axis=1)
                    confidences = np.max(probs, axis=1) * 100
        else:
            preds = model.predict(X_scaled)
            if task_type != "regression":
                probs = model.predict_proba(X_scaled)
                confidences = np.max(probs, axis=1) * 100
            else:
                confidences = [None] * len(preds)

        if task_type != "regression" and le is not None:
            pred_labels = le.inverse_transform(preds.astype(int))
            df["prediction"] = pred_labels
            df["confidence_pct"] = np.round(confidences, 1)
        else:
            df["prediction"] = preds
            df["confidence_pct"] = ""
            
        csv_data = df.to_csv(index=False)
        output = io.BytesIO(csv_data.encode('utf-8'))
        
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="predictions.csv"'}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
