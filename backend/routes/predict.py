import traceback
import numpy as np
import torch
from fastapi import APIRouter, Header, HTTPException
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

    try:
        # Build feature vector
        row = [req.features.get(f, 0) for f in feature_names]
        X = np.array([row], dtype=float)
        X_scaled = scaler.transform(X)

        if model_name == "Neural Network":
            import torch
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_scaled))
                if num_classes == 2:
                    prob = float(torch.sigmoid(logits.squeeze()).item())
                    probs = [1 - prob, prob]
                    pred_idx = int(prob >= 0.5)
                else:
                    probs_t = torch.softmax(logits, dim=1).numpy()[0]
                    probs = probs_t.tolist()
                    pred_idx = int(np.argmax(probs))
        else:
            pred_idx = int(model.predict(X_scaled)[0])
            probs = model.predict_proba(X_scaled)[0].tolist()

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
