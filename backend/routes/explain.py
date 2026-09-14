import traceback
import numpy as np
import shap
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from services.ml_trainer import get_trained_session
from services.data_processor import get_session

router = APIRouter(tags=["Explain"])

class ExplainRequest(BaseModel):
    model_name: str = "best"
    row_index: int = -1
    features: dict = None

@router.post("/explain")
async def explain_prediction(
    req: ExplainRequest,
    x_session_id: str = Header(alias="X-Session-ID", default="default")
):
    trained = get_trained_session(x_session_id)
    if not trained:
        raise HTTPException(status_code=400, detail="No trained model found. Please train first.")
        
    session = get_session(x_session_id)
    if not session:
        raise HTTPException(status_code=400, detail="No dataset loaded in session.")
        
    df = session["df"]
    
    target_model_name = trained["best_model"] if req.model_name == "best" else req.model_name
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
        # Prepare the single row to explain
        if req.features is not None:
            row_dict = {f: req.features.get(f, 0) for f in feature_names}
            row_df = pd.DataFrame([row_dict])
        else:
            if req.row_index < 0 or req.row_index >= len(df):
                req.row_index = 0
            row_df = df.iloc[[req.row_index]]
        from services.ml_trainer import preprocess_inference
        X_scaled = preprocess_inference(row_df, trained)
        
        # We need background data. We sample up to 50 rows from the dataset.
        bg_df = df.sample(min(50, len(df)), random_state=42)
        X_bg = preprocess_inference(bg_df, trained)
        X_bg_scaled = X_bg

        # Initialize Explainer
        if target_model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values_obj = explainer(X_scaled)
            shap_values = shap_values_obj.values[0]
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
                
        elif target_model_name == "Logistic Regression" or target_model_name == "Linear Regression":
            explainer = shap.LinearExplainer(model, X_bg_scaled)
            shap_values_obj = explainer(X_scaled)
            shap_values = shap_values_obj.values[0]
            base_value = explainer.expected_value
            
        elif target_model_name == "Neural Network":
            import torch
            
            def model_predict(X_numpy):
                model.eval()
                with torch.no_grad():
                    logits = model(torch.FloatTensor(X_numpy))
                    if task_type == "regression":
                        return logits.squeeze().numpy()
                    elif num_classes == 2:
                        return torch.sigmoid(logits.squeeze()).numpy()
                    else:
                        return torch.softmax(logits, dim=1).numpy()[:, 0]

            explainer = shap.KernelExplainer(model_predict, shap.kmeans(X_bg_scaled, 10))
            shap_values_obj = explainer.shap_values(X_scaled)
            if isinstance(shap_values_obj, list):
                shap_values = shap_values_obj[0][0]
                base_value = explainer.expected_value[0]
            else:
                shap_values = shap_values_obj[0]
                if isinstance(explainer.expected_value, np.ndarray):
                    base_value = explainer.expected_value[0]
                else:
                    base_value = explainer.expected_value
        else:
            raise ValueError(f"Explainer not supported for {target_model_name}")

        if len(np.shape(shap_values)) > 1:
            shap_values = shap_values[:, 0]
            
        feature_contributions = []
        for i, f in enumerate(feature_names):
            val = float(shap_values[i])
            feature_contributions.append({
                "feature": f,
                "shap_value": val,
                "feature_value": float(X_row[0][i]),
                "direction": "positive" if val > 0 else "negative"
            })
            
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_10 = feature_contributions[:10]
        
        if target_model_name == "Neural Network":
            import torch
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_scaled))
                if task_type == "regression":
                    pred = float(logits.squeeze().item())
                elif num_classes == 2:
                    prob = float(torch.sigmoid(logits.squeeze()).item())
                    pred_idx = int(prob >= 0.5)
                else:
                    probs_t = torch.softmax(logits, dim=1).numpy()[0]
                    pred_idx = int(np.argmax(probs_t))
        else:
            if task_type == "regression":
                pred = float(model.predict(X_scaled)[0])
            else:
                pred_idx = int(model.predict(X_scaled)[0])

        if task_type == "regression":
            prediction_label = str(round(pred, 4))
            predicted_value = pred
        else:
            prediction_label = str(le.inverse_transform([pred_idx])[0])
            predicted_value = float(pred_idx)
            
        if isinstance(base_value, np.ndarray) and len(base_value) > 1:
            base_value = base_value[0]
            
        return {
            "feature_contributions": top_10,
            "base_value": float(base_value) if base_value is not None else 0.0,
            "predicted_value": predicted_value,
            "prediction_label": prediction_label
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
