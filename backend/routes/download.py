import io
import joblib
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from services.ml_trainer import get_trained_session

router = APIRouter(tags=["Download"])

@router.get("/download/{model_name}")
async def download_model(model_name: str, x_session_id: str = Header(alias="X-Session-ID", default="default")):
    session = get_trained_session(x_session_id)
    if not session:
        raise HTTPException(status_code=400, detail="No trained models found for this session.")
        
    if model_name == "best":
        model_name = session["best_model"]
        if not model_name:
            raise HTTPException(status_code=400, detail="Best model not available.")
            
    if model_name not in session["results"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
        
    res = session["results"][model_name]
    if "error" in res or "model" not in res:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' failed to train or is not available.")
        
    model = res["model"]
    
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{model_name.replace(" ", "_")}.pkl"'}
    )

@router.get("/download/pipeline/{model_name}")
async def download_pipeline(model_name: str, x_session_id: str = Header(alias="X-Session-ID", default="default")):
    session = get_trained_session(x_session_id)
    if not session:
        raise HTTPException(status_code=400, detail="No trained models found for this session.")
        
    if model_name == "best":
        model_name = session["best_model"]
        if not model_name:
            raise HTTPException(status_code=400, detail="Best model not available.")
            
    if model_name not in session["results"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
        
    res = session["results"][model_name]
    if "error" in res or "model" not in res:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' failed to train or is not available.")
        
    pipeline = {
        "model": res["model"],
        "scaler": session.get("scaler"),
        "label_encoder": session.get("le"),
        "feature_names": session.get("feature_names")
    }
    
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{model_name.replace(" ", "_")}_pipeline.pkl"'}
    )
