import traceback
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from services.data_processor import get_session
from services.ml_trainer import train_all_models

router = APIRouter(tags=["Train"])


class TrainRequest(BaseModel):
    target_column: str


@router.post("/train")
async def train(
    req: TrainRequest,
    x_session_id: str = Header(alias="X-Session-ID", default="default"),
):
    session = get_session(x_session_id)
    if not session:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a CSV first.")

    df = session["df"]
    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_column}' not found in dataset.")

    target_vals = df[req.target_column].nunique()
    if target_vals < 2:
        raise HTTPException(status_code=400, detail="Target column must have at least 2 unique values.")
    if target_vals > 20:
        raise HTTPException(status_code=400, detail="Target column has too many unique values (>20). Please choose a classification target.")

    try:
        result = train_all_models(x_session_id, df, req.target_column)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
