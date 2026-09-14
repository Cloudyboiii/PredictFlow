import traceback
import pandas as pd
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from services.data_processor import get_session
from services.ml_trainer import train_all_models
from services.ml_trainer_regression import train_regression_models

router = APIRouter(tags=["Train"])


class TrainRequest(BaseModel):
    target_column: str
    feature_columns: list[str] | None = None


@router.post("/train")
async def train(req: TrainRequest, x_session_id: str = Header(alias="X-Session-ID", default="default")):
    try:
        session = get_session(x_session_id)
        if not session:
            raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a CSV first.")
        df = session["df"]
        if req.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.target_column}' not found.")
        target_vals = df[req.target_column].nunique()
        
        is_numeric = pd.api.types.is_numeric_dtype(df[req.target_column])
        if target_vals > 20 and is_numeric:
            task_type = "regression"
        else:
            if target_vals < 2:
                raise HTTPException(status_code=400, detail="Target column must have at least 2 unique values.")
            if target_vals > 20:
                raise HTTPException(status_code=400, detail="Target column has too many unique values (>20).")
            task_type = "classification"
            
        if task_type == "regression":
            result = train_regression_models(x_session_id, df, req.target_column, req.feature_columns)
        else:
            result = train_all_models(x_session_id, df, req.target_column, req.feature_columns)
            
        result["task_type"] = task_type
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
