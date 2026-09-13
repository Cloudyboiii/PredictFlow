import traceback
from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from config import get_settings
from services.data_processor import ingest_csv, delete_session

router = APIRouter(tags=["Upload"])
settings = get_settings()


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    x_session_id: str = Header(alias="X-Session-ID", default="default"),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit.")

    try:
        result = ingest_csv(x_session_id, content, file.filename)
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/dataset")
async def delete_dataset(
    x_session_id: str = Header(alias="X-Session-ID", default="default"),
):
    delete_session(x_session_id)
    return {"status": "deleted"}
