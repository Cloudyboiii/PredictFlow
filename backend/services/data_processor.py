import io
import pandas as pd
import numpy as np
from config import get_settings

settings = get_settings()
_sessions: dict[str, dict] = {}


def ingest_csv(session_id: str, file_content: bytes, filename: str) -> dict:
    df = pd.read_csv(io.BytesIO(file_content))
    if df.empty:
        raise ValueError("CSV file is empty.")
    if len(df) > settings.MAX_ROWS:
        df = df.head(settings.MAX_ROWS)

    # Clean column names
    df.columns = [
        col.strip().replace(" ", "_").replace("-", "_").replace(".", "_").lower()
        for col in df.columns
    ]

    # Identify column types
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        unique_count = int(df[col].nunique())
        if "int" in dtype or "float" in dtype:
            col_type = "numeric"
        elif unique_count <= 20:
            col_type = "categorical"
        else:
            col_type = "text"
        columns.append({
            "name": col,
            "type": col_type,
            "dtype": dtype,
            "null_count": null_count,
            "unique_count": unique_count,
            "sample_values": df[col].dropna().head(3).tolist(),
        })

    _sessions[session_id] = {
        "df": df,
        "filename": filename,
        "columns": columns,
        "row_count": len(df),
    }

    return {
        "filename": filename,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }


def get_session(session_id: str):
    return _sessions.get(session_id)


def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False
