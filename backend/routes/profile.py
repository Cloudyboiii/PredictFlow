import pandas as pd
import numpy as np
import traceback
from fastapi import APIRouter, Header, HTTPException
from services.data_processor import get_session

router = APIRouter(tags=["Profile"])

def clean_float(val):
    if pd.isna(val) or np.isinf(val):
        return None
    return float(val)

@router.post("/profile")
async def profile_dataset(x_session_id: str = Header(alias="X-Session-ID", default="default")):
    session = get_session(x_session_id)
    if not session:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a CSV first.")
    
    try:
        df = session["df"]
        total_rows = len(df)
        
        columns_profile = []
        class_balance = []
        health_score = 100
        has_severe_nulls = False
        
        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            null_percent = round((null_count / total_rows) * 100, 2)
            unique_count = int(series.nunique(dropna=True))
            
            if null_percent > 50:
                has_severe_nulls = True
            elif null_percent > 5:
                health_score -= 10
                
            is_num = pd.api.types.is_numeric_dtype(series)
            
            if 2 <= unique_count <= 20:
                counts = series.value_counts(dropna=True)
                classes = [{"label": str(k), "count": int(v), "percent": round((int(v)/total_rows)*100, 2)} for k, v in counts.items()]
                class_balance.append({"column_name": col, "classes": classes})
                
            if is_num:
                s_clean = series.dropna()
                if len(s_clean) > 0:
                    min_val = clean_float(s_clean.min())
                    max_val = clean_float(s_clean.max())
                    mean_val = clean_float(s_clean.mean())
                    median_val = clean_float(s_clean.median())
                    std_dev = clean_float(s_clean.std())
                    skewness = clean_float(s_clean.skew())
                    
                    if skewness is not None and abs(skewness) > 2.0:
                        health_score -= 5
                        
                    # Outliers (>3 std dev)
                    outliers = 0
                    if std_dev is not None and mean_val is not None and std_dev > 0:
                        outliers = int(((s_clean - mean_val).abs() > 3 * std_dev).sum())
                        
                    # Histogram
                    hist_data = []
                    if max_val is not None and min_val is not None and max_val > min_val:
                        counts, bins = np.histogram(s_clean, bins=10)
                        for i in range(len(counts)):
                            label = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
                            hist_data.append({"label": label, "count": int(counts[i])})
                            
                    columns_profile.append({
                        "name": col,
                        "col_type": "numeric",
                        "null_count": null_count,
                        "null_percent": null_percent,
                        "unique_count": unique_count,
                        "min": min_val,
                        "max": max_val,
                        "mean": mean_val,
                        "median": median_val,
                        "std_dev": std_dev,
                        "skewness": skewness,
                        "outlier_count": outliers,
                        "histogram": hist_data
                    })
                else:
                    columns_profile.append({
                        "name": col,
                        "col_type": "numeric",
                        "null_count": null_count,
                        "null_percent": null_percent,
                        "unique_count": 0
                    })
            else:
                top_vals = series.value_counts(dropna=True).head(5)
                top_list = [{"value": str(k), "count": int(v)} for k, v in top_vals.items()]
                columns_profile.append({
                    "name": col,
                    "col_type": "categorical",
                    "null_count": null_count,
                    "null_percent": null_percent,
                    "unique_count": unique_count,
                    "top_values": top_list
                })
                
        if has_severe_nulls:
            health_score -= 15
            
        health_score = max(0, health_score)
        
        # Correlation matrix
        correlation_matrix = []
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty and num_df.shape[1] > 1:
            corr = num_df.corr()
            cols = corr.columns
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    val = corr.iloc[i, j]
                    if not pd.isna(val) and abs(val) > 0.3:
                        correlation_matrix.append({
                            "col1": cols[i],
                            "col2": cols[j],
                            "coefficient": clean_float(val)
                        })
        
        # Sort correlations by absolute magnitude
        correlation_matrix.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        
        return {
            "columns": columns_profile,
            "class_balance": class_balance,
            "correlation_matrix": correlation_matrix,
            "overall_health_score": health_score,
            "row_count": total_rows,
            "col_count": len(df.columns)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")
