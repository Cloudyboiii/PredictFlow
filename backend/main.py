import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routes.upload import router as upload_router
from routes.train import router as train_router
from routes.predict import router as predict_router
from routes.health import router as health_router

app = FastAPI(title="PredictFlow API", version="1.0.0")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://predict-flow.vercel.app",
    "https://predict-flow.vercel.app/",
]

frontend_url = os.getenv("FRONTEND_URL", "")
if frontend_url and frontend_url not in origins:
    origins.append(frontend_url)
    origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(train_router, prefix="/api")
app.include_router(predict_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
