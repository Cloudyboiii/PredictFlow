# PredictFlow ⚡

**Automated ML Model Training & Evaluation Pipeline**

Upload a CSV, select a target column, and PredictFlow automatically trains 4 ML models simultaneously, evaluates each with comprehensive metrics, and serves the best model for real-time inference.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)

---

## What is PredictFlow?

PredictFlow is a full-stack AutoML platform that demonstrates the complete ML lifecycle — from raw data to production-ready model. Users upload any CSV dataset, select a classification target, and the system automatically:

1. Preprocesses the data (imputation, encoding, scaling)
2. Trains 4 ML models in parallel
3. Evaluates each with accuracy, precision, recall, F1, ROC-AUC, and cross-validation
4. Generates ROC curves, confusion matrices, and feature importance charts
5. Serves the best model via a live prediction API

---

## Models Trained

| Model | Library | Key Strength |
|---|---|---|
| **Logistic Regression** | Scikit-learn | Fast, interpretable baseline |
| **Random Forest** | Scikit-learn | Handles non-linearity, feature importance |
| **XGBoost** | XGBoost | High accuracy gradient boosting |
| **Neural Network** | PyTorch | Deep learning (2-layer MLP) |

---

## Features

- **4 Models in Parallel** — all trained simultaneously, results compared side by side
- **Automated Preprocessing** — missing value imputation, categorical encoding, feature scaling
- **Comprehensive Metrics** — accuracy, precision, recall, F1, ROC-AUC, cross-validation (mean ± std)
- **Model Comparison Dashboard** — interactive bar charts comparing all models on all metrics
- **ROC Curves** — binary classification ROC curves with AUC score
- **Confusion Matrix** — visual confusion matrix per model
- **Feature Importance** — top 10 features for Logistic Regression, Random Forest, XGBoost
- **Live Prediction** — enter feature values and get real-time predictions from any trained model
- **Best Model Selection** — auto-selects highest ROC-AUC model
- **Session Isolation** — each browser gets its own trained models via UUID

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML | Scikit-learn, XGBoost, PyTorch | Model training and evaluation |
| Backend | Python + FastAPI | API, preprocessing, model serving |
| Frontend | Next.js 14 + React + Tailwind CSS | Interactive ML dashboard |
| Charts | Recharts | ROC curves, confusion matrices, feature importance |
| Hosting | Render + Vercel | Free tier deployment |

---

## Quick Start

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 10000
```

### Frontend

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:10000" > .env.local
npm run dev
```

---

## Deployment

| Component | Platform | Configuration |
|---|---|---|
| Backend | [Render](https://render.com) — Free | Root: `backend`, Runtime: Docker |
| Frontend | [Vercel](https://vercel.com) — Free | Root: `frontend`, Env: `NEXT_PUBLIC_API_URL` |

---

Built by **[Badal Gupta](https://github.com/Cloudyboiii)** — MS Data Science, University at Albany (SUNY)
