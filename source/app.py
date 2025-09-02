from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Literal
import joblib
import pandas as pd
import numpy as np

# ---- App + lifecycle ---------------------------------------------------------

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load your trained pipeline ONCE at startup
    try:
        loaded = joblib.load("gb_tuned.joblib")   # this should be your full pipeline
        models["phone_addiction_model"] = loaded
        print("Model loaded: gb_tuned.joblib")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    print("App shutting down...")

app = FastAPI(title="Teen Phone Addiction Predictor", lifespan=lifespan)

# (Optional) allow local dev frontends to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request schema ----------------------------------------------------------

class PredictionRequest(BaseModel):
    Age: int = Field(..., ge=0, lt=120)
    Daily_Usage_Hours: float = Field(..., ge=0)
    Sleep_Hours: float = Field(..., ge=0)
    Academic_Performance: int = Field(..., ge=0, le=100)
    Social_Interactions: int = Field(..., ge=0, le=10)
    Exercise_Hours: float = Field(..., ge=0)
    Anxiety_Level: int = Field(..., ge=0, le=10)
    Depression_Level: int = Field(..., ge=0, le=10)
    Self_Esteem: int = Field(..., ge=0, le=10)
    Phone_Checks_Per_Day: int = Field(..., ge=0)
    Apps_Used_Daily: int = Field(..., ge=0)
    Time_on_Social_Media: float = Field(..., ge=0)
    Time_on_Gaming: float = Field(..., ge=0)
    Time_on_Education: float = Field(..., ge=0)
    Family_Communication: int = Field(..., ge=0, le=10)
    Weekend_Usage_Hours: float = Field(..., ge=0)

    # Categorical features — keep values aligned with what the pipeline was trained on
    Gender: Literal["Male", "Female", "Other"]
    Phone_Usage_Purpose: Literal["Browsing", "Education", "Social Media", "Gaming", "Other"]
    School_Grade: Literal["7th", "8th", "9th", "10th", "11th", "12th"]

# ---- Routes ------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Teen Phone Addiction Predictor is up. See /docs for Swagger UI."}

@app.post("/predict")
async def predict(payload: PredictionRequest):
    model = models.get("phone_addiction_model")
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"status_code": 500, "message": "Model not loaded. Check server logs."}
        )

    # Convert incoming payload to the exact column structure the pipeline expects
    # (Your saved pipeline should include its preprocessing, so raw columns are fine.)
    data = pd.DataFrame([payload.model_dump()])

    try:
        y_pred = model.predict(data)
        # Not all classifiers expose predict_proba; guard for that.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)[0]
            classes = getattr(model, "classes_", np.arange(len(proba)))
            probabilities = [
                {"class": str(classes[i]), "probability": float(proba[i])}
                for i in range(len(proba))
            ]
        else:
            probabilities = None

        return {
            "status_code": 200,
            "predicted_class": str(y_pred[0]),
            "probabilities": probabilities
        }

    except Exception as e:
        # Helpful error if columns/mappings differ from training
        return JSONResponse(
            status_code=400,
            content={
                "status_code": 400,
                "message": f"Inference failed: {e}",
                "hint": "Ensure categorical values and column names match the training pipeline."
            }
        )

# ---- Local dev entrypoint ----------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
