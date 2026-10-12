import json
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    app.state.metadata = None
    directory = Path(os.environ.get('MODEL_DIR', 'models'))
    try:
        # Only load locally trained or otherwise trusted joblib artifacts.
        app.state.model = joblib.load(directory / 'titanic_pipeline.joblib')
        app.state.metadata = json.loads((directory / 'model_metadata.json').read_text())
    except (OSError, ValueError) as exc:
        app.state.model = None
        print(f'Model unavailable: {exc}')
    yield

app = FastAPI(title='Titanic Survival Prediction API', lifespan=lifespan)

class PassengerInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    Pclass: Literal[1, 2, 3]
    Sex: Literal['male', 'female']
    Age: float = Field(ge=0, le=120, allow_inf_nan=False)
    SibSp: int = Field(ge=0, strict=True)
    Parch: int = Field(ge=0, strict=True)
    Fare: float = Field(ge=0, allow_inf_nan=False)
    Embarked: Literal['S', 'C', 'Q']

def predict_one(passenger):
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail='Train the model before serving predictions.')
    frame = pd.DataFrame([passenger.model_dump()])
    probability = model.predict_proba(frame)[0, list(model.classes_).index(1)]
    return {'survived': int(model.predict(frame)[0]), 'survival_probability': float(probability)}

@app.get('/')
def root():
    return {'message': 'Titanic Survival Prediction API', 'model_loaded': app.state.model is not None}

@app.get('/health')
def health():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail='Model unavailable')
    return {'status': 'ready'}

@app.get('/model-info')
def model_info():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail='Model unavailable')
    return app.state.metadata

@app.post('/predict')
def predict(passenger: PassengerInput):
    return predict_one(passenger)

@app.post('/predict-batch')
def predict_batch(passengers: list[PassengerInput] = Body(min_length=1, max_length=1000)):
    results = [predict_one(p) for p in passengers]
    return {'predictions': results, 'count': len(results)}
