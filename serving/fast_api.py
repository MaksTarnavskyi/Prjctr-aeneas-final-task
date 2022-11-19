from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from aeneas.predictor import AeneasPredictor


class Payload(BaseModel):
    texts: List[str]


class Prediction(BaseModel):
    output: List[str]


app = FastAPI()
predictor = AeneasPredictor(transformer_name = 'distilroberta-base', model_dir = 'models/distilroberta-base')


@app.get("/health_check")
def health_check() -> str:
    return "ok"


@app.post("/predict", response_model=Prediction)
def predict(payload: Payload) -> Prediction:
    prediction = predictor.predict_for_online(input_texts=payload.texts)
    return Prediction(probs=prediction.tolist())
