from pydantic import BaseModel


class PredictionRequest(BaseModel):
    data: str


class PredictionResponse(BaseModel):
    prediction: str