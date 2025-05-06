from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.utils import load_model

app = FastAPI()
model = load_model('models/gbc_model.pkl')

class Applicant(BaseModel):
    age: float
    income: float
    credit_amount: float

@app.post("/predict")
def predict(applicant: Applicant):
    df = pd.DataFrame([applicant.dict()])
    score = model.predict_proba(df)[0,1]
    decision = "approve" if score < 0.5 else "decline"
    return {"score": score, "decision": decision}
