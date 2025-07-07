#Importatoin des modules
from fastapi import FastAPI , Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from pydantic import BaseModel


app = FastAPI()
templates = Jinja2Templates(directory="template")


model = joblib.load("logistic_model.pkl")
# schéma pour les données d'entrée
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Route de prédiction
@app.post("/predict")
def predict(data: PatientData):
    # Convertion en DataFrame pour le modèle
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}
