#Importatoin des modules
from fastapi import FastAPI , Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from pydantic import BaseModel


app = FastAPI()
templates = Jinja2Templates(directory="template")
#==============================BASIC======================================
@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    context = {
        "request" : request,
        "Nom": "N'DIAYE",
        "Prenom": "Salimata"
    }
    return templates.TemplateResponse('home.html', context)

produits = {
    "1" :"Sac",
    "2" : "Veste"
}


#==========================================================================
@app.get('/produit/{id_produit}/')
def voir_produits(id_produit: str):
    context = {
        'produit': produits.get(id_produit, "Produit non trouvé")
    }
    return context


#===========================================================================


model = joblib.load("logistic_model.pkl")
# Définir un schéma pour les données d'entrée
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
    # Convertir en DataFrame pour le modèle
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}
