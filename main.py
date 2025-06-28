from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()
model = pickle.load(open("budget_model.pkl", "rb"))

class BudgetInput(BaseModel):
    savings: float

@app.post("/predict")
def predict(data: BudgetInput):
    prediction = model.predict([[data.savings]])[0]
    suggested_budget = round(prediction, 2)

    recommendations = [
        f"Try to keep your total monthly expenses under RM {suggested_budget}.",
        "Save at least 20% of your savings if possible.",
        "Cut down on non-essential expenses like shopping or entertainment."
    ]

    return {
        "predicted_expense": suggested_budget,
        "recommendations": recommendations
    }

