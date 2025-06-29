from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle

app = FastAPI()
model = pickle.load(open("budget_model.pkl", "rb"))

class ExpenseItem(BaseModel):
    category: str
    amount: float

class BudgetInput(BaseModel):
    savings: float
    expenses: List[ExpenseItem]

@app.post("/predict")
def predict(data: BudgetInput):
    prediction = model.predict([[data.savings]])[0]
    suggested_budget = round(prediction, 2)

    # Basic tips
    base_tips = [
        f"Try to keep your total monthly expenses under RM {suggested_budget}.",
        "Save at least 20% of your savings if possible.",
        "Cut down on non-essential expenses like shopping or entertainment."
    ]

    # 💡 Analyze category spending
    category_totals = {}
    for exp in data.expenses:
        category_totals[exp.category] = category_totals.get(exp.category, 0) + exp.amount

    personalized = []
    for category, amount in category_totals.items():
        percent = amount / data.savings if data.savings > 0 else 0
        if percent > 0.4:
            personalized.append(
                f"You're spending over 40% of your savings on {category.lower()}. Try cutting back."
            )
        elif percent < 0.05:
            personalized.append(
                f"Nice job keeping {category.lower()} costs low!"
            )

    return {
        "predicted_expense": suggested_budget,
        "recommendations": base_tips + personalized
    }
