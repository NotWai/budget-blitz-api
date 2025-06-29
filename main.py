from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
import pandas as pd
import pickle

app = FastAPI()
model = pickle.load(open("budget_model.pkl", "rb"))

class ExpenseItem(BaseModel):
    category: str
    amount: float
    date: str  # ISO format from Flutter

class BudgetInput(BaseModel):
    savings: float
    expenses: List[ExpenseItem]

@app.post("/predict")
def predict(data: BudgetInput):
    prediction = model.predict([[data.savings]])[0]
    suggested_budget = round(prediction, 2)

    # ------------------------------
    # Category-Based Analysis (Existing)
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

    # ------------------------------
    # Forecasting Future Expenses
    df = pd.DataFrame([e.dict() for e in data.expenses])
    df['date'] = pd.to_datetime(df['date'])

    if df.empty:
        forecast = 0
    else:
        df['day'] = df['date'].dt.day
        total_spent = df['amount'].sum()
        days_so_far = df['date'].dt.day.nunique()
        today = datetime.today()
        days_in_month = pd.Period(today.strftime('%Y-%m')).days_in_month

        daily_avg = total_spent / days_so_far
        forecast = round(daily_avg * days_in_month, 2)

    overspending = forecast > suggested_budget

    # ------------------------------
    # Final Tips
    base_tips = [
        f"Try to keep your total monthly expenses under RM {suggested_budget}.",
        "Save at least 20% of your savings if possible.",
        "Cut down on non-essential expenses like shopping or entertainment."
    ]

    forecast_tip = f"At your current pace, you're projected to spend RM {forecast} this month."
    if overspending:
        forecast_tip += " This exceeds your recommended budget."

    return {
        "predicted_expense": suggested_budget,
        "forecast_expense": forecast,
        "overspending": overspending,
        "recommendations": base_tips + personalized + [forecast_tip]
    }
