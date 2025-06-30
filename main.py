from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
import pickle
import calendar

app = FastAPI()

# Load trained model (based on 75% of savings)
with open("budget_model.pkl", "rb") as f:
    model = pickle.load(f)

class ExpenseItem(BaseModel):
    category: str
    amount: float
    date: str  # ISO format from Flutter

class BudgetInput(BaseModel):
    savings: float
    expenses: List[ExpenseItem]

@app.post("/predict")
def predict(data: BudgetInput):
    # Predict suggested budget (based on 75% savings logic)
    prediction = model.predict([[data.savings]])[0]
    suggested_budget = round(float(prediction), 2)  # Convert numpy.float to float

    # Category Spending Breakdown
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

    # Forecasting future expenses
    if not data.expenses:
        forecast = 0.0
    else:
        parsed_dates = [datetime.fromisoformat(e.date) for e in data.expenses]
        amounts = [e.amount for e in data.expenses]

        total_spent = sum(amounts)
        unique_days = set(d.date() for d in parsed_dates)
        days_so_far = len(unique_days)

        today = datetime.today()
        days_in_month = calendar.monthrange(today.year, today.month)[1]

        daily_avg = total_spent / days_so_far if days_so_far else 0
        forecast = round(daily_avg * days_in_month, 2)

    overspending = bool(forecast > suggested_budget)

    # Smart Recommendations
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
