import pickle
from sklearn.linear_model import LinearRegression

X = [
    [300], [500], [700], [900], [1100], [1300],
    [1500], [1700], [1900], [2100], [2300], [2500]
]

# This gives you 75% budgets
y = [round(x[0] * 0.75) for x in X]

model = LinearRegression()
model.fit(X, y)

print("✅ Model coef (should be 0.75):", model.coef_)

with open("budget_model.pkl", "wb") as f:
    pickle.dump(model, f)
