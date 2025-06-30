import pickle
from sklearn.linear_model import LinearRegression

# 🧑‍🎓 Estimated monthly student savings
X = [
    [300], [500], [700], [900], [1100], [1300],
    [1500], [1700], [1900], [2100], [2300], [2500]
]

# Updated budget targets (≈75% of savings)
y = [round(x[0] * 0.75) for x in X]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save it
with open("budget_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Student-budget model trained and saved with 75% spending target!")
