import pickle
from sklearn.linear_model import LinearRegression

# 🧑‍🎓 Estimated monthly student income → suggested budget
X = [
    [300], [500], [700], [900], [1100], [1300],
    [1500], [1700], [1900], [2100], [2300], [2500]
]

# Budget targets: assumes students spend ~30% to 40% of their income
y = [
    100, 150, 210, 270, 330, 390,
    450, 500, 550, 600, 650, 700
]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save it
with open("budget_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Student-budget model trained and saved!")
