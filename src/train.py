import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json
import pickle
import os

df = pd.read_csv("data/demand.csv")

X = df[["week"]]
y = df["shipments"]

model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Metrics
score = model.score(X, y)
metrics = {"r2_score": score}

os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f)

# Plot
plt.plot(df["week"], y, label="Actual")
plt.plot(df["week"], model.predict(X), label="Predicted")
plt.legend()
plt.savefig("plots/demand.png")

