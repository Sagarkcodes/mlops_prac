# save_model.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Iris dataset
X, y = load_iris(return_X_y=True)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model to a file
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'iris_model.pkl'")
