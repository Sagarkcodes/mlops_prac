# use_model.py

import pickle
from sklearn.datasets import load_iris

# Load saved model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data to test
X, y = load_iris(return_X_y=True)

# Predict using the loaded model
predictions = model.predict(X[:5])
print("✅ Predictions on sample input:", predictions)
