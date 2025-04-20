# train_model.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import sys

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Read hyperparameter from command line
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# Train model
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"n_estimators: {n_estimators} -> Accuracy: {accuracy:.4f}")

# Save model
model_filename = f"model_v{n_estimators}.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved as: {model_filename}")
