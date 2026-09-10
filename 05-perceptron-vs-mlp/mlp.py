from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
 
# Load dataset
DATA_DIR = Path(__file__).resolve().parent / "data"
training_data = pd.read_csv(DATA_DIR / "RingSynTrain.csv")
test_data = pd.read_csv(DATA_DIR / "RingSynTest.csv")

X_train = training_data.drop("class", axis=1).values
y_train = training_data["class"].values

X_test = test_data.drop("class", axis=1).values
y_test = test_data["class"].values


# Standardise features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# Train MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)
 
predictions = mlp.predict(X_test) 
accuracy = accuracy_score(y_test, predictions)

print(f"MLP: accuracy = {accuracy:.2f} ，iterations = {mlp.n_iter_}")
