from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
 
DATA_DIR = Path(__file__).resolve().parent / "data"

def load_data():
    training_data = pd.read_csv(DATA_DIR / "RingSynTrain.csv")
    test_data = pd.read_csv(DATA_DIR / "RingSynTest.csv")
    X_train = training_data.drop("class", axis=1).values
    y_train = training_data["class"].values

    X_test = test_data.drop("class", axis=1).values
    y_test = test_data["class"].values

    return X_train, y_train, X_test, y_test
 
def train_and_evaluate(activation_function):
    X_train, y_train, X_test, y_test = load_data()

    # Use the same scaling for both activation functions
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Same architecture as Task c: one hidden layer with 8 neurons
    mlp = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation=activation_function,
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy, mlp.n_iter_

sigmoid_accuracy, sigmoid_iterations = train_and_evaluate("logistic")
relu_accuracy, relu_iterations = train_and_evaluate("relu")

print(f"sigmoid, accuracy: {sigmoid_accuracy:.2f}, iterations: {sigmoid_iterations}")
print(f"relu, accuracy: {relu_accuracy:.2f}, iterations: {relu_iterations}")
