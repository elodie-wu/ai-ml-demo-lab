from pathlib import Path

import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def activate(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        return np.array([
            self.activate(np.dot(x, self.weights) + self.bias)
            for x in X
        ])

    def fit(self, X, y, epochs):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_predicted = self.activate(linear_output)

                update = self.learning_rate * (y[i] - y_predicted)

                self.weights += update * X[i]
                self.bias += update


def evaluate_dataset(train_file, test_file, task_name):
    training_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = training_data.drop("class", axis=1).values
    y_train = training_data["class"].values

    X_test = test_data.drop("class", axis=1).values
    y_test = test_data["class"].values

    epoch_list = [1, 5, 10, 15, 20, 50, 60, 80, 100, 120, 150, 200]

    print(task_name)

    for epoch in epoch_list:
        model = Perceptron(learning_rate=0.1)
        model.fit(X_train, y_train, epochs=epoch)

        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        print(f"epoch: {epoch}, accuracy: {accuracy:.2f}")


DATA_DIR = Path(__file__).resolve().parent / "data"

# Task a  
evaluate_dataset(DATA_DIR / "SeaSynTrain.csv", DATA_DIR / "SeaSynTest.csv", "Task a: SeaSyn dataset")

print("==============================")

# Task b 
evaluate_dataset(DATA_DIR / "RingSynTrain.csv", DATA_DIR / "RingSynTest.csv", "Task b: RingSyn dataset")
