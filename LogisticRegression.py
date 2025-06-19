import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_epochs=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Numerically stable sigmoid function
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (np.exp(z) + 1))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.max_epochs):
            # Linear model: z = X @ weights + bias
            z = np.dot(X, self.weights) + self.bias

            # Apply sigmoid to get probabilities
            predictions = self.sigmoid(z)

            # Compute the loss (cross-entropy loss)
            loss = -np.mean(y * np.log(predictions + 1e-15) +
                            (1 - y) * np.log(1 - predictions + 1e-15))

            # Gradient of loss w.r.t. weights and bias
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            # Stopping criterion (early stopping)
            if np.all(np.abs(dw) < self.tolerance) and np.abs(db) < self.tolerance:
                break

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        predictions = np.array([1 if p > 0.5 else 0 for p in probabilities])
        # Print first 10 predictions for debugging
        print(f"Predictions: {predictions[:10]}")
        return predictions

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
