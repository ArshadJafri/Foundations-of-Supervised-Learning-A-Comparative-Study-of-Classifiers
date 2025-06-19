import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegressionMultiOutput:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        n_samples, n_features = X.shape
        _, n_outputs = y.shape

        # Initialize weights and bias for multi-output regression
        self.weights = np.zeros((n_features, n_outputs))
        self.bias = np.zeros((1, n_outputs))

        # Split the data into 90% training and 10% validation for early stopping
        validation_size = int(0.1 * n_samples)
        X_train, X_val = X[:-validation_size], X[-validation_size:]
        y_train, y_val = y[:-validation_size], y[-validation_size:]

        best_weights = np.copy(self.weights)
        best_bias = np.copy(self.bias)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]

            # Mini-batch gradient descent
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Compute predictions (y_pred = X @ W + b)
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # Compute error and gradients
                error = y_pred - y_batch
                dw = (2 / len(X_batch)) * np.dot(X_batch.T, error) + \
                    (2 * regularization * self.weights)
                db = (2 / len(X_batch)) * np.sum(error, axis=0, keepdims=True)

                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Validation set evaluation
            val_pred = np.dot(X_val, self.weights) + self.bias
            val_loss = np.mean((val_pred - y_val) ** 2) + \
                regularization * np.sum(self.weights ** 2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = np.copy(self.weights)
                best_bias = np.copy(self.bias)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore the best weights and bias
        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        return mse


# Load the Iris dataset
iris = load_iris()

# Inputs: Sepal length and Sepal width
X = iris.data[:, [0, 1]]  # Sepal length and Sepal width
# Outputs: Petal length and Petal width
y = iris.data[:, [2, 3]]  # Petal length and Petal width

# Split the dataset into 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = LinearRegressionMultiOutput(learning_rate=0.01)
model.fit(X_train, y_train, batch_size=16,
          regularization=0.1, max_epochs=100, patience=5)

# Predict and evaluate the model on the test set
mse = model.score(X_test, y_test)
print(f"Mean Squared Error on the test set: {mse}")
