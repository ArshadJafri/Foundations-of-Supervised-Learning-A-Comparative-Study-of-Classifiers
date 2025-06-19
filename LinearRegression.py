import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        n_samples, n_features = X.shape
        validation_size = int(0.1 * n_samples)
        X_train, X_val = X[:-validation_size], X[-validation_size:]
        y_train, y_val = y[:-validation_size], y[-validation_size:]

        self.weights = np.zeros(n_features)
        self.bias = 0

        best_weights = np.copy(self.weights)
        best_bias = self.bias
        best_val_loss = float('inf')
        patience_counter = 0

        # To track training loss
        train_losses = []

        for epoch in range(max_epochs):
            indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]

            # Mini-batch gradient descent
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Compute predictions
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # Compute gradients
                error = y_pred - y_batch
                dw = (2 / len(X_batch)) * np.dot(X_batch.T, error) + \
                    (2 * regularization * self.weights)
                db = (2 / len(X_batch)) * np.sum(error)

                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Compute training loss for this epoch
            y_train_pred = np.dot(X_train, self.weights) + self.bias
            train_loss = np.mean((y_train_pred - y_train) ** 2) + \
                regularization * np.sum(self.weights ** 2)
            train_losses.append(train_loss)

            # Validation loss for early stopping
            val_pred = np.dot(X_val, self.weights) + self.bias
            val_loss = np.mean((val_pred - y_val) ** 2) + \
                regularization * np.sum(self.weights ** 2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = np.copy(self.weights)
                best_bias = self.bias
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best weights and bias
        self.weights = best_weights
        self.bias = best_bias

        return train_losses

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        return mse
