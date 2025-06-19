import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression


def plot_loss(train_losses, title, label):
    plt.plot(train_losses, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


# Load Iris dataset
iris = load_iris()
X = iris.data[:, [0]]  # Sepal length
y = iris.data[:, 3]    # Petal width

# Split the dataset into 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Non-Regularized Model
model_non_reg = LinearRegression(learning_rate=0.01)
train_losses_non_reg = model_non_reg.fit(
    X_train, y_train, batch_size=16, regularization=0, max_epochs=100, patience=5)

# Regularized Model (L2 Regularization)
model_reg = LinearRegression(learning_rate=0.01)
train_losses_reg = model_reg.fit(
    X_train, y_train, batch_size=16, regularization=0.1, max_epochs=100, patience=5)

# Save the model parameters
np.save('weights_non_reg2.npy', model_non_reg.weights)
np.save('bias_non_reg2.npy', model_non_reg.bias)
np.save('weights_reg2.npy', model_reg.weights)
np.save('bias_reg2.npy', model_reg.bias)

# Plot the loss over epochs for both models
plt.figure(figsize=(10, 5))
plot_loss(train_losses_non_reg,
          "Training Loss - Non-Regularized", "No Regularization")
plot_loss(train_losses_reg, "Training Loss - L2 Regularization",
          "L2 Regularization (λ = 0.1)")
plt.show()

# Print the difference in weights and biases between regularized and non-regularized models
weight_diff = model_non_reg.weights - model_reg.weights
bias_diff = model_non_reg.bias - model_reg.bias
print(f"Difference in Weights: {weight_diff}")
print(f"Difference in Biases: {bias_diff}")
