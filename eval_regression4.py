import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression

# Load Iris dataset
iris = load_iris()
X = iris.data[:, [0, 1, 2]]  # Sepal length, Sepal width, and Petal length
y = iris.data[:, 3]          # Petal width

# Split the dataset into 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load saved model parameters for non-regularized model
weights_non_reg = np.load('weights_non_reg4.npy')
bias_non_reg = np.load('bias_non_reg4.npy')

# Load saved model parameters for regularized model
weights_reg = np.load('weights_reg4.npy')
bias_reg = np.load('bias_reg4.npy')

# Initialize Linear Regression objects for evaluation
model_non_reg = LinearRegression()
model_non_reg.weights = weights_non_reg
model_non_reg.bias = bias_non_reg

model_reg = LinearRegression()
model_reg.weights = weights_reg
model_reg.bias = bias_reg

# Evaluate both models on the test set
mse_non_reg = model_non_reg.score(X_test, y_test)
mse_reg = model_reg.score(X_test, y_test)

print(f"Mean Squared Error on test set (Non-Regularized Model): {mse_non_reg}")
print(f"Mean Squared Error on test set (Regularized Model): {mse_reg}")
