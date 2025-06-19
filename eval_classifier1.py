import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression

# Load Iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]  # Petal length and width
y = (iris.target != 0) * 1  # Binary classification: Setosa vs non-Setosa

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load saved model parameters (trained in train_classifier1.py)
weights = np.load('weights_classifier1.npy')
bias = np.load('bias_classifier1.npy')

# Initialize logistic regression model and set the weights and bias
model = LogisticRegression()
model.weights = weights
model.bias = bias

# Evaluate model accuracy on the test set
test_accuracy = model.accuracy(X_test, y_test)
print(f"Test Accuracy (Petal Length/Width): {test_accuracy}")
