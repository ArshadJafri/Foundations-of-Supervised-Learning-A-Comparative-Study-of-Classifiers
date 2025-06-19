import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from LogisticRegression import LogisticRegression

# Load Iris dataset
iris = load_iris()
X = iris.data  # All features: sepal length, sepal width, petal length, petal width
y = (iris.target != 0) * 1  # Binary classification: Setosa vs non-Setosa

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reduce dimensionality using PCA (to match the training process)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Load saved model parameters (trained in train_classifier3.py)
weights = np.load('weights_classifier3.npy')
bias = np.load('bias_classifier3.npy')

# Initialize logistic regression model and set the weights and bias
model = LogisticRegression()
model.weights = weights
model.bias = bias

# Evaluate model accuracy on the test set (PCA-transformed)
test_accuracy = model.accuracy(X_test_pca, y_test)
print(f"Test Accuracy (All Features with PCA): {test_accuracy}")
