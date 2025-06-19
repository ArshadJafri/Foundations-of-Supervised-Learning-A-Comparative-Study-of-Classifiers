import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data[:, [0, 1]]  # Sepal length and width
y = (iris.target != 0) * 1  # Binary classification: Setosa vs non-Setosa

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train logistic regression model
model = LogisticRegression(learning_rate=0.01, max_epochs=1000)
model.fit(X_train, y_train)

# Save model parameters
np.save('weights_classifier2.npy', model.weights)
np.save('bias_classifier2.npy', model.bias)

# Plot decision regions using mlxtend
plot_decision_regions(X_train, y_train, clf=model, legend=2)
plt.title("Logistic Regression - Sepal Length/Width")
plt.show()

# Print accuracy on the training set
train_accuracy = model.accuracy(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}")
