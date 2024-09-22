# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

# Fetch heart disease dataset from UCI ML repository
heart_disease = fetch_ucirepo(id=45)

# Extract features (X) and target labels (y)
X = heart_disease.data.features
y = heart_disease.data.targets

# Display metadata and variables for understanding the dataset
print("Dataset Metadata:")
print(heart_disease.metadata)

print("\nVariable Information:")
print(heart_disease.variables)

# Handle missing values by imputing with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Display the first few rows of the dataset to ensure it's loaded correctly
print("\nFirst few rows of the features (X):")
print(pd.DataFrame(X_imputed).head())

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the Logistic Regression model: {accuracy * 100:.2f}%")

# Generate and display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate and display a confusion matrix for better understanding
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
