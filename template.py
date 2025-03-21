import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filepath, target_column):
    """Load data from a file and split into features and target."""
    data = pd.read_csv(filepath)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_logistic_regression(X_train, y_train, solver='liblinear'):
    """Train a logistic regression model."""
    model = LogisticRegression(solver=solver)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    file_path = 'sheets_data.xlsx'  # Replace with your data file
    target_column = 'target'  # Replace with your target column

    X, y = load_data(file_path, target_column)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_logistic_regression(X_train, y_train)

    evaluate_model(model, X_test, y_test)
