from Data_Prep import X_train, y_train
from Data_Prep import X_test, y_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pickle

# Base model
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# Parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 12, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Train model
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Best model
model = grid_search.best_estimator_

print("\nModel trained successfully!")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report (VERY IMPORTANT)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross Validation
scores = cross_val_score(model, X_train, y_train, cv=5)

print("\nCross Validation Accuracy:", scores.mean())

# Save model
pickle.dump(model, open("diabetes_model.pkl", "wb"))
print("\nModel saved as diabetes_model.pkl")