import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ==========================================
# 1. INPUT PHASE (Data Loading)
# ==========================================
# Load the classic 150-sample balanced Iris domain dataset
iris = load_iris()
X, y = iris.data, iris.target

# Use stratify=y to ensure perfectly balanced class distributions in splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

# ==========================================
# 2. PROCESS PHASE (Robust Pipeline & Tuning)
# ==========================================
# Construct a machine learning pipeline to eliminate training data leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Normalizes data to mean=0, var=1
    ('knn', KNeighborsClassifier())     # Core algorithm core
])

# Define a hyperparameter grid search space to find the optimal 'k' value
param_grid = {
    'knn__n_neighbors': list(range(1, 12)),
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]  # 1: Manhattan Distance, 2: Euclidean Distance
}

# Apply 5-Fold Stratified Cross-Validation across the parameter grid
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Extract the absolute best model configuration discovered by optimization
best_model = grid_search.best_estimator_

# Generate predictions using the optimized architecture
y_pred = best_model.predict(X_test)

# ==========================================
# 3. OUTPUT PHASE (Advanced Evaluation)
# ==========================================
print("=== Optimization Discovery ===")
print(f"Optimal Parameters Discovered: {grid_search.best_params_}")
print(f"Best Validation Cross-Validation F1-Score: {grid_search.best_score_:.4f}\n")

print("=== Final Validation Set Metrics ===")
# Generate final Confusion Matrix and structured multi-class scores
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

final_f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Final Optimized Weighted F1-Score: {final_f1:.4f}")