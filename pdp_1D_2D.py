"""
Very simple example: Partial Dependence Plots (PDP) in Python with scikit-learn.
- Binary classification on the Breast Cancer dataset
- RandomForestClassifier as the model
- 1D PDPs for two features
- 2D PDP for a pair of features
- Optional ICE curves

Run:
    python Simple_PDP_in_Python.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay



# 1) Data
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2) Model (any fitted estimator with predict_proba works)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# 3) 1D PDPs
# Choose two informative, human-readable features
f_idx_1 = feature_names.index("mean radius")
f_idx_2 = feature_names.index("mean texture")

fig1 = plt.figure(figsize=(8, 4))
PartialDependenceDisplay.from_estimator(
    clf,
    X_test,
    features=[f_idx_1, f_idx_2],
    feature_names=feature_names,
    kind="average",  # partial dependence (no ICE)
    target=1,  # class=1 (malignant vs benign depends on dataset encoding)
)
plt.suptitle("1D Partial Dependence (Random Forest)")
plt.tight_layout()

# 4) 2D PDP (interaction surface)
fig2 = plt.figure(figsize=(5, 4))
PartialDependenceDisplay.from_estimator(
    clf,
    X_test,
    features=[(f_idx_1, f_idx_2)],
    feature_names=feature_names,
    kind="average",
    target=1,
)
plt.suptitle("2D Partial Dependence: mean radius × mean texture")
plt.tight_layout()

