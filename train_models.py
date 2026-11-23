"""
Standalone script to train Logistic Regression, kNN, and Decision Tree models
on the Iris dataset, evaluate their accuracy and confusion matrices, and save
the trained models as .pkl files.
"""

import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    print("=" * 60)
    print("IRIS MULTICLASS CLASSIFICATION PIPELINE (TOPIC 31)")
    print("Group 16 | Class: LYCORE 2 | MIT School of Computing")
    print("=" * 60)

    # 1. Load Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"\n[+] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[+] Features: {feature_names}")
    print(f"[+] Target classes: {target_names}")

    # 2. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Initialize & Train Models
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=200), "logistic_model.pkl"),
        "k-Nearest Neighbors (k=5)": (KNeighborsClassifier(n_neighbors=5), "knn_model.pkl"),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), "decision_tree_model.pkl")
    }

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    for name, (clf, filename) in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n>>> Model: {name}")
        print(f"    Testing Accuracy: {acc:.2f}%")
        print("    Confusion Matrix:")
        for row in cm:
            print(f"      {row}")

        # Save model
        joblib.dump(clf, filename)
        print(f"    [Saved]: {filename}")

    print("\n[+] All models trained, evaluated, and saved successfully!")

if __name__ == "__main__":
    main()
