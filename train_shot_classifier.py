import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def main():
    if not os.path.exists("shot_features_labeled.csv"):
        print("Run annotate_shots.py first!")
        return

    df = pd.read_csv("shot_features_labeled.csv")
    
    # Drop rows without labels or with all-NaN proximities
    df = df.dropna(subset=['label'])
    
    if len(df) < 5:
        print("Not enough labeled data to train. Need at least 5 examples.")
        return

    print(f"Training on {len(df)} samples...")
    print("Class distribution:")
    print(df['label'].value_counts())

    # Features: prox_{-10} to prox_{10}, rel_height, player_y_m
    prox_cols = [c for c in df.columns if c.startswith('prox_')]
    feature_cols = prox_cols + ['rel_height']
    
    # Add optional features if they exist
    if 'player_y_m' in df.columns:
        feature_cols.append('player_y_m')
    
    X = df[feature_cols].fillna(0) # Basic imputation
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    if not os.path.exists("weights"):
        os.makedirs("weights")
        
    joblib.dump(clf, "weights/shot_classifier.pkl")
    # Save feature names to ensure consistency during inference
    joblib.dump(feature_cols, "weights/shot_classifier_features.pkl")
    
    print("\nModel saved to weights/shot_classifier.pkl")

if __name__ == "__main__":
    main()
