# src/models/classical_ml.py
from sklearn.ensemble import RandomForestClassifier

def train_classifier(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier
    
    Args:
        X: Feature matrix
        y: Target labels
    
    Returns:
        Trained Random Forest classifier
    """
    return RandomForestClassifier().fit(X, y)