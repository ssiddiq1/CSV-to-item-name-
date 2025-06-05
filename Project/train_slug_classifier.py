import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import json
import re
from urllib.parse import urlparse, unquote
import logging
from typing import Dict, List, Tuple
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlugFeatureExtractor:
    """Extract features from URL segments for slug classification"""
    
    def __init__(self):
        self.stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through'}
        self.feature_names = [
            'length', 'num_hyphens', 'num_underscores', 'alpha_ratio', 
            'digit_ratio', 'token_count', 'avg_token_length', 'has_stopwords',
            'non_alnum_ratio', 'ends_html', 'ends_php', 'ends_aspx',
            'starts_digit', 'all_caps_ratio', 'camel_case_score',
            'has_uuid_pattern', 'has_sku_pattern', 'entropy'
        ]
    
    def extract_features(self, segment: str) -> Dict[str, float]:
        """Extract features from a single URL segment"""
        decoded = unquote(segment)
        
        features = {
            'length': len(decoded),
            'num_hyphens': decoded.count('-'),
            'num_underscores': decoded.count('_'),
            'alpha_ratio': sum(c.isalpha() for c in decoded) / max(len(decoded), 1),
            'digit_ratio': sum(c.isdigit() for c in decoded) / max(len(decoded), 1),
            'token_count': len(re.findall(r'[a-zA-Z0-9]+', decoded)),
            'avg_token_length': np.mean([len(t) for t in re.findall(r'[a-zA-Z0-9]+', decoded)] or [0]),
            'has_stopwords': int(any(word.lower() in self.stopwords for word in re.findall(r'[a-zA-Z]+', decoded))),
            'non_alnum_ratio': sum(not c.isalnum() for c in decoded) / max(len(decoded), 1),
            'ends_html': int(decoded.lower().endswith('.html')),
            'ends_php': int(decoded.lower().endswith('.php')),
            'ends_aspx': int(decoded.lower().endswith('.aspx')),
            'starts_digit': int(decoded[0].isdigit() if decoded else 0),
            'all_caps_ratio': sum(c.isupper() for c in decoded) / max(sum(c.isalpha() for c in decoded), 1),
            'camel_case_score': len(re.findall(r'[a-z][A-Z]', decoded)),
            'has_uuid_pattern': int(bool(re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', decoded.lower()))),
            'has_sku_pattern': int(bool(re.search(r'^[A-Z0-9]{6,}$', decoded))),
            'entropy': self._calculate_entropy(decoded)
        }
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(text)]
        return -sum(p * np.log2(p) for p in prob if p > 0)

def prepare_training_data(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and prepare training data from labeled CSV"""
    logger.info(f"Loading training data from {csv_path}")
    
    # Expected format: url,segment,is_slug
    df = pd.read_csv(csv_path)
    
    extractor = SlugFeatureExtractor()
    features_list = []
    
    # Extract features in parallel for speed
    with concurrent.futures.ProcessPoolExecutor() as executor:
        features = list(executor.map(extractor.extract_features, df['segment']))
    
    X = pd.DataFrame(features)
    y = df['is_slug'].values
    
    return X, y

def train_classifier(X: pd.DataFrame, y: np.ndarray) -> Tuple[RandomForestClassifier, StandardScaler, Dict]:
    """Train the slug classifier with proper validation"""
    logger.info("Training slug classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Train model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    clf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': clf.score(X_test_scaled, y_test),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': dict(zip(X.columns, clf.feature_importances_))
    }
    
    logger.info(f"Model AUC: {metrics['auc']:.4f}")
    logger.info(f"Model Accuracy: {metrics['accuracy']:.4f}")
    
    return clf, scaler, metrics

def main():
    # Load and prepare data
    X, y = prepare_training_data('data/training_data.csv')
    
    # Train classifier
    clf, scaler, metrics = train_classifier(X, y)
    
    # Save model and scaler
    joblib.dump(clf, 'models/slug_classifier.joblib')
    joblib.dump(scaler, 'models/slug_scaler.joblib')
    
    # Save feature names
    with open('config/feature_cols.json', 'w') as f:
        json.dump(list(X.columns), f, indent=2)
    
    # Save metrics
    with open('logs/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()