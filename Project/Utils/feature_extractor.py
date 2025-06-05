# This is already defined in train_slug_classifier.py as SlugFeatureExtractor
# We'll create a separate file for reusability

from typing import Dict
import numpy as np
import re

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
        features = {
            'length': len(segment),
            'num_hyphens': segment.count('-'),
            'num_underscores': segment.count('_'),
            'alpha_ratio': sum(c.isalpha() for c in segment) / max(len(segment), 1),
            'digit_ratio': sum(c.isdigit() for c in segment) / max(len(segment), 1),
            'token_count': len(re.findall(r'[a-zA-Z0-9]+', segment)),
            'avg_token_length': np.mean([len(t) for t in re.findall(r'[a-zA-Z0-9]+', segment)] or [0]),
            'has_stopwords': int(any(word.lower() in self.stopwords for word in re.findall(r'[a-zA-Z]+', segment))),
            'non_alnum_ratio': sum(not c.isalnum() for c in segment) / max(len(segment), 1),
            'ends_html': int(segment.lower().endswith('.html')),
            'ends_php': int(segment.lower().endswith('.php')),
            'ends_aspx': int(segment.lower().endswith('.aspx')),
            'starts_digit': int(segment[0].isdigit() if segment else 0),
            'all_caps_ratio': sum(c.isupper() for c in segment) / max(sum(c.isalpha() for c in segment), 1),
            'camel_case_score': len(re.findall(r'[a-z][A-Z]', segment)),
            'has_uuid_pattern': int(bool(re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', segment.lower()))),
            'has_sku_pattern': int(bool(re.search(r'^[A-Z0-9]{6,}# Universal E-commerce URL Slug Extraction System
