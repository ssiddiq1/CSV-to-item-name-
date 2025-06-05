import pandas as pd
import numpy as np
import json
import re
from urllib.parse import urlparse, unquote
import joblib
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import logging
import random
from concurrent.futures import ProcessPoolExecutor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternDiscoverer:
    def __init__(self, classifier_path: str, scaler_path: str, feature_cols_path: str):
        self.classifier = joblib.load(classifier_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_cols_path, 'r') as f:
            self.feature_cols = json.load(f)
        
        self.extractor = SlugFeatureExtractor()  # Reuse from train_slug_classifier.py
        self.min_confidence = 0.55
        self.validation_threshold = 0.85
    
    def discover_patterns_for_domain(self, domain: str, urls: List[str], 
                                   time_limit: int = 10) -> List[str]:
        """Discover regex patterns for a domain within time limit"""
        start_time = time.time()
        logger.info(f"Discovering patterns for {domain} with {len(urls)} URLs")
        
        # Sample if too many URLs
        if len(urls) > 20000:
            urls = random.sample(urls, 20000)
        
        # Extract and classify segments
        segment_scores = self._classify_url_segments(urls[:10000])
        
        # Find common patterns
        patterns = self._infer_patterns(segment_scores, domain)
        
        # Validate on held-out set
        if len(urls) > 10000:
            validated_patterns = self._validate_patterns(patterns, urls[10000:], segment_scores)
        else:
            validated_patterns = patterns[:3]  # Take top 3 if no validation set
        
        elapsed = time.time() - start_time
        logger.info(f"Pattern discovery for {domain} completed in {elapsed:.2f}s")
        
        return validated_patterns
    
    def _classify_url_segments(self, urls: List[str]) -> Dict[str, List[Tuple[str, float, int]]]:
        """Classify all segments in URLs and return scores"""
        segment_scores = defaultdict(list)
        
        for url in urls:
            parsed = urlparse(url)
            path_segments = [s for s in parsed.path.split('/') if s]
            
            for idx, segment in enumerate(path_segments):
                features = self.extractor.extract_features(segment)
                X = pd.DataFrame([features])[self.feature_cols]
                X_scaled = self.scaler.transform(X)
                
                proba = self.classifier.predict_proba(X_scaled)[0, 1]
                segment_scores[url].append((segment, proba, idx))
        
        return segment_scores
    
    def _infer_patterns(self, segment_scores: Dict, domain: str) -> List[str]:
        """Infer regex patterns from classified segments"""
        slug_contexts = []
        
        for url, segments in segment_scores.items():
            # Find most likely slug
            if not segments:
                continue
                
            best_segment = max(segments, key=lambda x: x[1])
            if best_segment[1] < self.min_confidence:
                continue
            
            # Extract context (before/after segments)
            parsed = urlparse(url)
            path_parts = [s for s in parsed.path.split('/') if s]
            slug_idx = best_segment[2]
            
            context = {
                'slug': best_segment[0],
                'before': path_parts[slug_idx-1] if slug_idx > 0 else None,
                'after': path_parts[slug_idx+1] if slug_idx < len(path_parts)-1 else None,
                'position': slug_idx,
                'total_segments': len(path_parts)
            }
            slug_contexts.append(context)
        
        # Group by patterns
        pattern_groups = self._group_by_pattern(slug_contexts)
        
        # Generate regexes
        regex_patterns = []
        for pattern_type, contexts in pattern_groups.items():
            if len(contexts) < 10:  # Need sufficient examples
                continue
                
            regex = self._generate_regex(pattern_type, contexts, domain)
            if regex:
                regex_patterns.append(regex)
        
        return regex_patterns
    
    def _group_by_pattern(self, contexts: List[Dict]) -> Dict[str, List[Dict]]:
        """Group contexts by structural pattern"""
        groups = defaultdict(list)
        
        for ctx in contexts:
            # Create pattern signature
            sig_parts = []
            if ctx['before']:
                sig_parts.append(f"before={ctx['before']}")
            if ctx['after']:
                sig_parts.append(f"after={ctx['after']}")
            sig_parts.append(f"pos={ctx['position']}")
            sig_parts.append(f"total={ctx['total_segments']}")
            
            signature = "|".join(sig_parts)
            groups[signature].append(ctx)
        
        return dict(groups)
    
    def _generate_regex(self, pattern_type: str, contexts: List[Dict], domain: str) -> Optional[str]:
        """Generate regex from pattern group"""
        # Parse pattern type
        parts = dict(p.split('=') for p in pattern_type.split('|'))
        
        # Build regex components
        regex_parts = [re.escape(domain)]
        
        # Add path components
        if 'before' in parts and parts['before'] != 'None':
            regex_parts.append(f"/{re.escape(parts['before'])}")
        
        # Add slug capture group
        # Analyze slug patterns to determine appropriate regex
        slug_patterns = [ctx['slug'] for ctx in contexts]
        if all('-' in s for s in slug_patterns[:20]):  # Hyphenated slugs
            regex_parts.append(r"/([a-z0-9]+(?:-[a-z0-9]+)*)")
        elif all('_' in s for s in slug_patterns[:20]):  # Underscored slugs
            regex_parts.append(r"/([a-z0-9]+(?:_[a-z0-9]+)*)")
        else:  # Generic slug
            regex_parts.append(r"/([^/]+)")
        
        # Add after component if consistent
        if 'after' in parts and parts['after'] != 'None':
            regex_parts.append(f"/{re.escape(parts['after'])}")
        
        return ''.join(regex_parts)
    
    def _validate_patterns(self, patterns: List[str], validation_urls: List[str], 
                          original_scores: Dict) -> List[str]:
        """Validate patterns on held-out set"""
        validated = []
        
        for pattern in patterns:
            regex = re.compile(pattern)
            matches = 0
            correct = 0
            
            for url in validation_urls[:1000]:  # Sample for speed
                match = regex.search(url)
                if match:
                    matches += 1
                    extracted = match.group(1)
                    
                    # Verify with classifier
                    features = self.extractor.extract_features(extracted)
                    X = pd.DataFrame([features])[self.feature_cols]
                    X_scaled = self.scaler.transform(X)
                    proba = self.classifier.predict_proba(X_scaled)[0, 1]
                    
                    if proba >= self.min_confidence:
                        correct += 1
            
            if matches > 0:
                accuracy = correct / matches
                if accuracy >= self.validation_threshold:
                    validated.append(pattern)
                    logger.info(f"Validated pattern: {pattern} (accuracy: {accuracy:.2%})")
        
        return validated

def update_patterns_json(domain: str, new_patterns: List[str], patterns_file: str = 'config/patterns.json'):
    """Update patterns.json with new patterns"""
    try:
        with open(patterns_file, 'r') as f:
            patterns_dict = json.load(f)
    except FileNotFoundError:
        patterns_dict = {}
    
    if domain not in patterns_dict:
        patterns_dict[domain] = []
    
    # Add new patterns, avoiding duplicates
    for pattern in new_patterns:
        if pattern not in patterns_dict[domain]:
            patterns_dict[domain].append(pattern)
    
    # Save with backup
    import shutil
    shutil.copy2(patterns_file, f"{patterns_file}.backup")
    
    with open(patterns_file, 'w') as f:
        json.dump(patterns_dict, f, indent=2, sort_keys=True)
    
    logger.info(f"Updated patterns for {domain}: {len(new_patterns)} new patterns added")

def main(urls_csv: str, sample_size: int = 20000):
    """Main pattern discovery pipeline"""
    logger.info(f"Starting pattern discovery from {urls_csv}")
    
    # Load URLs
    df = pd.read_csv(urls_csv, nrows=sample_size)
    
    # Group by domain
    df['domain'] = df['URL'].apply(lambda x: urlparse(x).netloc)
    domain_groups = df.groupby('domain')['URL'].apply(list).to_dict()
    
    # Initialize discoverer
    discoverer = PatternDiscoverer(
        'models/slug_classifier.joblib',
        'models/slug_scaler.joblib',
        'config/feature_cols.json'
    )
    
    # Discover patterns for each domain
    for domain, urls in domain_groups.items():
        if len(urls) < 100:  # Skip domains with too few URLs
            continue
            
        try:
            patterns = discoverer.discover_patterns_for_domain(domain, urls)
            if patterns:
                update_patterns_json(domain, patterns)
        except Exception as e:
            logger.error(f"Error processing {domain}: {e}")
            continue

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data/urls.csv'
    main(csv_file)