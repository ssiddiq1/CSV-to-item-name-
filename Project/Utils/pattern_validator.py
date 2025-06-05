"""
Pattern validation utilities for the slug extraction system
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternValidator:
    """Validates regex patterns for accuracy and performance"""
    
    def __init__(self, min_accuracy: float = 0.85, min_coverage: float = 0.10):
        self.min_accuracy = min_accuracy
        self.min_coverage = min_coverage
        
    def validate_pattern(self, pattern: str, test_urls: List[str], 
                        classifier=None, threshold: float = 0.55) -> Dict:
        """
        Validate a single pattern against test URLs
        
        Returns:
            dict: Validation metrics including accuracy, coverage, and performance
        """
        try:
            regex = re.compile(pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {pattern}, error: {e}")
            return {'valid': False, 'error': str(e)}
        
        matches = 0
        correct = 0
        total = len(test_urls)
        extraction_times = []
        
        for url in test_urls:
            import time
            start_time = time.time()
            match = regex.search(url)
            extraction_time = time.time() - start_time
            extraction_times.append(extraction_time)
            
            if match:
                matches += 1
                extracted = match.group(1)
                
                # If classifier provided, verify the extraction
                if classifier:
                    score = self._score_with_classifier(extracted, classifier)
                    if score >= threshold:
                        correct += 1
                else:
                    # Without classifier, assume all matches are correct
                    correct += 1
        
        coverage = matches / total if total > 0 else 0
        accuracy = correct / matches if matches > 0 else 0
        avg_extraction_time = np.mean(extraction_times) * 1000  # Convert to ms
        
        validation_result = {
            'valid': accuracy >= self.min_accuracy and coverage >= self.min_coverage,
            'accuracy': accuracy,
            'coverage': coverage,
            'matches': matches,
            'correct': correct,
            'total': total,
            'avg_extraction_time_ms': avg_extraction_time,
            'pattern': pattern
        }
        
        return validation_result
    
    def _score_with_classifier(self, segment: str, classifier) -> float:
        """Score a segment using the ML classifier"""
        # This is a placeholder - in real implementation, would use actual feature extraction
        # and classifier prediction as shown in the main system
        return np.random.random()  # Placeholder score
    
    def validate_pattern_set(self, patterns: List[str], test_urls: List[str], 
                           domain: str = None) -> Dict:
        """
        Validate a set of patterns for a specific domain
        
        Returns:
            dict: Aggregated validation metrics for the pattern set
        """
        logger.info(f"Validating {len(patterns)} patterns for domain: {domain}")
        
        if domain:
            # Filter URLs for this domain
            domain_urls = [url for url in test_urls if urlparse(url).netloc == domain]
        else:
            domain_urls = test_urls
        
        if not domain_urls:
            logger.warning(f"No test URLs found for domain: {domain}")
            return {'valid': False, 'error': 'No test URLs'}
        
        # Track which URLs are matched by any pattern
        matched_urls = set()
        pattern_results = []
        
        for pattern in patterns:
            result = self.validate_pattern(pattern, domain_urls)
            pattern_results.append(result)
            
            # Track matched URLs
            if result['valid']:
                regex = re.compile(pattern)
                for url in domain_urls:
                    if regex.search(url):
                        matched_urls.add(url)
        
        # Calculate aggregate metrics
        total_coverage = len(matched_urls) / len(domain_urls) if domain_urls else 0
        valid_patterns = [r for r in pattern_results if r['valid']]
        
        aggregate_result = {
            'domain': domain,
            'total_patterns': len(patterns),
            'valid_patterns': len(valid_patterns),
            'total_coverage': total_coverage,
            'pattern_results': pattern_results,
            'matched_urls': len(matched_urls),
            'total_urls': len(domain_urls),
            'all_valid': len(valid_patterns) == len(patterns) and len(valid_patterns) > 0
        }
        
        return aggregate_result
    
    def find_conflicting_patterns(self, patterns: List[str], sample_urls: List[str]) -> List[Dict]:
        """
        Find patterns that might conflict (match the same URLs)
        
        Returns:
            list: Conflicts with pattern pairs and overlap statistics
        """
        conflicts = []
        pattern_matches = {}
        
        # First, find all matches for each pattern
        for i, pattern in enumerate(patterns):
            try:
                regex = re.compile(pattern)
                matches = set()
                
                for url in sample_urls:
                    if regex.search(url):
                        matches.add(url)
                
                pattern_matches[i] = matches
            except re.error:
                continue
        
        # Now find overlaps
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                if i in pattern_matches and j in pattern_matches:
                    overlap = pattern_matches[i] & pattern_matches[j]
                    
                    if overlap:
                        overlap_ratio = len(overlap) / min(len(pattern_matches[i]), len(pattern_matches[j]))
                        
                        conflicts.append({
                            'pattern1': patterns[i],
                            'pattern2': patterns[j],
                            'overlap_count': len(overlap),
                            'overlap_ratio': overlap_ratio,
                            'sample_overlaps': list(overlap)[:5]  # Show first 5 examples
                        })
        
        return conflicts
    
    def suggest_pattern_improvements(self, pattern: str, failed_urls: List[str]) -> List[str]:
        """
        Suggest improvements to a pattern based on URLs it failed to match