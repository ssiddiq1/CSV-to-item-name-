import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import os
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousImprover:
    def __init__(self, extraction_stats_path: str = 'logs/extraction_stats.json'):
        self.stats_path = extraction_stats_path
        self.hard_cases_path = 'data/hard_cases.csv'
        self.failure_threshold = 0.1  # Flag domains with >10% failure rate
    
    def mine_hard_cases(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        """Mine URLs where extraction failed or confidence is low"""
        logger.info("Mining hard cases from extraction results...")
        
        # Load results
        df = pd.read_csv(output_csv)
        input_df = pd.read_csv(input_csv)
        
        # Merge to get original URLs
        merged = input_df.merge(df, on='URL', how='left')
        
        # Find failures
        failures = merged[merged['product_name'].isna()]
        
        # Group by domain to find problematic patterns
        failures['domain'] = failures['URL'].apply(lambda x: urlparse(x).netloc)
        domain_stats = failures.groupby('domain').size()
        total_by_domain = merged.groupby('domain').size()
        failure_rates = domain_stats / total_by_domain
        
        # Identify domains needing pattern updates
        problematic_domains = failure_rates[failure_rates > self.failure_threshold].index.tolist()
        
        logger.info(f"Found {len(problematic_domains)} domains with high failure rates")
        
        # Sample hard cases for retraining
        hard_cases = []
        for domain in problematic_domains:
            domain_failures = failures[failures['domain'] == domain]
            # Sample up to 1000 URLs per domain
            sample = domain_failures.sample(min(1000, len(domain_failures)))
            hard_cases.append(sample)
        
        if hard_cases:
            hard_cases_df = pd.concat(hard_cases, ignore_index=True)
            
            # Append to existing hard cases
            if os.path.exists(self.hard_cases_path):
                existing = pd.read_csv(self.hard_cases_path)
                hard_cases_df = pd.concat([existing, hard_cases_df], ignore_index=True)
                hard_cases_df = hard_cases_df.drop_duplicates(subset=['URL'])
            
            hard_cases_df.to_csv(self.hard_cases_path, index=False)
            logger.info(f"Saved {len(hard_cases_df)} hard cases to {self.hard_cases_path}")
        
        return hard_cases_df
    
    def retrain_classifier_with_hard_cases(self):
        """Retrain classifier including newly mined hard cases"""
        logger.info("Retraining classifier with hard cases...")
        
        # This would call train_slug_classifier.py with augmented data
        # For now, we'll outline the process
        
        # 1. Load original training data
        original_data = pd.read_csv('data/training_data.csv')
        
        # 2. Load hard cases and manually label a sample
        # In production, this might involve active learning or human review
        hard_cases = pd.read_csv(self.hard_cases_path)
        
        # 3. Combine datasets
        # 4. Retrain model
        # 5. Evaluate improvement
        
        logger.info("Retraining process initiated - requires manual labeling step")
    
    def update_patterns_for_failures(self, hard_cases_df: pd.DataFrame):
        """Trigger pattern re-discovery for problematic domains"""
        domains = hard_cases_df['domain'].value_counts().head(10).index.tolist()
        
        for domain in domains:
            domain_urls = hard_cases_df[hard_cases_df['domain'] == domain]['URL'].tolist()
            logger.info(f"Re-discovering patterns for {domain} with {len(domain_urls)} failed URLs")
            
            # This would call discover_patterns.py for specific domain
            # discover_patterns.main() with domain-specific URLs

def automated_improvement_pipeline():
    """Run the complete improvement pipeline"""
    logger.info("Starting automated improvement pipeline")
    
    improver = ContinuousImprover()
    
    # 1. Mine hard cases from recent extraction
    hard_cases = improver.mine_hard_cases(
        'data/urls.csv',
        'output/products_extracted.csv'
    )
    
    # 2. Update patterns for high-failure domains
    if not hard_cases.empty:
        improver.update_patterns_for_failures(hard_cases)
    
    # 3. Schedule classifier retraining (requires manual labeling)
    improver.retrain_classifier_with_hard_cases()
    
    logger.info("Improvement pipeline complete")

if __name__ == "__main__":
    automated_improvement_pipeline(), segment))),
            'entropy': self._calculate_entropy(segment)
        }
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(text)]
        return -sum(p * np.log2(p) for p in prob if p > 0)