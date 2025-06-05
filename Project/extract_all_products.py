import pandas as pd
import numpy as np
import json
import re
import joblib
from urllib.parse import urlparse, unquote
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BulkSlugExtractor:
    def __init__(self, patterns_file: str, classifier_path: str, 
                 scaler_path: str, feature_cols_path: str):
        # Load patterns
        with open(patterns_file, 'r') as f:
            self.patterns_dict = json.load(f)
        
        # Compile all regexes once
        self.compiled_patterns = {}
        for domain, patterns in self.patterns_dict.items():
            self.compiled_patterns[domain] = [re.compile(p) for p in patterns]
        
        # Load ML components
        self.classifier = joblib.load(classifier_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_cols_path, 'r') as f:
            self.feature_cols = json.load(f)
        
        self.extractor = SlugFeatureExtractor()
        self.min_ml_confidence = 0.55
    
    def extract_slug(self, url: str) -> Optional[str]:
        """Extract product slug from a single URL"""
        domain = urlparse(url).netloc
        
        # Try regex patterns first
        if domain in self.compiled_patterns:
            for pattern in self.compiled_patterns[domain]:
                match = pattern.search(url)
                if match:
                    return unquote(match.group(1))
        
        # Fall back to ML classifier
        return self._ml_extract(url)
    
    def _ml_extract(self, url: str) -> Optional[str]:
        """Extract slug using ML classifier"""
        parsed = urlparse(url)
        path_segments = [s for s in parsed.path.split('/') if s]
        
        if not path_segments:
            return None
        
        # Score all segments
        best_segment = None
        best_score = 0
        
        for segment in path_segments:
            features = self.extractor.extract_features(segment)
            X = pd.DataFrame([features])[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            
            proba = self.classifier.predict_proba(X_scaled)[0, 1]
            if proba > best_score:
                best_score = proba
                best_segment = segment
        
        if best_score >= self.min_ml_confidence:
            return unquote(best_segment)
        
        return None

def process_chunk(chunk_data: Tuple[pd.DataFrame, str, str, str, str]) -> pd.DataFrame:
    """Process a single chunk of URLs"""
    chunk, patterns_file, classifier_path, scaler_path, feature_cols_path = chunk_data
    
    # Initialize extractor for this process
    extractor = BulkSlugExtractor(
        patterns_file, classifier_path, scaler_path, feature_cols_path
    )
    
    # Extract slugs
    chunk['product_name'] = chunk['URL'].apply(extractor.extract_slug)
    
    return chunk[['URL', 'product_name']]

def parallel_extract(input_csv: str, output_csv: str, chunk_size: int = 100000, 
                    n_workers: Optional[int] = None):
    """Main parallel extraction pipeline"""
    logger.info(f"Starting extraction from {input_csv}")
    start_time = time.time()
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Count total rows
    total_rows = sum(1 for _ in open(input_csv)) - 1  # Subtract header
    logger.info(f"Total URLs to process: {total_rows:,}")
    
    # Prepare chunk arguments
    chunk_args = []
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        chunk_args.append((
            chunk,
            'config/patterns.json',
            'models/slug_classifier.joblib',
            'models/slug_scaler.joblib',
            'config/feature_cols.json'
        ))
    
    # Process chunks in parallel
    results = []
    processed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_chunk, args) for args in chunk_args]
        
        for future in futures:
            result = future.result()
            results.append(result)
            processed += len(result)
            
            # Log progress
            logger.info(f"Processed {processed:,}/{total_rows:,} URLs "
                       f"({processed/total_rows*100:.1f}%)")
    
    # Combine and save results
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    
    # Calculate performance metrics
    elapsed = time.time() - start_time
    urls_per_second = total_rows / elapsed
    
    logger.info(f"Extraction complete!")
    logger.info(f"Total time: {elapsed:.2f} seconds")
    logger.info(f"Processing speed: {urls_per_second:,.0f} URLs/second")
    logger.info(f"Output saved to: {output_csv}")
    
    # Save extraction stats
    stats = {
        'total_urls': total_rows,
        'extracted_count': final_df['product_name'].notna().sum(),
        'extraction_rate': final_df['product_name'].notna().sum() / total_rows,
        'elapsed_seconds': elapsed,
        'urls_per_second': urls_per_second,
        'n_workers': n_workers,
        'chunk_size': chunk_size
    }
    
    with open('logs/extraction_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract product slugs from URLs')
    parser.add_argument('input_csv', help='Input CSV file with URL column')
    parser.add_argument('-o', '--output', default='output/products_extracted.csv',
                       help='Output CSV file')
    parser.add_argument('-c', '--chunk-size', type=int, default=100000,
                       help='Chunk size for processing')
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    parallel_extract(args.input_csv, args.output, args.chunk_size, args.workers)

if __name__ == "__main__":
    main()