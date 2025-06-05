import os
import sys
import logging
import argparse
from datetime import datetime
import subprocess
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Create necessary directories"""
    dirs = ['models', 'config', 'data', 'logs', 'output']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def check_prerequisites():
    """Check if required files exist"""
    required = {
        'training_data': 'data/training_data.csv',
        'patterns': 'config/patterns.json'
    }
    
    missing = []
    for name, path in required.items():
        if not os.path.exists(path):
            missing.append(f"{name} ({path})")
    
    if missing:
        logger.warning(f"Missing files: {', '.join(missing)}")
        
        # Create empty patterns.json if missing
        if not os.path.exists('config/patterns.json'):
            with open('config/patterns.json', 'w') as f:
                json.dump({}, f)
            logger.info("Created empty patterns.json")

def run_step(name: str, command: List[str]):
    """Run a pipeline step"""
    logger.info(f"Running {name}...")
    start = datetime.now()
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"{name} failed: {result.stderr}")
        sys.exit(1)
    
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"{name} completed in {elapsed:.1f}s")

def main():
    parser = argparse.ArgumentParser(description='Run complete slug extraction pipeline')
    parser.add_argument('input_csv', help='Input CSV with URLs')
    parser.add_argument('--train', action='store_true', help='Train classifier first')
    parser.add_argument('--discover', action='store_true', help='Run pattern discovery')
    parser.add_argument('--extract', action='store_true', help='Run extraction')
    parser.add_argument('--improve', action='store_true', help='Run improvement loop')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    # Setup
    ensure_directories()
    check_prerequisites()
    
    # Determine which steps to run
    if args.all:
        args.train = args.discover = args.extract = args.improve = True
    
    # Execute pipeline
    if args.train:
        run_step("Classifier Training", 
                ["python", "train_slug_classifier.py"])
    
    if args.discover:
        run_step("Pattern Discovery",
                ["python", "discover_patterns.py", args.input_csv])
    
    if args.extract:
        run_step("Bulk Extraction",
                ["python", "extract_all_products.py", args.input_csv])
    
    if args.improve:
        run_step("Continuous Improvement",
                ["python", "continuous_improve.py"])
    
    logger.info("Pipeline complete!")

if __name__ == "__main__":
    main()