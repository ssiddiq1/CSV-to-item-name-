#!/usr/bin/env python3
"""
Performance monitoring and reporting for the slug extraction system
"""

import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, logs_dir: str = 'logs', output_dir: str = 'reports'):
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_extraction_stats(self) -> Dict:
        """Load the most recent extraction statistics"""
        stats_file = os.path.join(self.logs_dir, 'extraction_stats.json')
        if not os.path.exists(stats_file):
            logger.warning(f"No extraction stats found at {stats_file}")
            return {}
        
        with open(stats_file, 'r') as f:
            return json.load(f)
    
    def load_patterns_stats(self) -> Dict:
        """Analyze patterns.json for coverage statistics"""
        patterns_file = 'config/patterns.json'
        if not os.path.exists(patterns_file):
            return {'total_domains': 0, 'total_patterns': 0}
        
        with open(patterns_file, 'r') as f:
            patterns = json.load(f)
        
        return {
            'total_domains': len(patterns),
            'total_patterns': sum(len(p) for p in patterns.values()),
            'avg_patterns_per_domain': np.mean([len(p) for p in patterns.values()]) if patterns else 0,
            'domains': list(patterns.keys())
        }
    
    def calculate_performance_metrics(self, extraction_stats: Dict) -> Dict:
        """Calculate additional performance metrics"""
        if not extraction_stats:
            return {}
        
        metrics = {
            'extraction_rate': extraction_stats.get('extraction_rate', 0),
            'urls_per_second': extraction_stats.get('urls_per_second', 0),
            'total_urls': extraction_stats.get('total_urls', 0),
            'extracted_count': extraction_stats.get('extracted_count', 0),
            'failed_count': extraction_stats.get('total_urls', 0) - extraction_stats.get('extracted_count', 0),
            'elapsed_time': extraction_stats.get('elapsed_seconds', 0),
            'efficiency_score': extraction_stats.get('extraction_rate', 0) * 
                              min(extraction_stats.get('urls_per_second', 0) / 50000, 1)  # Normalized to 50k/s target
        }
        
        return metrics
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        logger.info("Generating performance report...")
        
        # Load data
        extraction_stats = self.load_extraction_stats()
        patterns_stats = self.load_patterns_stats()
        metrics = self.calculate_performance_metrics(extraction_stats)
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'extraction_metrics': metrics,
            'pattern_coverage': patterns_stats,
            'system_performance': {
                'urls_per_second': metrics.get('urls_per_second', 0),
                'extraction_rate': metrics.get('extraction_rate', 0),
                'efficiency_score': metrics.get('efficiency_score', 0)
            },
            'recommendations': self._generate_recommendations(metrics, patterns_stats)
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_file}")
        return report
    
    def _generate_recommendations(self, metrics: Dict, patterns_stats: Dict) -> List[str]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []
        
        # Check extraction rate
        if metrics.get('extraction_rate', 0) < 0.85:
            recommendations.append(f"Extraction rate is {metrics['extraction_rate']:.1%}. Consider running pattern discovery on failed domains.")
        
        # Check processing speed
        if metrics.get('urls_per_second', 0) < 30000:
            recommendations.append(f"Processing speed ({metrics['urls_per_second']:,.0f} URLs/s) is below target. Consider increasing worker processes.")
        
        # Check pattern coverage
        if patterns_stats['total_domains'] < 100:
            recommendations.append(f"Only {patterns_stats['total_domains']} domains have patterns. Run discovery on more domains.")
        
        # Check efficiency
        if metrics.get('efficiency_score', 0) < 0.7:
            recommendations.append("Overall efficiency is low. Review both extraction rate and processing speed.")
        
        return recommendations
    
    def plot_performance_trends(self, history_file: str = None):
        """Create performance visualization plots"""
        if not history_file:
            logger.warning("No history file provided for trend plotting")
            return
        
        # Set up plotting style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Slug Extraction System Performance', fontsize=16)
        
        # Plot 1: Extraction Rate Over Time
        ax1 = axes[0, 0]
        ax1.set_title('Extraction Rate Trend')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Extraction Rate (%)')
        
        # Plot 2: Processing Speed
        ax2 = axes[0, 1]
        ax2.set_title('Processing Speed')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('URLs per Second')
        
        # Plot 3: Pattern Coverage Growth
        ax3 = axes[1, 0]
        ax3.set_title('Pattern Coverage Growth')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Domains')
        
        # Plot 4: Efficiency Score
        ax4 = axes[1, 1]
        ax4.set_title('Overall Efficiency Score')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Efficiency (0-1)')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, f'performance_plot_{datetime.now().strftime("%Y%m%d")}.png')
        plt.savefig(plot_file)
        logger.info(f"Performance plots saved to {plot_file}")
    
    def print_summary(self):
        """Print performance summary to console"""
        stats = self.load_extraction_stats()
        patterns = self.load_patterns_stats()
        metrics = self.calculate_performance_metrics(stats)
        
        print("\n" + "="*60)
        print("SLUG EXTRACTION SYSTEM PERFORMANCE SUMMARY")
        print("="*60)
        
        print("\n📊 EXTRACTION METRICS:")
        print(f"  • Extraction Rate: {metrics.get('extraction_rate', 0):.2%}")
        print(f"  • URLs Processed: {metrics.get('total_urls', 0):,}")
        print(f"  • Successfully Extracted: {metrics.get('extracted_count', 0):,}")
        print(f"  • Failed Extractions: {metrics.get('failed_count', 0):,}")
        
        print("\n⚡ PERFORMANCE METRICS:")
        print(f"  • Processing Speed: {metrics.get('urls_per_second', 0):,.0f} URLs/second")
        print(f"  • Total Time: {metrics.get('elapsed_time', 0):.2f} seconds")
        print(f"  • Efficiency Score: {metrics.get('efficiency_score', 0):.2%}")
        
        print("\n🎯 PATTERN COVERAGE:")
        print(f"  • Total Domains: {patterns['total_domains']}")
        print(f"  • Total Patterns: {patterns['total_patterns']}")
        print(f"  • Avg Patterns/Domain: {patterns['avg_patterns_per_domain']:.1f}")
        
        print("\n💡 RECOMMENDATIONS:")
        recommendations = self._generate_recommendations(metrics, patterns)
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  ✓ System is performing optimally!")
        
        print("\n" + "="*60)

def analyze_domain_performance(output_csv: str, top_n: int = 20):
    """Analyze extraction performance by domain"""
    logger.info(f"Analyzing domain-specific performance from {output_csv}")
    
    if not os.path.exists(output_csv):
        logger.error(f"Output file {output_csv} not found")
        return
    
    # Load extraction results
    df = pd.read_csv(output_csv)
    
    # Add domain column
    from urllib.parse import urlparse
    df['domain'] = df['URL'].apply(lambda x: urlparse(x).netloc)
    
    # Calculate metrics by domain
    domain_stats = df.groupby('domain').agg({
        'product_name': [
            'count',
            lambda x: x.notna().sum(),
            lambda x: x.isna().sum(),
            lambda x: x.notna().sum() / len(x)
        ]
    }).round(4)
    
    domain_stats.columns = ['total_urls', 'extracted', 'failed', 'success_rate']
    domain_stats = domain_stats.sort_values('total_urls', ascending=False).head(top_n)
    
    print("\n📈 TOP DOMAINS BY VOLUME:")
    print(domain_stats.to_string())
    
    # Find problematic domains
    problematic = df.groupby('domain').agg({
        'product_name': lambda x: x.notna().sum() / len(x)
    })
    problematic = problematic[problematic['product_name'] < 0.8].sort_values('product_name')
    
    if not problematic.empty:
        print("\n⚠️  PROBLEMATIC DOMAINS (< 80% extraction rate):")
        print(problematic.to_string())
    
    return domain_stats

def main():
    """Main monitoring execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor slug extraction performance')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--summary', action='store_true', help='Print performance summary')
    parser.add_argument('--analyze', type=str, help='Analyze specific output CSV')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor()
    
    if args.report:
        report = monitor.generate_performance_report()
        print("Performance report generated successfully!")
    
    if args.summary or (not any(vars(args).values())):
        monitor.print_summary()
    
    if args.analyze:
        analyze_domain_performance(args.analyze)
    
    if args.plot:
        monitor.plot_performance_trends()

if __name__ == "__main__":
    main()