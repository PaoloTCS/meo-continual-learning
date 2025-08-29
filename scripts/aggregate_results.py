#!/usr/bin/env python3
"""
Aggregate EWC grid search results and select best lambda.
"""

import json
import os
import glob
from typing import Dict, List, Tuple

def load_results(logdir: str = "results/logs") -> List[Dict]:
    """Load all JSON result files."""
    results = []
    
    # Find all EWC result files
    pattern = os.path.join(logdir, "ewc_seed42_lam*.json")
    files = glob.glob(pattern)
    
    for filepath in files:
        with open(filepath, 'r') as f:
            result = json.load(f)
            results.append(result)
    
    return results

def find_best_lambda(results: List[Dict]) -> Tuple[int, float]:
    """Find lambda with highest final average accuracy."""
    if not results:
        raise ValueError("No results found")
    
    best_result = max(results, key=lambda x: x['final_avg_acc'])
    return best_result['lambda'], best_result['final_avg_acc']

def print_results_table(results: List[Dict]):
    """Print results in a formatted table."""
    print("EWC Lambda Grid Search Results")
    print("=" * 40)
    print(f"{'Lambda':<8} {'Final Avg Acc':<15}")
    print("-" * 40)
    
    # Sort by lambda
    sorted_results = sorted(results, key=lambda x: x['lambda'])
    
    for result in sorted_results:
        print(f"{result['lambda']:<8} {result['final_avg_acc']:<15.2f}")
    
    print("-" * 40)

def main():
    """Main function."""
    print("Aggregating EWC grid search results...")
    
    # Load results
    results = load_results()
    
    if not results:
        print("No EWC results found in results/logs/")
        return
    
    # Print results table
    print_results_table(results)
    
    # Find best lambda
    best_lambda, best_acc = find_best_lambda(results)
    
    print(f"\nBest lambda: {best_lambda}")
    print(f"Best final average accuracy: {best_acc:.2f}%")
    
    # Save best result summary
    summary = {
        "best_lambda": best_lambda,
        "best_final_avg_acc": best_acc,
        "all_results": results
    }
    
    summary_path = "results/logs/ewc_grid_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    print(f"\nTo update the paper, run:")
    print(f"make update-tex EWC={best_acc:.1f}")

if __name__ == "__main__":
    main()
