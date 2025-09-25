#!/usr/bin/env python3
"""
Result Discovery Script for Distillation Project

This script discovers all available result combinations and creates a comprehensive CSV file
that contains every piece of information about each result in a single row.
This allows easy filtering and analysis in spreadsheet applications.

Usage:
    python discover_all_results.py --base_dir ../plots/unified --output_file all_results.csv
"""

import os
import sys
import json
import argparse
import pandas as pd
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

def discover_summary_files(base_dir: str) -> List[str]:
    """Find all plotting summary JSON files"""
    patterns = [
        "**/plotting_summary.json",
        "**/*plotting_summary*.json", 
        "**/unified_batch_plotting_*.json"
    ]
    
    summary_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        summary_files.extend(files)
 
    # Remove duplicates and sort
    summary_files = sorted(list(set(summary_files)))
    return summary_files

def extract_model_name_from_path(file_path: str) -> str:
    """Extract model name from file path"""
    filename = os.path.basename(file_path)
    # Try to extract model name from filename
    if '_' in filename:
        return filename.split('_')[0]
    else:
        return 'unknown'

def flatten_results_to_rows(summary_files: List[str], include_std: bool, include_all: bool) -> List[Dict[str, Any]]:
    """Convert all summary files into flat rows for CSV"""
    all_rows = []
    
    for file_path in summary_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            config = data.get('config', {})
            results = data.get('results', {})
            best_hyperparams = data.get('best_hyperparams', {})
            
            # Extract model name from file path
            model_name = extract_model_name_from_path(file_path)
            
            # Extract basic configuration info
            trained_dataset = config.get('trained_dataset', 'unknown')
            eval_dataset = config.get('eval_dataset', 'unknown')
            icl_source_dataset = config.get('icl_source_dataset', 'unknown')
            base_method = config.get('base_method', 'unknown')
            training_variants = config.get('training_variants', [])
            uncertainty_mode = config.get('uncertainty_mode', False)
            hp_selection = config.get('hp_selection', 'unknown')
            label_type = config.get('label_type', 'unknown')
            icl_max_demos = config.get('icl_max_demos', 'unknown')
            
            # Create triple representation
            triple = f"{trained_dataset}→{eval_dataset} (ICL: {icl_source_dataset})"
            
            # Process each training method's results
            for training_method, method_data in results.items():
                for n_val, metrics in method_data.items():
                    metrics_to_include = metrics.keys()
                    if include_all:
                        metrics_to_include = [metric for metric in metrics.keys() if metric.endswith('_all')]
                    else:
                        if include_std:
                            metrics_to_include = [metric for metric in metrics.keys() if not metric.endswith('_all')]
                        else:
                            metrics_to_include = [metric for metric in metrics.keys() if not metric.endswith('_std') and not metric.endswith('_all')]

                    metrics = {metric: metrics[metric] for metric in metrics_to_include}

                    if isinstance(n_val, (int, str)) and str(n_val).isdigit():
                        n_val = int(n_val)
                        
                        # Create a row for each metric
                        for metric_name, metric_value in metrics.items():
                            if metric_name.endswith('_all') and not include_all:
                                continue    
                            if metric_name.endswith('_std') and not include_std:
                                continue
                            if metric_value is not None:
                                row = {
                                    # File and model info
                                    # 'file_path': file_path,
                                    'model_name': model_name,
                                    
                                    # Configuration info
                                    'trained_dataset': trained_dataset,
                                    'eval_dataset': eval_dataset,
                                    'icl_source_dataset': icl_source_dataset,
                                    # 'triple': triple,
                                    'base_method': base_method,
                                    'training_method': training_method,
                                    'label_type': label_type,
                                    'uncertainty_mode': uncertainty_mode,
                                    # 'hp_selection': hp_selection,
                                    # 'icl_max_demos': icl_max_demos,
                                    # 'training_variants': ','.join(training_variants) if training_variants else 'unknown',
                                    
                                    # Result info
                                    'n_value': n_val,
                                    'metric_name': metric_name,
                                    'metric_value': metric_value,
                                    
                                    # # Additional context
                                    # 'is_base_model': training_method == 'Base Model',
                                    # 'is_with_icl': 'with_icl' in metric_name,
                                    # 'is_without_icl': 'without_icl' in metric_name,
                                    # 'metric_type': 'accuracy' if 'accuracy' in metric_name else 
                                    #              'uncertainty' if 'uncertainty' in metric_name else
                                    #              'entropy' if 'entropy' in metric_name else
                                    #              'ece' if 'ece' in metric_name else
                                    #              'other'
                                }
                                
                                # Add best hyperparameters if available
                                if training_method in best_hyperparams and n_val in best_hyperparams[training_method]:
                                    hp_info = best_hyperparams[training_method][n_val]
                                    if isinstance(hp_info, dict):
                                        for hp_key, hp_value in hp_info.items():
                                            row[f'best_hp_{hp_key}'] = hp_value
                                
                                all_rows.append(row)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return all_rows

def create_summary_statistics(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics about the discovered data"""
    if not all_rows:
        return {}
    
    df = pd.DataFrame(all_rows)
    
    summary = {
        'total_rows': len(all_rows),
        # 'unique_files': df['file_path'].nunique(),
        'unique_models': df['model_name'].nunique(),
        # 'unique_triples': df['triple'].nunique(),
        'unique_training_methods': df['training_method'].nunique(),
        'unique_metrics': df['metric_name'].nunique(),
        'unique_n_values': sorted(df['n_value'].unique().tolist()),
        'unique_base_methods': df['base_method'].unique().tolist(),
        'unique_label_types': df['label_type'].unique().tolist(),
        'uncertainty_mode_counts': df['uncertainty_mode'].value_counts().to_dict(),
        # 'hp_selection_counts': df['hp_selection'].value_counts().to_dict(),
    }
    
    return summary

def print_summary(summary: Dict[str, Any]):
    """Print summary statistics"""
    print("="*80)
    print("RESULT DISCOVERY SUMMARY")
    print("="*80)
    print(f"Total result rows: {summary.get('total_rows', 0)}")
    print(f"Unique files: {summary.get('unique_files', 0)}")
    print(f"Unique models: {summary.get('unique_models', 0)}")
    print(f"Unique triples: {summary.get('unique_triples', 0)}")
    print(f"Unique training methods: {summary.get('unique_training_methods', 0)}")
    print(f"Unique metrics: {summary.get('unique_metrics', 0)}")
    print(f"Unique N values: {summary.get('unique_n_values', [])}")
    print(f"Unique base methods: {summary.get('unique_base_methods', [])}")
    print(f"Unique label types: {summary.get('unique_label_types', [])}")
    print(f"Uncertainty mode distribution: {summary.get('uncertainty_mode_counts', {})}")
    print(f"HP selection distribution: {summary.get('hp_selection_counts', {})}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Discover all result combinations and create comprehensive CSV")
    parser.add_argument("--base_dir", type=str, default="../plots/unified",
                        help="Base directory containing plotting summary JSONs")
    parser.add_argument("--output_file", type=str, default="../plots/aggregated/all_results_list.csv",
                        help="Output CSV file path")
    parser.add_argument("--include_std", action='store_true', default=False,
                        help="Include standard deviation in CSV")
    parser.add_argument("--include_all", action='store_true', default=False,
                        help="Include all metrics in CSV")
    parser.add_argument("--print_summary", action='store_true', default=True,
                        help="Print summary statistics")
    parser.add_argument("--include_file_path", action='store_true', default=False,
                        help="Include full file paths in CSV (default: False)")
    
    args = parser.parse_args()
    
    print("Discovering all result combinations...")
    print(f"Base directory: {args.base_dir}")
    print(f"Output file: {args.output_file}")
    
    # Discover summary files
    summary_files = discover_summary_files(args.base_dir)
    print(f"Found {len(summary_files)} summary files")
    
    if not summary_files:
        print("No summary files found!")
        return
    
    # Convert to flat rows
    print("Converting results to flat rows...")
    all_rows = flatten_results_to_rows(summary_files, args.include_std, args.include_all)
    print(f"Created {len(all_rows)} result rows")
    
    if not all_rows:
        print("No results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # # Remove file path if not requested
    # if not args.include_file_path:
    #     df = df.drop('file_path', axis=1)
    
    # Sort by model, triple, training method, n_value, metric
    df = df.sort_values(['model_name', 'trained_dataset', 'eval_dataset', 'icl_source_dataset', 'base_method', 'training_method', 'n_value', 'metric_name'])
    
    # Save to CSV
    print(f"Saving to {args.output_file}...")
    df.to_csv(args.output_file, index=False)
    print(f"Saved {len(df)} rows to {args.output_file}")
    
    # Create and print summary
    if args.print_summary:
        summary = create_summary_statistics(all_rows)
        print_summary(summary)
    
    # Print column information
    print(f"\nCSV columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nFile saved: {os.path.abspath(args.output_file)}")
    print("You can now open this file in Excel, Google Sheets, or any spreadsheet application")
    print("to filter and analyze your results as needed.")

if __name__ == "__main__":
    main()
