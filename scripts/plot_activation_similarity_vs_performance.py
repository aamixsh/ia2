#!/usr/bin/env python3
"""
Activation Similarity vs Performance Scatter Plot Script

This script creates scatter plots showing the relationship between activation similarity
and performance metrics (accuracy, ECE) for different training methods.

Features:
- Loads activation similarity results from activation similarity analysis summaries
- Loads performance metrics from aggregated results CSV
- Creates scatter plots for activation similarity vs accuracy (single and multi-token)
- Creates scatter plots for activation similarity vs ECE (single-token only)
- Uses different markers and colors for different training methods
- Separates plots by single-token (uncertainty mode) and multi-token scenarios
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import pickle
from scipy import stats
from sklearn.mixture import GaussianMixture

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Set style for better plots
plt.style.use('seaborn-v0_8')


def load_activation_similarity_results(activation_similarity_dir: str) -> Dict[str, Any]:
    """Load activation similarity results from summary files"""
    print("Loading activation similarity results...")
    
    # Find all activation similarity summary files
    summary_files = glob.glob(os.path.join(activation_similarity_dir, "*activation_similarity_summary*.json"))
    
    if not summary_files:
        print(f"No activation similarity summary files found in {activation_similarity_dir}")
        return {}
    
    print(f"Found {len(summary_files)} activation similarity summary files")
    
    all_similarity_data = {}
    
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            config = data.get('config', {})
            completed_configs = data.get('completed_configurations', [])
            
            # Extract key configuration parameters
            model_id = config.get('model_id', 'unknown')
            trained_datasets = config.get('trained_datasets', [])
            eval_datasets = config.get('eval_datasets', [])
            label_types = config.get('label_types', [])
            icl_sources = config.get('icl_sources', [])
            num_generated_tokens = config.get('num_generated_tokens', 1)
            uncertainty = config.get('uncertainty', False)
            
            print(f"Processing {os.path.basename(summary_file)}: {model_id}, tokens={num_generated_tokens}, uncertainty={uncertainty}")
            
            # Load similarity results for each completed configuration
            for config_info in completed_configs:
                eval_dataset = config_info['eval_dataset']
                trained_dataset = config_info['trained_dataset']
                label_type = config_info['label_type']
                icl_source = config_info['icl_source']
                num_train_examples = config_info['num_train_examples']
                run_idx = config_info['run_idx']
                
                # Create a unique key for this configuration
                config_key = (model_id, trained_dataset, eval_dataset, icl_source, label_type, 
                             num_train_examples, run_idx, num_generated_tokens, uncertainty)

                # Load similarity results from pickle files
                similarity_paths = config_info.get('similarity_paths', {})
                similarity_results = {}
                
                for model_type in ['tok', 'act', 'tna', 'a2t']:
                    if model_type in similarity_paths:
                        similarity_file = similarity_paths[model_type]
                        similarity_file = similarity_file.replace(f'/{model_type}/', '/')
                        
                        if os.path.exists(similarity_file):
                            try:
                                with open(similarity_file, 'rb') as f:
                                    similarities = pickle.load(f)

                                # Extract overall similarity (mean across examples)
                                if similarities and 'overall' in similarities and similarities['overall']:
                                    similarity_results[model_type] = np.mean(similarities['overall'])
                                else:
                                    similarity_results[model_type] = None
                            except Exception as e:
                                print(f"Error loading similarity file {similarity_file}: {e}")
                                similarity_results[model_type] = None
                        else:
                            similarity_results[model_type] = None
                    else:
                        similarity_results[model_type] = None
                
                all_similarity_data[config_key] = similarity_results
                
        except Exception as e:
            print(f"Error processing {summary_file}: {e}")
            continue
    
    print(f"Loaded similarity data for {len(all_similarity_data)} configurations")
    return all_similarity_data


def load_performance_metrics(csv_file: str) -> pd.DataFrame:
    """Load performance metrics from aggregated results CSV"""
    print(f"Loading performance metrics from {csv_file}...")
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} performance metric records")
    
    # Print available columns and sample data
    print(f"Available columns: {list(df.columns)}")
    print(f"Sample data:")
    print(df.head())
    
    return df


def match_similarity_with_performance(similarity_data: Dict[str, Any], 
                                    performance_df: pd.DataFrame,
                                    accuracy_metric: str = 'top1',
                                    base_method: str = 'none') -> List[Dict[str, Any]]:
    """Match activation similarity results with performance metrics"""
    print("Matching similarity data with performance metrics...")
    
    matched_data = []
    
    for config_key, similarity_results in similarity_data.items():
        model_id, trained_dataset, eval_dataset, icl_source, label_type, \
        num_train_examples, run_idx, num_generated_tokens, uncertainty = config_key
        
        # Extract model name from model_id
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        # Filter performance data for this configuration
        perf_filter = (
            (performance_df['model_name'] == model_name) &
            (performance_df['trained_dataset'] == trained_dataset) &
            (performance_df['eval_dataset'] == eval_dataset) &
            (performance_df['icl_source_dataset'] == icl_source) &
            (performance_df['label_type'] == label_type) &
            (performance_df['n_value'] == num_train_examples) &
            (performance_df['uncertainty_mode'] == uncertainty) &
            (performance_df['base_method'] == base_method if base_method != 'none' else True)
        )

        perf_data = performance_df[perf_filter]
        
        if perf_data.empty:
            print(f"No performance data found for {config_key}")
            continue
        
        # Extract performance metrics for each training method
        for model_type in ['tok', 'act', 'tna', 'a2t']:
            if similarity_results.get(model_type) is not None:
                # Map model types to training method names in performance data
                training_method_map = {
                    'tok': 'lora Token Training',
                    'act': 'lora Activation Training', 
                    'tna': 'lora Token + Activation Training',
                    'a2t': 'lora Sequential (act→tok) Training'
                }
                
                training_method = training_method_map.get(model_type)
                if not training_method:
                    continue
                
                # Filter for this specific training method
                method_perf = perf_data[perf_data['training_method'] == training_method]
                
                if method_perf.empty:
                    continue
                
                # Extract accuracy and ECE metrics
                accuracy_data = method_perf[method_perf['metric_name'].str.contains(f'without_icl_accuracy_{accuracy_metric}_mean', na=False)]
                ece_data = method_perf[method_perf['metric_name'].str.contains(f'without_icl_ece_{accuracy_metric.split('_')[0]}_mean', na=False)]

                accuracy_value = None
                ece_value = None


                # Get the most relevant accuracy metric based on user preference
                if not accuracy_data.empty:
                    accuracy_value = accuracy_data['metric_value'].iloc[0]
                if not ece_data.empty:
                    ece_value = ece_data['metric_value'].iloc[0]

                # accuracy_value = None
                # if not accuracy_data.empty:
                #     if accuracy_metric == 'label_set':
                #         # Prefer label_set_mean if available, otherwise use top1_mean
                #         if 'without_icl_accuracy_label_set_mean' in accuracy_data['metric_name'].values:
                #             accuracy_value = accuracy_data[accuracy_data['metric_name'] == 'without_icl_accuracy_label_set_mean']['metric_value'].iloc[0]
                #         elif 'without_icl_accuracy_top1_mean' in accuracy_data['metric_name'].values:
                #             accuracy_value = accuracy_data[accuracy_data['metric_name'] == 'without_icl_accuracy_top1_mean']['metric_value'].iloc[0]
                #         else:
                #             # Take the first available accuracy metric
                #             accuracy_value = accuracy_data['metric_value'].iloc[0]
                #     elif accuracy_metric == 'top1':
                #         # Prefer top1_mean if available, otherwise use label_set_mean
                #         if 'without_icl_accuracy_top1_mean' in accuracy_data['metric_name'].values:
                #             accuracy_value = accuracy_data[accuracy_data['metric_name'] == 'without_icl_accuracy_top1_mean']['metric_value'].iloc[0]
                #         elif 'without_icl_accuracy_label_set_mean' in accuracy_data['metric_name'].values:
                #             accuracy_value = accuracy_data[accuracy_data['metric_name'] == 'without_icl_accuracy_label_set_mean']['metric_value'].iloc[0]
                #         else:
                #             # Take the first available accuracy metric
                #             accuracy_value = accuracy_data['metric_value'].iloc[0]
                #     else:
                #         # Take the first available accuracy metric
                #         accuracy_value = accuracy_data['metric_value'].iloc[0]
                
                # # Get ECE value (only for single-token/uncertainty mode)
                # ece_value = None
                # if not ece_data.empty and uncertainty:
                #     ece_value = ece_data['metric_value'].iloc[0]
                
                if accuracy_value is not None:
                    matched_data.append({
                        'model_name': model_name,
                        'trained_dataset': trained_dataset,
                        'eval_dataset': eval_dataset,
                        'icl_source': icl_source,
                        'label_type': label_type,
                        'num_train_examples': num_train_examples,
                        'run_idx': run_idx,
                        'num_generated_tokens': num_generated_tokens,
                        'uncertainty': uncertainty,
                        'model_type': model_type,
                        'training_method': training_method,
                        'activation_similarity': similarity_results[model_type],
                        'accuracy': accuracy_value,
                        'ece': ece_value
                    })
    
    print(f"Matched {len(matched_data)} similarity-performance pairs")
    return matched_data


def fit_gaussian_and_plot_contours(ax, data, color, alpha=0.3, levels=[0.5, 0.8, 0.95], y_column='accuracy'):
    """Fit a Gaussian distribution to data and plot contours"""
    if len(data) < 3:  # Need at least 3 points for Gaussian fitting
        return None
    
    x_data = data['activation_similarity'].values
    y_data = data[y_column].values
    
    # Fit a 2D Gaussian using Gaussian Mixture Model with 1 component
    try:
        gmm = GaussianMixture(n_components=1, random_state=42)
        gmm.fit(np.column_stack([x_data, y_data]))
        
        # Get the mean of the fitted Gaussian
        mean_x, mean_y = gmm.means_[0]
        
        # Create a grid for contour plotting
        x_range_factor = 0.05
        y_range_factor = (y_data.max() - y_data.min()) * 0.05
        x_min, x_max = x_data.min() - x_range_factor, x_data.max() + x_range_factor
        y_min, y_max = y_data.min() - y_range_factor, y_data.max() + y_range_factor
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Calculate the probability density
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        zz = gmm.score_samples(grid_points)
        zz = np.exp(zz).reshape(xx.shape)
        
        # Normalize to get proper probability contours
        zz = zz / zz.max()
        
        # Plot contours
        contour = ax.contour(xx, yy, zz, levels=levels, colors=[color], 
                           alpha=alpha, linewidths=1.5, linestyles='--')
        
        # Fill contours with lighter color
        ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=alpha*0.3)
        
        # Return the mean coordinates for highlighting
        return (mean_x, mean_y)
        
    except Exception as e:
        print(f"Warning: Could not fit Gaussian for data with {len(data)} points: {e}")
        return None


def create_plot_for_label_type(ax, data, method_styles, accuracy_metric, y_column='accuracy', title_suffix=''):
    """Create a single plot for a specific label type"""
    if data.empty:
        return
    
    # Collect Gaussian means for highlighting
    gaussian_means = {}
    
    # First, plot Gaussian contours for each method
    for model_type in ['tok', 'act', 'tna', 'a2t']:
        method_data = data[data['model_type'] == model_type]
        if not method_data.empty and len(method_data) >= 3:
            style = method_styles[model_type]
            mean_coords = fit_gaussian_and_plot_contours(ax, method_data, style['color'], y_column=y_column)
            if mean_coords:
                gaussian_means[model_type] = mean_coords
    
    # Then, plot scatter points (lighter)
    for model_type in ['tok', 'act', 'tna', 'a2t']:
        method_data = data[data['model_type'] == model_type]
        if not method_data.empty:
            style = method_styles[model_type]
            ax.scatter(method_data['activation_similarity'], method_data[y_column],
                      c=style['color'], marker=style['marker'], s=10, alpha=0.4,
                      label=style['label'], edgecolors='black', linewidth=0.3)
    
    # Highlight Gaussian means with big markers
    for model_type, (mean_x, mean_y) in gaussian_means.items():
        style = method_styles[model_type]
        ax.scatter(mean_x, mean_y, c=style['color'], marker=style['marker'], 
                  s=200, alpha=0.9, edgecolors='black', linewidth=2, zorder=10)


def create_scatter_plots(matched_data: List[Dict[str, Any]], output_dir: str, accuracy_metric: str = 'label_set'):
    """Create scatter plots for activation similarity vs performance metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not matched_data:
        print("No matched data available for plotting")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(matched_data)
    
    # Define colors and markers for different training methods
    method_styles = {
        'tok': {'color': '#1f77b4', 'marker': 'o', 'label': 'SFT only'},
        'act': {'color': '#ff7f0e', 'marker': 's', 'label': 'IA2 only'},
        'tna': {'color': '#2ca02c', 'marker': '^', 'label': 'IA2 + SFT'},
        'a2t': {'color': '#d62728', 'marker': 'D', 'label': 'IA2 → SFT'}
    }
    
    # Separate data by token type, uncertainty, and label_type
    single_token_data = df[(df['num_generated_tokens'] == 1) & (df['uncertainty'] == True)]
    multi_token_data = df[df['num_generated_tokens'] > 1]
    
    # Separate by label_type
    single_token_ground_truth = single_token_data[single_token_data['label_type'] == 'ground_truth']
    single_token_icl_outputs = single_token_data[single_token_data['label_type'] == 'icl_outputs']
    multi_token_ground_truth = multi_token_data[multi_token_data['label_type'] == 'ground_truth']
    multi_token_icl_outputs = multi_token_data[multi_token_data['label_type'] == 'icl_outputs']

    print(f"Single-token data points: {len(single_token_data)}")
    print(f"  - Ground truth: {len(single_token_ground_truth)}")
    print(f"  - ICL outputs: {len(single_token_icl_outputs)}")
    print(f"Multi-token data points: {len(multi_token_data)}")
    print(f"  - Ground truth: {len(multi_token_ground_truth)}")
    print(f"  - ICL outputs: {len(multi_token_icl_outputs)}")
    
    # ============ SINGLE-TOKEN PLOTS (with uncertainty) ============
    if not single_token_data.empty:
        print("Creating single-token plots...")
        
        # Plot 1: Activation Similarity vs Accuracy (Single-token) - Ground Truth
        if not single_token_ground_truth.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            create_plot_for_label_type(ax, single_token_ground_truth, method_styles, accuracy_metric, 'accuracy')
            
            # ax.set_xlabel('Activation Similarity (Cosine Similarity)', fontsize=12)
            ax.set_xlabel('Activation Similarity', fontsize=12)
            # ax.set_ylabel(f'Accuracy ({accuracy_metric}) (%)', fontsize=12)
            ax.set_ylabel(f'Accuracy', fontsize=12)
            # ax.set_title(f'Activation Similarity vs Accuracy ({accuracy_metric}) (Single-token, Ground Truth)', fontsize=14)
            # ax.set_title(f'Activation Similarity vs Accuracy', fontsize=14)
            ax.legend(fontsize=10, facecolor='white', framealpha=0.7)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_single_token_ground_truth.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_single_token_ground_truth.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved single-token accuracy plot (ground truth): {plot_path}")
        
        # Plot 2: Activation Similarity vs Accuracy (Single-token) - ICL Outputs
        if not single_token_icl_outputs.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            create_plot_for_label_type(ax, single_token_icl_outputs, method_styles, accuracy_metric, 'accuracy')
            
            ax.set_xlabel('Activation Similarity (Cosine Similarity)', fontsize=12)
            ax.set_ylabel(f'Accuracy ({accuracy_metric}) (%)', fontsize=12)
            ax.set_title(f'Activation Similarity vs Accuracy ({accuracy_metric}) (Single-token, ICL Outputs)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_single_token_icl_outputs.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_single_token_icl_outputs.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved single-token accuracy plot (ICL outputs): {plot_path}")
        
        # Plot 3: Activation Similarity vs ECE (Single-token) - Ground Truth
        ece_ground_truth = single_token_ground_truth[single_token_ground_truth['ece'].notna()]
        if not ece_ground_truth.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            create_plot_for_label_type(ax, ece_ground_truth, method_styles, accuracy_metric, 'ece')
            
            # ax.set_xlabel('Activation Similarity (Cosine Similarity)', fontsize=12)
            ax.set_xlabel('Activation Similarity', fontsize=12)
            ax.set_ylabel('Expected Calibration Error', fontsize=12)
            # ax.set_title('Activation Similarity vs ECE (Single-token)', fontsize=14)
            # ax.legend(fontsize=10, facecolor='white', framealpha=0.7)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "activation_similarity_vs_ece_single_token_ground_truth.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path = os.path.join(output_dir, "activation_similarity_vs_ece_single_token_ground_truth.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved single-token ECE plot (ground truth): {plot_path}")
        
        # Plot 4: Activation Similarity vs ECE (Single-token) - ICL Outputs
        ece_icl_outputs = single_token_icl_outputs[single_token_icl_outputs['ece'].notna()]
        if not ece_icl_outputs.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            create_plot_for_label_type(ax, ece_icl_outputs, method_styles, accuracy_metric, 'ece')
            
            ax.set_xlabel('Activation Similarity (Cosine Similarity)', fontsize=12)
            ax.set_ylabel('ECE (Expected Calibration Error)', fontsize=12)
            ax.set_title('Activation Similarity vs ECE (Single-token, ICL Outputs)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "activation_similarity_vs_ece_single_token_icl_outputs.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path = os.path.join(output_dir, "activation_similarity_vs_ece_single_token_icl_outputs.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved single-token ECE plot (ICL outputs): {plot_path}")
        
        if ece_ground_truth.empty and ece_icl_outputs.empty:
            print("No ECE data available for single-token plots")
    
    # ============ MULTI-TOKEN PLOTS ============
    if not multi_token_data.empty:
        print("Creating multi-token plots...")
        
        # Plot 5: Activation Similarity vs Accuracy (Multi-token) - Ground Truth
        if not multi_token_ground_truth.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            create_plot_for_label_type(ax, multi_token_ground_truth, method_styles, accuracy_metric, 'accuracy')
            
            ax.set_xlabel('Activation Similarity (Cosine Similarity)', fontsize=12)
            ax.set_ylabel(f'Accuracy ({accuracy_metric}) (%)', fontsize=12)
            ax.set_title(f'Activation Similarity vs Accuracy ({accuracy_metric}) (Multi-token, Ground Truth)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_multi_token_ground_truth.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_multi_token_ground_truth.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved multi-token accuracy plot (ground truth): {plot_path}")
        
        # Plot 6: Activation Similarity vs Accuracy (Multi-token) - ICL Outputs
        if not multi_token_icl_outputs.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            create_plot_for_label_type(ax, multi_token_icl_outputs, method_styles, accuracy_metric, 'accuracy')
            
            ax.set_xlabel('Activation Similarity (Cosine Similarity)', fontsize=12)
            ax.set_ylabel(f'Accuracy ({accuracy_metric}) (%)', fontsize=12)
            ax.set_title(f'Activation Similarity vs Accuracy ({accuracy_metric}) (Multi-token, ICL Outputs)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_multi_token_icl_outputs.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_multi_token_icl_outputs.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved multi-token accuracy plot (ICL outputs): {plot_path}")
    
    # ============ COMBINED PLOTS (All data) ============
    if not df.empty:
        print("Creating combined plots...")
        
        # Plot 7: Activation Similarity vs Accuracy (All data) - Ground Truth
        ground_truth_data = df[df['label_type'] == 'ground_truth']
        if not ground_truth_data.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            create_plot_for_label_type(ax, ground_truth_data, method_styles, accuracy_metric, 'accuracy')
            
            ax.set_xlabel('Activation Similarity (Cosine Similarity)', fontsize=12)
            ax.set_ylabel(f'Accuracy ({accuracy_metric}) (%)', fontsize=12)
            ax.set_title(f'Activation Similarity vs Accuracy ({accuracy_metric}) (All Data, Ground Truth)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_all_ground_truth.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_all_ground_truth.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined accuracy plot (ground truth): {plot_path}")
        
        # Plot 8: Activation Similarity vs Accuracy (All data) - ICL Outputs
        icl_outputs_data = df[df['label_type'] == 'icl_outputs']
        if not icl_outputs_data.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            create_plot_for_label_type(ax, icl_outputs_data, method_styles, accuracy_metric, 'accuracy')
            
            ax.set_xlabel('Activation Similarity (Cosine Similarity)', fontsize=12)
            ax.set_ylabel(f'Accuracy ({accuracy_metric}) (%)', fontsize=12)
            ax.set_title(f'Activation Similarity vs Accuracy ({accuracy_metric}) (All Data, ICL Outputs)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_all_icl_outputs.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path = os.path.join(output_dir, "activation_similarity_vs_accuracy_all_icl_outputs.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined accuracy plot (ICL outputs): {plot_path}")


def filter_data(matched_data: List[Dict[str, Any]], 
                models: List[str] = None,
                datasets: List[str] = None,
                training_methods: List[str] = None) -> List[Dict[str, Any]]:
    """Filter matched data based on specified criteria"""
    if not matched_data:
        return matched_data
    
    df = pd.DataFrame(matched_data)
    original_count = len(df)
    
    # Apply filters
    if models:
        df = df[df['model_name'].isin(models)]
        print(f"Filtered by models {models}: {len(df)} remaining (from {original_count})")
    
    if datasets:
        df = df[df['eval_dataset'].isin(datasets)]
        print(f"Filtered by datasets {datasets}: {len(df)} remaining")
    
    if training_methods:
        df = df[df['model_type'].isin(training_methods)]
        print(f"Filtered by training methods {training_methods}: {len(df)} remaining")
    
    return df.to_dict('records')


def print_available_options(matched_data: List[Dict[str, Any]]):
    """Print available options for filtering"""
    if not matched_data:
        print("No data available to show options")
        return
    
    df = pd.DataFrame(matched_data)
    
    print("\n" + "="*80)
    print("AVAILABLE FILTERING OPTIONS")
    print("="*80)
    
    print(f"Available models: {sorted(df['model_name'].unique())}")
    print(f"Available datasets: {sorted(df['eval_dataset'].unique())}")
    print(f"Available training methods: {sorted(df['model_type'].unique())}")
    print(f"Available token types: {sorted(df['num_generated_tokens'].unique())}")
    print(f"Available uncertainty modes: {sorted(df['uncertainty'].unique())}")


def print_data_summary(matched_data: List[Dict[str, Any]]):
    """Print summary statistics of the matched data"""
    if not matched_data:
        print("No matched data available")
        return
    
    df = pd.DataFrame(matched_data)
    
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    
    print(f"Total data points: {len(df)}")
    print(f"Models: {df['model_name'].unique()}")
    print(f"Training methods: {df['model_type'].unique()}")
    print(f"Token types: {df['num_generated_tokens'].unique()}")
    print(f"Uncertainty modes: {df['uncertainty'].unique()}")
    
    print(f"\nData points by training method:")
    for model_type in ['tok', 'act', 'tna', 'a2t']:
        count = len(df[df['model_type'] == model_type])
        print(f"  {model_type}: {count}")
    
    print(f"\nData points by token type:")
    single_token = len(df[(df['num_generated_tokens'] == 1) & (df['uncertainty'] == True)])
    multi_token = len(df[df['num_generated_tokens'] > 1])
    print(f"  Single-token (uncertainty): {single_token}")
    print(f"  Multi-token: {multi_token}")
    
    print(f"\nActivation similarity statistics:")
    print(f"  Mean: {df['activation_similarity'].mean():.4f}")
    print(f"  Std: {df['activation_similarity'].std():.4f}")
    print(f"  Min: {df['activation_similarity'].min():.4f}")
    print(f"  Max: {df['activation_similarity'].max():.4f}")
    
    print(f"\nAccuracy statistics:")
    print(f"  Mean: {df['accuracy'].mean():.2f}")
    print(f"  Std: {df['accuracy'].std():.2f}")
    print(f"  Min: {df['accuracy'].min():.2f}")
    print(f"  Max: {df['accuracy'].max():.2f}")
    
    ece_data = df[df['ece'].notna()]
    if not ece_data.empty:
        print(f"\nECE statistics (single-token only):")
        print(f"  Mean: {ece_data['ece'].mean():.4f}")
        print(f"  Std: {ece_data['ece'].std():.4f}")
        print(f"  Min: {ece_data['ece'].min():.4f}")
        print(f"  Max: {ece_data['ece'].max():.4f}")
    
    # Correlation analysis
    print(f"\nCorrelations:")
    print(f"  Activation Similarity vs Accuracy: {df['activation_similarity'].corr(df['accuracy']):.4f}")
    if not ece_data.empty:
        print(f"  Activation Similarity vs ECE: {ece_data['activation_similarity'].corr(ece_data['ece']):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Create scatter plots for activation similarity vs performance")
    
    # Data sources
    parser.add_argument("--activation_similarity_dir", type=str, 
                       default="../plots/activation_similarity",
                       help="Directory containing activation similarity summary files")
    parser.add_argument("--performance_csv", type=str,
                       default="../plots/aggregated/all_results.csv", 
                       help="CSV file containing performance metrics")
    
    # Accuracy metric selection
    parser.add_argument("--accuracy_metric", type=str, choices=['label_set', 'top1'], 
                       default='top1',
                       help="Accuracy metric to use: 'label_set' or 'top1'")

    # base_method = parser.add_argument("--base_method", type=str, choices=['lora', 'ia3', 'prompt', 'prefix'], default=None,
    base_method = parser.add_argument("--base_method", type=str, choices=['lora', 'ia3', 'prompt', 'prefix'], default='lora',
                        help="Base method to use (lora, ia3, prompt, prefix)")
    
    # Filtering options
    parser.add_argument("--models", nargs='+', type=str, default=None,
    # parser.add_argument("--models", nargs='+', type=str, default=['Qwen3-4B-Base'],
    # parser.add_argument("--models", nargs='+', type=str, default=['Qwen3-4B-Base', 'Llama-3.2-1B-Instruct'],
                       help="Filter by specific models (e.g., Llama-3.2-1B Qwen3-4B-Base)")
    # parser.add_argument("--datasets", nargs='+', type=str, default=None,
    # parser.add_argument("--datasets", nargs='+', type=str, default=['sst2', 'finsen', 'poems', 'sciq_remap', 'qasc_remap', 'strategytf'],
    parser.add_argument("--datasets", nargs='+', type=str, default=['gsm8k', 'sciqa', 'hmath_algebra'],
                       help="Filter by specific evaluation datasets (e.g., sst2 agnews)")
    parser.add_argument("--training_methods", nargs='+', type=str, 
                    #    choices=['tok', 'act', 'a2t', 'tna'], default=None,
                       choices=['tok', 'act', 'a2t', 'tna'], default=['tok', 'act', 'a2t'],
                       help="Filter by specific training methods")
    
    # Output
    parser.add_argument("--output_dir", type=str, 
                       default="../plots/activation_similarity_vs_performance",
                       help="Output directory for scatter plots")
    parser.add_argument("--show_options_only", action="store_true",
                       help="Only show available filtering options, don't create plots")
    
    args = parser.parse_args()
    
    print("="*80)
    print("ACTIVATION SIMILARITY VS PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Load activation similarity results
    similarity_data = load_activation_similarity_results(args.activation_similarity_dir)
    
    if not similarity_data:
        print("No activation similarity data found!")
        return
    
    # Load performance metrics
    performance_df = load_performance_metrics(args.performance_csv)
    
    if performance_df.empty:
        print("No performance data found!")
        return
    
    # Match similarity data with performance metrics
    matched_data = match_similarity_with_performance(similarity_data, performance_df, args.accuracy_metric, args.base_method)
    
    if not matched_data:
        print("No matching data found between similarity and performance!")
        return
    
    # Apply filters if specified
    if args.models or args.datasets or args.training_methods or args.base_method:
        print(f"\nApplying filters...")
        matched_data = filter_data(matched_data, args.models, args.datasets, args.training_methods)
        
        if not matched_data:
            print("No data remaining after filtering!")
            return
    
    # Print available options and data summary
    print_available_options(matched_data)
    print_data_summary(matched_data)
    
    if args.show_options_only:
        print("\nOptions displayed. Use --help to see filtering options.")
        return
    
    # Create scatter plots
    print(f"\nCreating scatter plots...")
    create_scatter_plots(matched_data, args.output_dir, args.accuracy_metric)
    
    # Save matched data for further analysis
    matched_df = pd.DataFrame(matched_data)
    output_csv = os.path.join(args.output_dir, "matched_similarity_performance_data.csv")
    matched_df.to_csv(output_csv, index=False)
    print(f"Saved matched data to: {output_csv}")
    
    print(f"\nAnalysis complete! Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
