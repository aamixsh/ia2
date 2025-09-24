#!/usr/bin/env python3
"""
Unified Plotting Script for Distillation Project

This script creates comprehensive plots for all evaluation metrics:
- Accuracy (top1, label_set)
- Uncertainty (top1, label_set) 
- Entropy (top1, label_set)
- ECE (top1, label_set)

Supports two hyperparameter selection methods:
a) Best mean performance (current method)
b) Best mean dev loss (loads final_loss_statistics.json from adapter directories)

Works with unified naming conventions: tok, act, tna, base, a2t, t2a
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
from pathlib import Path
import itertools

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')


def load_evaluation_results(results_path):
    """Load evaluation results from JSON file"""
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {results_path}: {e}")
        return None


def load_dev_loss_statistics(model_type, trained_dataset, model_id, lora_type, lora_r, lora_alpha, 
                           num_generated_tokens_train, num_train_examples, lr, run_idx, 
                           ce_loss_weight=None, label_type=None, ia3_type=None, num_virtual_tokens=None,
                           ldr_mode=False, num_labelled_samples=None, num_unlabelled_samples=None, max_permutations=None):
    """Load dev loss statistics from adapter directory"""
    model_name_base = model_id.split('/')[-1]
    
    # Parse model_type to extract base method and training variant
    if '-' in model_type:
        base_method, training_variant = model_type.split('-', 1)
    else:
        base_method = 'lora'
        training_variant = model_type

    
    # Construct adapter path based on model type
    if base_method == 'lora' or model_type in ['tok', 'act', 'tna', 'a2t', 't2a']:
        base_name = f"../outputs/{training_variant}/{trained_dataset}/{model_name_base}_{lora_type}_{lora_r}_{lora_alpha}_{num_generated_tokens_train}_{num_train_examples}_{lr}_{run_idx}"
        
        if training_variant in ['tok', 'a2t']:
            adapter_name = f"{base_name}_{label_type}"
        elif training_variant in ['act', 't2a']:
            adapter_name = base_name
        elif training_variant == 'tna':
            adapter_name = f"{base_name}_{ce_loss_weight}"
    elif base_method == 'ia3':
        base_name = f"../outputs/{model_type}/{trained_dataset}/{model_name_base}_ia3_{ia3_type}_{num_generated_tokens_train}_{num_train_examples}_{lr}_{run_idx}"
        if training_variant in ['tok', 'a2t']:
            adapter_name = f"{base_name}_{label_type}"
        elif training_variant in ['act', 't2a']:
            adapter_name = base_name
        elif training_variant == 'tna':
            adapter_name = f"{base_name}_{ce_loss_weight}"
    elif base_method in ['prompt', 'prefix']:
        base_name = f"../outputs/{model_type}/{trained_dataset}/{model_name_base}_{base_method}_{num_virtual_tokens}_{num_generated_tokens_train}_{num_train_examples}_{lr}_{run_idx}"
        if training_variant in ['tok', 'a2t']:
            adapter_name = f"{base_name}_{label_type}"
        elif training_variant in ['act', 't2a']:
            adapter_name = base_name
        elif training_variant == 'tna':
            adapter_name = f"{base_name}_{ce_loss_weight}"
    else:
        return None
    
    # Load final_loss_statistics.json
    stats_path = os.path.join(adapter_name, "final_loss_statistics.json")
    
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            return stats.get('best_dev_loss')
        except Exception as e:
            print(f"Error loading dev loss from {stats_path}: {e}")
            return None
    else:
        print(f"Dev loss file not found: {stats_path}")
        return None


def find_evaluation_results(base_dir, model_type, trained_dataset, eval_dataset, 
                          icl_source, icl_demos, uncertainty=False):
    """Find evaluation result files for a specific configuration"""
    # Construct search directory
    suffix = "_uncertainty" if uncertainty else ""
    if model_type == 'base':
        search_dir = os.path.join(base_dir, f"base{suffix}", eval_dataset)
    else:
        # Parse model_type to extract base method and training variant
        if '-' in model_type:
            base_method, training_variant = model_type.split('-', 1)
            if base_method == 'lora':
                search_dir = os.path.join(base_dir, f"{training_variant}{suffix}", trained_dataset)
            else:
                search_dir = os.path.join(base_dir, f"{model_type}{suffix}", trained_dataset)
        else:
            # Legacy lora format
            search_dir = os.path.join(base_dir, f"{model_type}{suffix}", trained_dataset)
    
    if not os.path.exists(search_dir):
        return []
    
    # Pattern for result files
    if model_type == 'base':
        pattern_parts = [
            "*",  # model name
            'base',
            f"on_{icl_source}",
            f"demos*"
        ]
    else:
        pattern_parts = [
            "*",  # model name
            eval_dataset,
            f"on_{icl_source}",
            f"demos*",
        ]
    pattern = "_".join(pattern_parts) + ".json"
    search_path = os.path.join(search_dir, pattern)

    results = []
    for file_path in glob.glob(search_path):
        data = load_evaluation_results(file_path)
        if data:
            # Add metadata
            data['file_path'] = file_path
            data['model_type'] = model_type
            results.append(data)
    
    return results


def extract_hyperparameters(result_data):
    """Extract hyperparameters from result data"""
    params = result_data.get('trained_model_params', {})
    return {
        'lora_r': params.get('lora_r'),
        'lora_alpha': params.get('lora_alpha'), 
        'lr': params.get('lr'),
        'num_train_examples': params.get('num_train_examples'),
        'num_generated_tokens_train': params.get('num_generated_tokens_train'),
        'label_type': params.get('label_type'),
        'ce_loss_weight': params.get('ce_loss_weight'),
        'run_idx': params.get('run_idx'),
        'ia3_type': params.get('ia3_type'),
        'num_virtual_tokens': params.get('num_virtual_tokens')
    }


def find_best_hyperparameters_by_performance(results, metric_key='with_icl_accuracy'):
    """Find best hyperparameters based on performance metric"""
    by_n_examples = defaultdict(list)
    
    for result in results:
        params = extract_hyperparameters(result)
        n_examples = params.get('num_train_examples')
        if n_examples is not None:
            metrics = result.get('metrics', {})
            score = metrics.get(metric_key, 0)
            by_n_examples[n_examples].append((score, params, result))
    
    best_params = {}
    best_results = {}

    for n_examples, candidates in by_n_examples.items():
        if candidates:
            best_score, best_param, best_result = max(candidates, key=lambda x: x[0])
            best_params[n_examples] = best_param
            best_results[n_examples] = best_result
    
    return best_params, best_results


def find_best_hyperparameters_by_dev_loss(results, model_id, model_type, trained_dataset):
    """Find best hyperparameters based on dev loss"""
    by_n_examples = defaultdict(list)
    
    for result in results:
        params = extract_hyperparameters(result)
        n_examples = params.get('num_train_examples')
        if n_examples is not None:
            # Load dev loss for this configuration
            dev_loss = load_dev_loss_statistics(
                model_type=model_type,
                trained_dataset=trained_dataset,
                model_id=model_id,
                lora_type=params.get('lora_type'),
                lora_r=params.get('lora_r'),
                lora_alpha=params.get('lora_alpha'),
                num_generated_tokens_train=params.get('num_generated_tokens_train'),
                num_train_examples=params.get('num_train_examples'),
                lr=params.get('lr'),
                run_idx=params.get('run_idx'),
                ce_loss_weight=params.get('ce_loss_weight'),
                label_type=params.get('label_type'),
                ia3_type=params.get('ia3_type'),
                num_virtual_tokens=params.get('num_virtual_tokens')
            )
            
            if dev_loss is not None:
                by_n_examples[n_examples].append((dev_loss, params, result))
    
    best_params = {}
    best_results = {}

    for n_examples, candidates in by_n_examples.items():
        if candidates:
            # Lower dev loss is better
            best_dev_loss, best_param, best_result = min(candidates, key=lambda x: x[0])
            best_params[n_examples] = best_param
            best_results[n_examples] = best_result
    
    return best_params, best_results


def aggregate_by_n_examples(results, metric_keys):
    """Aggregate results by number of training examples"""
    by_n_examples = defaultdict(list)
    
    for result in results:
        params = extract_hyperparameters(result)
        n_examples = params.get('num_train_examples')
        if n_examples is not None:
            metrics = result.get('metrics', {})
            by_n_examples[n_examples].append(metrics)
    
    aggregated = {}
    for n_examples, metrics_list in by_n_examples.items():
        aggregated[n_examples] = {}
        for metric_key in metric_keys:
            values = []
            for metrics in metrics_list:
                if metric_key in metrics and metrics[metric_key] is not None:
                    values.append(metrics[metric_key])
            
            if values:
                aggregated[n_examples][f'{metric_key}_mean'] = np.mean(values)
                aggregated[n_examples][f'{metric_key}_std'] = np.std(values)
            else:
                aggregated[n_examples][f'{metric_key}_mean'] = None
                aggregated[n_examples][f'{metric_key}_std'] = None
    
    return aggregated


def plot_method_comparison(all_results, output_path, title, ylabel, ylim,
                          metric_key_mean, metric_key_std=None, uncertainty_mode=False):
    """Create comparison plot across different training methods"""
    
    plt.figure(figsize=(12, 8))
    
    # Define method styles with updated naming
    method_styles = {
        "Base Model": {"color": "black", "marker": "o", "linestyle": "--", "label": "Base Model"},
        "lora Token Training": {"color": "blue", "marker": "s", "linestyle": "-", "label": "Token Training (tok)"},
        "ia3 Token Training": {"color": "blue", "marker": "s", "linestyle": "-", "label": "Token Training (tok)"},
        "prefix Token Training": {"color": "blue", "marker": "s", "linestyle": "-", "label": "Token Training (tok)"},
        "lora Activation Training": {"color": "green", "marker": "^", "linestyle": "-", "label": "Activation Training (act)"},
        "ia3 Activation Training": {"color": "green", "marker": "^", "linestyle": "-", "label": "Activation Training (act)"},
        "prefix Activation Training": {"color": "green", "marker": "^", "linestyle": "-", "label": "Activation Training (act)"},
        "lora Token + Activation Training": {"color": "red", "marker": "D", "linestyle": "-", "label": "Token + Activation Training (tna)"},
        "ia3 Token + Activation Training": {"color": "red", "marker": "D", "linestyle": "-", "label": "Token + Activation Training (tna)"},
        "prefix Token + Activation Training": {"color": "red", "marker": "D", "linestyle": "-", "label": "Token + Activation Training (tna)"},
        "lora Sequential (act→tok) Training": {"color": "purple", "marker": "v", "linestyle": "-", "label": "Sequential (act→tok)"},
        "ia3 Sequential (act→tok) Training": {"color": "purple", "marker": "v", "linestyle": "-", "label": "Sequential (act→tok)"},
        "prefix Sequential (act→tok) Training": {"color": "purple", "marker": "v", "linestyle": "-", "label": "Sequential (act→tok)"},
        # "Sequential T2A": {"color": "orange", "marker": "<", "linestyle": "-", "label": "Sequential (tok→act)"},
        # "IA3 Training": {"color": "brown", "marker": ">", "linestyle": "-", "label": "IA3 Training"},
        # "Prompt Tuning": {"color": "pink", "marker": "P", "linestyle": "-", "label": "Prompt Tuning"},
        # "Prefix Tuning": {"color": "cyan", "marker": "X", "linestyle": "-", "label": "Prefix Tuning"},
    }
    
    all_n_values = set()
    plotted_methods = []
    
    for method_name, method_data in all_results.items():
        if not method_data:
            continue
            
        n_values = []
        means = []
        stds = []
        
        for n_examples in sorted(method_data.keys()):
            mean_val = method_data[n_examples].get(metric_key_mean)
            if mean_val is not None and not np.isnan(mean_val):
                n_values.append(n_examples)
                means.append(mean_val)
                all_n_values.add(n_examples)
                
                if metric_key_std:
                    std_val = method_data[n_examples].get(metric_key_std, 0)
                    stds.append(std_val if std_val is not None and not np.isnan(std_val) else 0)
                else:
                    stds.append(0)

        if n_values:
            style = method_styles.get(method_name, {"color": "gray", "marker": "x", "linestyle": ":"})
            if metric_key_std and any(s > 0 for s in stds):
                plt.errorbar(n_values, means, yerr=stds, label=style["label"], 
                           marker=style["marker"], linestyle=style["linestyle"], 
                           color=style["color"], capsize=3, capthick=1)
            else:
                plt.plot(n_values, means, label=style["label"], 
                        marker=style["marker"], linestyle=style["linestyle"], 
                        color=style["color"], markersize=6)
            plotted_methods.append(method_name)
    
    # Add Base Model with ICL performance to the Without ICL plots for reference
    if 'without_icl' in metric_key_mean:
        base_model_icl_means = []
        base_model_icl_stds = []
        base_model_icl_n = []
        base_model_data = all_results.get("Base Model", {})
        
        # Determine which with_icl metric to use for reference
        with_icl_key = metric_key_mean.replace('without_icl', 'with_icl')
        with_icl_std_key = metric_key_std.replace('without_icl', 'with_icl') if metric_key_std else None
        
        for n_val, metrics in sorted(base_model_data.items()):
            mean_val = metrics.get(with_icl_key)
            std_val = metrics.get(with_icl_std_key) if with_icl_std_key else 0
            if mean_val is not None and not np.isnan(mean_val):
                base_model_icl_means.append(mean_val)
                base_model_icl_stds.append(std_val if std_val is not None and not np.isnan(std_val) else 0)
                base_model_icl_n.append(n_val)

        if base_model_icl_n:
            plt.errorbar(base_model_icl_n, base_model_icl_means, yerr=base_model_icl_stds, 
                        label="Base Model (With ICL)", 
                        marker="o", linestyle=":", color="black", capsize=3)
            all_n_values.update(base_model_icl_n)
    
    # Add Base Model without ICL performance to the With ICL plots for reference
    if 'with_icl' in metric_key_mean:
        base_model_no_icl_means = []
        base_model_no_icl_stds = []
        base_model_no_icl_n = []
        base_model_data = all_results.get("Base Model", {})
        
        # Determine which without_icl metric to use for reference
        without_icl_key = metric_key_mean.replace('with_icl', 'without_icl')
        without_icl_std_key = metric_key_std.replace('with_icl', 'without_icl') if metric_key_std else None
        
        for n_val, metrics in sorted(base_model_data.items()):
            mean_val = metrics.get(without_icl_key)
            std_val = metrics.get(without_icl_std_key) if without_icl_std_key else 0
            if mean_val is not None and not np.isnan(mean_val):
                base_model_no_icl_means.append(mean_val)
                base_model_no_icl_stds.append(std_val if std_val is not None and not np.isnan(std_val) else 0)
                base_model_no_icl_n.append(n_val)

        if base_model_no_icl_n:
            plt.errorbar(base_model_no_icl_n, base_model_no_icl_means, yerr=base_model_no_icl_stds, 
                        label="Base Model (Without ICL)", 
                        marker="o", linestyle="--", color="black", capsize=3)
            all_n_values.update(base_model_no_icl_n)
    
    if not plotted_methods:
        plt.close()
        print(f"No data to plot for {title}")
        return
    
    plt.xlabel("Number of Training Examples")
    plt.ylabel(ylabel)
    plt.ylim(0, ylim)
    plt.title(title)
    
    # Only plot N values for which we have results
    if all_n_values:
        sorted_n_values = sorted(all_n_values)
        if len(sorted_n_values) > 1:
            if max(sorted_n_values) / min(sorted_n_values) > 10:
                plt.xscale('log')
                plt.xticks(sorted_n_values, labels=[str(n) for n in sorted_n_values])
            else:
                plt.xticks(sorted_n_values)
        else:
            plt.xticks(sorted_n_values)
    
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_path}")


def process_base_model_results(base_dir, eval_dataset, icl_source, icl_demos, uncertainty_mode, label_type, model_id):
    """Process base model evaluation results"""
    import glob
    import re
    import json
    import numpy as np
    from collections import defaultdict
    
    # Find base model files
    if uncertainty_mode:
        search_dir = os.path.join(base_dir, "base_uncertainty", eval_dataset)
    else:
        search_dir = os.path.join(base_dir, "base", eval_dataset)

    if not os.path.exists(search_dir):
        return {}
    
    # First, load the without_icl results from the separate file
    model_name = model_id.split('/')[-1]
    # Match with any number after T in the filename
    without_icl_pattern = os.path.join(search_dir, f"{model_name}_base_without_icl_T*.json")
    without_icl_files = glob.glob(without_icl_pattern)
    without_icl_filename = None
    if without_icl_files:
        # If multiple, pick the first (or you could sort and pick the lowest/highest T)
        without_icl_filename = os.path.basename(without_icl_files[0])
    else:
        # Fallback to T1 if nothing found
        without_icl_filename = f"{model_name}_base_without_icl_T1.json"
    without_icl_path = os.path.join(search_dir, without_icl_filename)
    
    base_without_icl_metrics = {}
    if os.path.exists(without_icl_path):
        try:
            with open(without_icl_path, 'r') as f:
                data = json.load(f)
            base_without_icl_metrics = data.get('metrics', {})
            print(f"Loaded base model without_icl metrics from: {without_icl_path}")
        except Exception as e:
            print(f"Error loading base model without_icl metrics: {e}")
    
    # Pattern for base model files (with_icl results)
    pattern = os.path.join(search_dir, "*.json")
    all_files = glob.glob(pattern)
    all_files = [file for file in all_files if 'without_icl' not in file and model_name in file]

    # Parse base model filenames to extract N values
    def parse_base_filename(filename):
        base = os.path.basename(filename)
        parts = base.split('_')
        if parts[2] != 'on':
            return None, None
        if parts[3] != icl_source.split('_')[0]:
            return None, None
        # Skip the without_icl file
        if 'without_icl' in base:
            return None, None
        # N: _N(\d+)_
        n_match = re.search(r'_N(\d+)_', base)
        N = int(n_match.group(1)) if n_match else None
        # run_idx: _(\d+)\.json$
        run_match = re.search(r'_(\d+)\.json$', base)
        run_idx = int(run_match.group(1)) if run_match else None
        return N, run_idx
    
    # Aggregate results by N
    if uncertainty_mode:
        results_by_n = defaultdict(lambda: {
            'with_icl_accuracy_top1': [], 'without_icl_accuracy_top1': [], 
            'with_icl_accuracy_label_set': [], 'without_icl_accuracy_label_set': [], 
            'with_icl_uncertainty_top1_mean': [], 'without_icl_uncertainty_top1_mean': [],
            'with_icl_uncertainty_label_mean': [], 'without_icl_uncertainty_label_mean': [],
            'with_icl_entropy_top1_mean': [], 'without_icl_entropy_top1_mean': [],
            'with_icl_entropy_label_mean': [], 'without_icl_entropy_label_mean': [],
            'with_icl_ece_top1': [], 'without_icl_ece_top1': [],
            'with_icl_ece_label': [], 'without_icl_ece_label': []
        })
    else:
        results_by_n = defaultdict(lambda: {'with_icl_accuracy': [], 'without_icl_accuracy': []})
    all_n_values = set()
    
    for filepath in all_files:
        N, run_idx = parse_base_filename(filepath)
        if N is None or run_idx is None:
            continue
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            metrics = data.get('metrics', {})

            if uncertainty_mode:
                # For uncertainty mode, get all relevant metrics
                for metric in ['accuracy_top1', 'accuracy_label_set', 'uncertainty_top1_mean', 
                              'uncertainty_label_mean', 'entropy_top1_mean', 'entropy_label_mean',
                              'ece_top1', 'ece_label']:
                    for prefix in ['with_icl', 'without_icl']:
                        key = f'{prefix}_{metric}'
                        if key in metrics:
                            results_by_n[N][key].append(metrics[key])
            else:
                with_icl_acc = metrics.get('with_icl_accuracy')
                without_icl_acc = metrics.get('without_icl_accuracy')
                if with_icl_acc is not None:
                    results_by_n[N]['with_icl_accuracy'].append(with_icl_acc)
                if without_icl_acc is not None:
                    results_by_n[N]['without_icl_accuracy'].append(without_icl_acc)
            all_n_values.add(N)
        except Exception as e:
            continue


    # Create final results structure
    final_results = {}
    for N in sorted(all_n_values):
        final_results[N] = {}
        
        if uncertainty_mode:
            # Process all uncertainty metrics
            for metric in ['accuracy_top1', 'accuracy_label_set', 'uncertainty_top1_mean', 
                          'uncertainty_label_mean', 'entropy_top1_mean', 'entropy_label_mean',
                          'ece_top1', 'ece_label']:
                for prefix in ['with_icl', 'without_icl']:
                    # Add without_icl metrics (same for all N)
                    if prefix == 'without_icl':
                        without_icl_key = f'{prefix}_{metric}'
                        if without_icl_key in base_without_icl_metrics:
                            final_results[N][f'{without_icl_key}_mean'] = base_without_icl_metrics[without_icl_key]
                            final_results[N][f'{without_icl_key}_std'] = 0.0  # No std for single result
                            # Store all individual scores (single value for without_icl)
                            final_results[N][f'{without_icl_key}_all'] = [
                                float(base_without_icl_metrics[without_icl_key])
                            ]
                    
                    else:
                        # Add with_icl metrics
                        with_icl_key = f'{prefix}_{metric}'
                        scores = results_by_n[N].get(with_icl_key, [])
                        if scores:
                            final_results[N][f'{with_icl_key}_mean'] = float(np.mean(scores))
                            final_results[N][f'{with_icl_key}_std'] = float(np.std(scores))
                            # Store all individual scores
                            final_results[N][f'{with_icl_key}_all'] = [float(v) for v in scores]
                        else:
                            final_results[N][f'{with_icl_key}_mean'] = None
                            final_results[N][f'{with_icl_key}_std'] = None
                            final_results[N][f'{with_icl_key}_all'] = []
        else:
            # Process standard accuracy metrics
            # Add without_icl metrics (same for all N)
            if 'without_icl_accuracy' in base_without_icl_metrics:
                final_results[N]['without_icl_accuracy_mean'] = base_without_icl_metrics['without_icl_accuracy']
                final_results[N]['without_icl_accuracy_std'] = 0.0  # No std for single result
                # Store all individual scores (single value for without_icl)
                final_results[N]['without_icl_accuracy_all'] = [
                    float(base_without_icl_metrics['without_icl_accuracy'])
                ]
            
            # Add with_icl metrics
            scores = results_by_n[N].get('with_icl_accuracy', [])
            if scores:
                final_results[N]['with_icl_accuracy_mean'] = float(np.mean(scores))
                final_results[N]['with_icl_accuracy_std'] = float(np.std(scores))
                # Store all individual scores
                final_results[N]['with_icl_accuracy_all'] = [float(v) for v in scores]
            else:
                final_results[N]['with_icl_accuracy_mean'] = None
                final_results[N]['with_icl_accuracy_std'] = None
                final_results[N]['with_icl_accuracy_all'] = []

    return final_results


def get_all_plot_metrics(uncertainty_mode):
    """Get all plot metrics for both uncertainty and standard modes"""
    if uncertainty_mode:
        return [
            # Accuracy metrics
            ("with_icl_accuracy_top1_mean", "with_icl_accuracy_top1_std", "Top-1 Accuracy (With ICL)", "Top-1 Accuracy (%)"),
            ("without_icl_accuracy_top1_mean", "without_icl_accuracy_top1_std", "Top-1 Accuracy (Without ICL)", "Top-1 Accuracy (%)"),
            ("with_icl_accuracy_label_set_mean", "with_icl_accuracy_label_set_std", "Label Set Accuracy (With ICL)", "Label Set Accuracy (%)"),
            ("without_icl_accuracy_label_set_mean", "without_icl_accuracy_label_set_std", "Label Set Accuracy (Without ICL)", "Label Set Accuracy (%)"),
            
            # Uncertainty metrics
            ("with_icl_uncertainty_top1_mean_mean", "with_icl_uncertainty_top1_mean_std", "Top-1 Uncertainty (With ICL)", "Top-1 Uncertainty"),
            ("without_icl_uncertainty_top1_mean_mean", "without_icl_uncertainty_top1_mean_std", "Top-1 Uncertainty (Without ICL)", "Top-1 Uncertainty"),
            ("with_icl_uncertainty_label_mean_mean", "with_icl_uncertainty_label_mean_std", "Label Uncertainty (With ICL)", "Label Uncertainty"),
            ("without_icl_uncertainty_label_mean_mean", "without_icl_uncertainty_label_mean_std", "Label Uncertainty (Without ICL)", "Label Uncertainty"),
            
            # Entropy metrics
            ("with_icl_entropy_top1_mean_mean", "with_icl_entropy_top1_mean_std", "Top-1 Entropy (With ICL)", "Top-1 Entropy"),
            ("without_icl_entropy_top1_mean_mean", "without_icl_entropy_top1_mean_std", "Top-1 Entropy (Without ICL)", "Top-1 Entropy"),
            ("with_icl_entropy_label_mean_mean", "with_icl_entropy_label_mean_std", "Label Entropy (With ICL)", "Label Entropy"),
            ("without_icl_entropy_label_mean_mean", "without_icl_entropy_label_mean_std", "Label Entropy (Without ICL)", "Label Entropy"),
            
            # ECE metrics
            ("with_icl_ece_top1_mean", "with_icl_ece_top1_std", "Top-1 ECE (With ICL)", "Top-1 ECE"),
            ("without_icl_ece_top1_mean", "without_icl_ece_top1_std", "Top-1 ECE (Without ICL)", "Top-1 ECE"),
            ("with_icl_ece_label_mean", "with_icl_ece_label_std", "Label ECE (With ICL)", "Label ECE"),
            ("without_icl_ece_label_mean", "without_icl_ece_label_std", "Label ECE (Without ICL)", "Label ECE"),
        ]
    else:
        return [
            ("with_icl_accuracy_mean", "with_icl_accuracy_std", "Accuracy (With ICL)", "Accuracy (%)"),
            ("without_icl_accuracy_mean", "without_icl_accuracy_std", "Accuracy (Without ICL)", "Accuracy (%)"),
        ]


def process_trained_model_results_full(model_id, base_dir, model_type, trained_dataset, eval_dataset, 
                                     icl_source, icl_demos, uncertainty_mode, label_type, hp_selection='performance'):
    """Process trained model results with full metric support and hyperparameter selection options"""
    import glob
    import numpy as np
    import os
    from collections import defaultdict
    import json
    import re

    if uncertainty_mode:
        search_dir = os.path.join(base_dir, model_type + "_uncertainty", trained_dataset)
    else:
        search_dir = os.path.join(base_dir, model_type, trained_dataset)

    if not os.path.exists(search_dir):
        return {}, {}, set()

    pattern = os.path.join(search_dir, f"*.json")
    all_files = glob.glob(pattern)
    filtered_files = []
    for file in all_files:
        # Skip files that don't match the eval_dataset
        if re.search(f'{model_id.split("/")[-1]}_{eval_dataset}_on_{icl_source}', os.path.basename(file)):
            if '_ldr' in file: # We don't process LDR results here
                continue
            filtered_files.append(file)
    all_files = filtered_files

    if not all_files:
        return {}, {}, set()

    # Parse model_type to extract base method and training variant
    if '-' in model_type:
        base_method, training_variant = model_type.split('-', 1)
    else:
        base_method = 'lora'
        training_variant = model_type

    # Helper: parse N, lr, run_idx, cew (if present) from filename
    def parse_filename(filename, model_type):
        base = os.path.basename(filename)

        # Parse model_type to extract base method and training variant
        if '-' in model_type:
            base_method, training_variant = model_type.split('-', 1)
        else:
            base_method = 'lora'
            training_variant = model_type
            
        if training_variant in ['tok', 'a2t']:
            parts = base.split('_')
            try:
                for i, part in enumerate(parts):
                    if part in ['ground', 'icl']:
                        if i+1 < len(parts) and parts[i+1] == 'truth':
                            label_type_in_file = 'ground_truth'
                        elif i+1 < len(parts) and parts[i+1] == 'outputs':
                            label_type_in_file = 'icl_outputs'
                        else:
                            label_type_in_file = None
                        if i + 4 < len(parts):
                            N = int(parts[i + 3])
                            lr = float(parts[i + 4])
                            break
                else:
                    N = None
                    lr = None
            except (ValueError, IndexError):
                N = None
                lr = None
            
            run_match = re.search(r'_(\d+)\.json$', base)
            run_idx = int(run_match.group(1)) if run_match else None
            cew = None

        elif training_variant in ['act', 't2a']:
            parts = base.split('_')
            try:
                if base_method == 'lora':
                    for i, part in enumerate(parts):
                        if part.startswith('a') and part[1:].isdigit():
                            if i + 3 < len(parts):
                                N = int(parts[i + 2])
                                lr = float(parts[i + 3])
                                break
                else:
                    if '_ldr' in base:
                        N = int(parts[-6])
                        lr = float(parts[-5])
                    else:
                        N = int(parts[-3])
                        lr = float(parts[-2])

            except (ValueError, IndexError):
                N = None
                lr = None
            
            run_match = re.search(r'_(\d+)\.json$', base)
            run_idx = int(run_match.group(1)) if run_match else None
            label_type_in_file = None
            cew = None
            
        elif training_variant == 'tna':
            n_match = re.search(r'_(\d+)ex_', base)
            N = int(n_match.group(1)) if n_match else None
            
            lr_match = re.search(r'_([\d.eE+-]+)lr_cew', base)
            lr = float(lr_match.group(1)) if lr_match else None
            
            cew_match = re.search(r'cew([\d.eE+-]+)_', base)
            cew = float(cew_match.group(1)) if cew_match else None

            run_match = re.search(r'_(\d+)\.json$', base)
            run_idx = int(run_match.group(1)) if run_match else None
            label_type_in_file = None
            
        else:
            N = None
            lr = None
            run_idx = None
            cew = None
            label_type_in_file = None
            
        return N, lr, run_idx, cew, label_type_in_file

    # Aggregate results by N, HPs
    if uncertainty_mode:
        raw_results = defaultdict(lambda: defaultdict(lambda: {
            'with_icl_accuracy_top1': [], 'without_icl_accuracy_top1': [], 
            'with_icl_accuracy_label_set': [], 'without_icl_accuracy_label_set': [], 
            'with_icl_uncertainty_top1_mean': [], 'without_icl_uncertainty_top1_mean': [],
            'with_icl_uncertainty_label_mean': [], 'without_icl_uncertainty_label_mean': [],
            'with_icl_entropy_top1_mean': [], 'without_icl_entropy_top1_mean': [],
            'with_icl_entropy_label_mean': [], 'without_icl_entropy_label_mean': [],
            'with_icl_ece_top1': [], 'without_icl_ece_top1': [],
            'with_icl_ece_label': [], 'without_icl_ece_label': []
        }))
    else:
        raw_results = defaultdict(lambda: defaultdict(lambda: {'with_icl_accuracy': [], 'without_icl_accuracy': []}))
    all_n_values = set()
    available_metrics = set()

    for filepath in all_files:
        N, lr, run_idx, cew, label_type_in_file = parse_filename(filepath, model_type)
        if N is None or lr is None or run_idx is None:
            continue
        # Filter by label_type if present in filename or in JSON
        if label_type_in_file and label_type_in_file != label_type:
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            trained_params = data.get('trained_model_params', {})
            if 'label_type' in trained_params and training_variant not in ['act', 't2a', 'tna'] and trained_params['label_type'] != label_type:
                continue

            metrics = data.get('metrics', {})
            available_metrics.update(metrics.keys())
            
            # HP key: (lr, cew) if cew present, else lr
            hp_key = (lr, cew) if cew is not None else lr
            all_n_values.add(N)
            
            if uncertainty_mode:
                # Collect all uncertainty metrics
                for metric in ['accuracy_top1', 'accuracy_label_set', 'uncertainty_top1_mean', 
                              'uncertainty_label_mean', 'entropy_top1_mean', 'entropy_label_mean',
                              'ece_top1', 'ece_label']:
                    for prefix in ['with_icl', 'without_icl']:
                        key = f'{prefix}_{metric}'
                        if key in metrics and metrics[key] is not None:
                            raw_results[N][hp_key][key].append(metrics[key])
            else:
                # Standard accuracy metrics
                for prefix in ['with_icl', 'without_icl']:
                    key = f'{prefix}_accuracy'
                    if key in metrics and metrics[key] is not None:
                        raw_results[N][hp_key][key].append(metrics[key])

        except Exception as e:
            continue

    # For each N, select best HP based on selection method
    method_data = {}
    best_hps = {}

    for N in sorted(all_n_values):
        method_data[N] = {}
        best_hps[N] = {}
        
        # Get all metrics for this N
        all_metrics = list(raw_results[N][next(iter(raw_results[N]))].keys())
        
        for metric in all_metrics:
            if hp_selection == 'performance':
                # Select best HP based on performance (higher is better)
                best_mean = -float('inf')
                best_hp = None
                for hp_key, metrics_dict in raw_results[N].items():
                    # vals = metrics_dict.get(metric, []) # ######################### IMPORTANT: previously selected best hp for each metric. Now only choosing best hp based on without_icl_accuracy_top1
                    vals = metrics_dict.get('without_icl_accuracy_top1', [])
                    if not vals:
                        vals = metrics_dict.get('without_icl_accuracy', [])
                    if vals:
                        mean_val = np.mean(vals)
                        if mean_val > best_mean:
                            best_mean = mean_val
                            best_hp = hp_key
            else:  # hp_selection == 'dev_loss'
                # Select best HP based on dev loss (lower is better)
                best_dev_loss = float('inf')
                best_hp = None
                for hp_key, metrics_dict in raw_results[N].items():
                    # Get dev loss for this HP combination
                    # We need to extract the parameters from hp_key
                    if isinstance(hp_key, tuple):
                        lr, cew = hp_key
                    else:
                        lr, cew = hp_key, None

                    # Find a result with these parameters to get other needed params
                    for filepath in all_files:
                        file_N, file_lr, file_run_idx, file_cew, _ = parse_filename(filepath, model_type)
                        if file_N == N and file_lr == lr and file_cew == cew:
                            # Load the result to get other parameters
                            try:
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                params = data.get('trained_model_params', {})
                                
                                # Load dev loss
                                dev_loss = load_dev_loss_statistics(
                                    model_type=model_type,
                                    trained_dataset=trained_dataset,
                                    model_id=model_id,
                                    lora_type=params.get('lora_type'),
                                    lora_r=params.get('lora_r'),
                                    lora_alpha=params.get('lora_alpha'),
                                    num_generated_tokens_train=params.get('num_generated_tokens_train'),
                                    num_train_examples=params.get('num_train_examples'),
                                    lr=params.get('lr'),
                                    run_idx=params.get('run_idx'),
                                    ce_loss_weight=params.get('ce_loss_weight'),
                                    label_type=params.get('label_type'),
                                    ia3_type=params.get('ia3_type'),
                                    num_virtual_tokens=params.get('num_virtual_tokens')
                                )
                                
                                if dev_loss is not None and dev_loss < best_dev_loss:
                                    best_dev_loss = dev_loss
                                    best_hp = hp_key
                                break
                            except:
                                continue
            
            # Save best HP and stats
            if best_hp is not None:
                vals = raw_results[N][best_hp][metric]
                method_data[N][f"{metric}_mean"] = float(np.mean(vals)) if vals else 0.0
                method_data[N][f"{metric}_std"] = float(np.std(vals)) if vals else 0.0
                # Store all individual scores for the selected HP
                method_data[N][f"{metric}_all"] = [float(v) for v in vals] if vals else []
                best_hps[N][metric] = best_hp

    return method_data, best_hps, available_metrics


def main():
    parser = argparse.ArgumentParser(description="Unified plotting for distillation project with comprehensive metrics")
    # Data source
    parser.add_argument("--base_dir", type=str, default="../outputs/evaluations",
                        help="Base directory for evaluation results")
    parser.add_argument("--trained_dataset", type=str, required=True,
                        help="Dataset models were trained on")
    parser.add_argument("--eval_dataset", type=str, required=True,
                        help="Dataset to evaluate on")
    parser.add_argument("--icl_source_dataset", type=str, required=True,
                        help="ICL source dataset")
    parser.add_argument("--icl_max_demos", type=int, required=True,
                        help="Number of ICL demonstrations")
    parser.add_argument("--label_types", nargs='+', type=str, default=["ground_truth", "icl_outputs"],
                        help="Label types to plot (default: both)")
    # Model selection
    parser.add_argument("--base_method", type=str, choices=['lora', 'ia3', 'prompt', 'prefix'], default='lora',
                        help="Base method to use (lora, ia3, prompt, prefix)")
    parser.add_argument("--training_variants", nargs='+', 
                        choices=['tok', 'act', 'tna', 'a2t', 't2a'],
                        default=['tok', 'act', 'tna', 'a2t', 't2a'],
                        help="Training variants to include in plots")
    parser.add_argument("--include_base_model", action='store_true', default=True,
                        help="Include base model in comparison")
    # Analysis options
    parser.add_argument("--uncertainty_mode", action='store_true',
                        help="Use uncertainty evaluation results")
    parser.add_argument("--hp_selection", type=str, choices=['performance', 'dev_loss'], default='performance',
                        help="Hyperparameter selection method: 'performance' (best mean performance) or 'dev_loss' (best mean dev loss)")
    # Output
    parser.add_argument("--output_dir", type=str, default="../plots/unified",
                        help="Output directory for plots")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model ID for plot naming")
    args = parser.parse_args()

    model_name = args.model_id.split('/')[-1]
    eval_suffix = "_uncertainty" if args.uncertainty_mode else ""

    # For each label type, plot everything available
    for label_type in args.label_types:
        print(f"\n==== Plotting for label_type: {label_type} ====")
        print(f"Hyperparameter selection method: {args.hp_selection}")
        
        output_base = os.path.join(args.output_dir, label_type, args.trained_dataset, 
                                  f"{args.eval_dataset}_{args.icl_source_dataset}_{args.base_method}_hp{args.hp_selection}_demos{args.icl_max_demos}{eval_suffix}")
        os.makedirs(output_base, exist_ok=True)
        all_results = {}
        all_best_hyperparams = {}

        # --- Collect results for all methods ---
        # 1. Base model (if requested)
        if args.include_base_model:
            print("\nProcessing base model results...")
            base_data = process_base_model_results(
                args.base_dir, args.eval_dataset, args.icl_source_dataset, 
                args.icl_max_demos, args.uncertainty_mode, label_type=label_type,
                model_id=args.model_id
            )
            if base_data:
                all_results["Base Model"] = base_data
                print(f"  Found base model results")
            else:
                print(f"  No base model results found")

        # 2. Trained models (loop over training variants for the selected base method)
        method_mapping = {
            'tok': "Token Training",
            'act': "Activation Training", 
            'tna': "Token + Activation Training",
            'a2t': "Sequential (act→tok) Training",
            't2a': "Sequential (tok→act) Training"
        }
        
        for training_variant in args.training_variants:
            # Construct model type based on base method and training variant
            if args.base_method == 'lora':
                model_type = training_variant
            else:
                model_type = f"{args.base_method}-{training_variant}"
            
            method_name = f"{args.base_method} {method_mapping.get(training_variant, training_variant)}"
            print(f"\nProcessing {method_name} results...")
            method_data, best_hps, available_metrics = process_trained_model_results_full(
                args.model_id, args.base_dir, model_type, args.trained_dataset, args.eval_dataset,
                args.icl_source_dataset, args.icl_max_demos, args.uncertainty_mode, 
                label_type=label_type, hp_selection=args.hp_selection
            )
            if method_data:
                all_results[method_name] = method_data
                all_best_hyperparams[method_name] = best_hps
                print(f"  Found {len(method_data)} data points, metrics: {list(available_metrics)}")
            else:
                print(f"  No results found for {method_name}")

        if not all_results:
            print("No results found to plot!")
            continue

        # --- Plot all available metrics ---
        for metric_key_mean, metric_key_std, plot_title, ylabel in get_all_plot_metrics(args.uncertainty_mode):
            plot_path = os.path.join(output_base, f"{model_name}_{metric_key_mean}.png")
            if 'accuracy' in metric_key_mean:
                ylim = 105.
            elif 'entropy' in metric_key_mean:
                ylim = 2.
            else:
                ylim = 1.
            plot_method_comparison(
                all_results, plot_path, 
                f"{plot_title}\nEval: {args.eval_dataset}, ICL Source: {args.icl_source_dataset} - max {args.icl_max_demos} demos\nBase Method: {args.base_method}, Label type: {label_type}\nHP Selection: {args.hp_selection}",
                ylabel, ylim, metric_key_mean, metric_key_std, args.uncertainty_mode
            )

        # --- Save summary data and best hyperparams ---
        summary_data = {
            'config': {
                'trained_dataset': args.trained_dataset,
                'eval_dataset': args.eval_dataset,
                'icl_source_dataset': args.icl_source_dataset,
                'icl_max_demos': args.icl_max_demos,
                'base_method': args.base_method,
                'training_variants': args.training_variants,
                'include_base_model': args.include_base_model,
                'uncertainty_mode': args.uncertainty_mode,
                'label_type': label_type,
                'hp_selection': args.hp_selection,
            },
            'results': all_results,
            'best_hyperparams': all_best_hyperparams
        }
        summary_path = os.path.join(output_base, f"{model_name}_plotting_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"\nPlotting complete for label_type={label_type}!")
        print(f"Plots saved to: {output_base}")
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()