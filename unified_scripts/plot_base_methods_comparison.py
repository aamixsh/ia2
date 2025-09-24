#!/usr/bin/env python3
"""
Base Methods Comparison Plotting Script for Distillation Project

This script creates comparison plots across multiple base methods (lora, ia3, prompt, prefix)
and their training variants (tok, act, tna, a2t, t2a).

Supports comprehensive evaluation metrics and hyperparameter selection methods.
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

# Import functions from the unified plotting script
from plot_unified import (
    load_evaluation_results, load_dev_loss_statistics, find_evaluation_results,
    extract_hyperparameters, aggregate_by_n_examples, get_all_plot_metrics,
    plot_method_comparison, process_base_model_results, process_trained_model_results_full
)


def plot_base_methods_comparison(all_results, output_path, title, ylabel, ylim,
                                metric_key_mean, metric_key_std=None, uncertainty_mode=False):
    """Create comparison plot across different base methods and their training variants"""
    
    plt.figure(figsize=(14, 10))
    
    # Define method styles with base method colors
    base_method_colors = {
        "lora": "blue",
        "ia3": "green", 
        "prompt": "red",
        "prefix": "purple"
    }
    
    method_styles = {
        # LoRA variants
        "LoRA Token Training": {"color": base_method_colors["lora"], "marker": "s", "linestyle": "-", "label": "LoRA Token Training"},
        "LoRA Activation Training": {"color": base_method_colors["lora"], "marker": "^", "linestyle": "-", "label": "LoRA Activation Training"},
        "LoRA Token + Activation": {"color": base_method_colors["lora"], "marker": "D", "linestyle": "-", "label": "LoRA Token + Activation"},
        "LoRA Sequential A2T": {"color": base_method_colors["lora"], "marker": "v", "linestyle": "-", "label": "LoRA Sequential A2T"},
        "LoRA Sequential T2A": {"color": base_method_colors["lora"], "marker": "<", "linestyle": "-", "label": "LoRA Sequential T2A"},
        
        # IA3 variants
        "IA3 Token Training": {"color": base_method_colors["ia3"], "marker": "s", "linestyle": "--", "label": "IA3 Token Training"},
        "IA3 Activation Training": {"color": base_method_colors["ia3"], "marker": "^", "linestyle": "--", "label": "IA3 Activation Training"},
        "IA3 Token + Activation": {"color": base_method_colors["ia3"], "marker": "D", "linestyle": "--", "label": "IA3 Token + Activation"},
        "IA3 Sequential A2T": {"color": base_method_colors["ia3"], "marker": "v", "linestyle": "--", "label": "IA3 Sequential A2T"},
        "IA3 Sequential T2A": {"color": base_method_colors["ia3"], "marker": "<", "linestyle": "--", "label": "IA3 Sequential T2A"},
        
        # Prompt variants
        "PROMPT Token Training": {"color": base_method_colors["prompt"], "marker": "s", "linestyle": ":", "label": "Prompt Token Training"},
        "PROMPT Activation Training": {"color": base_method_colors["prompt"], "marker": "^", "linestyle": ":", "label": "Prompt Activation Training"},
        "PROMPT Token + Activation": {"color": base_method_colors["prompt"], "marker": "D", "linestyle": ":", "label": "Prompt Token + Activation"},
        "PROMPT Sequential A2T": {"color": base_method_colors["prompt"], "marker": "v", "linestyle": ":", "label": "Prompt Sequential A2T"},
        "PROMPT Sequential T2A": {"color": base_method_colors["prompt"], "marker": "<", "linestyle": ":", "label": "Prompt Sequential T2A"},
        
        # Prefix variants
        "PREFIX Token Training": {"color": base_method_colors["prefix"], "marker": "s", "linestyle": "-.", "label": "Prefix Token Training"},
        "PREFIX Activation Training": {"color": base_method_colors["prefix"], "marker": "^", "linestyle": "-.", "label": "Prefix Activation Training"},
        "PREFIX Token + Activation": {"color": base_method_colors["prefix"], "marker": "D", "linestyle": "-.", "label": "Prefix Token + Activation"},
        "PREFIX Sequential A2T": {"color": base_method_colors["prefix"], "marker": "v", "linestyle": "-.", "label": "Prefix Sequential A2T"},
        "PREFIX Sequential T2A": {"color": base_method_colors["prefix"], "marker": "<", "linestyle": "-.", "label": "Prefix Sequential T2A"},
        
        # Base model
        "Base Model": {"color": "black", "marker": "o", "linestyle": "--", "label": "Base Model"},
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
    
    # Add Base Model with ICL performance to the Without ICL plots for reference (only for accuracy metrics)
    if 'without_icl' in metric_key_mean and 'accuracy' in metric_key_mean:
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
    
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Base methods comparison plotting for distillation project")
    
    # Data source
    parser.add_argument("--base_dir", type=str, default="../outputs/evaluations",
                        help="Base directory for evaluation results")
    parser.add_argument("--trained_dataset", default="sst", type=str,
                        help="Dataset models were trained on")
    parser.add_argument("--eval_dataset", default="sst", type=str,
                        help="Dataset to evaluate on")
    parser.add_argument("--icl_source_dataset", default="sst", type=str,
                        help="ICL source dataset")
    parser.add_argument("--icl_max_demos", default=256, type=int,
                        help="Number of ICL demonstrations")
    parser.add_argument("--label_types", nargs='+', type=str, default=["ground_truth", "icl_outputs"],
                        help="Label types to plot (default: both)")
    
    # Base method selection
    parser.add_argument("--base_methods", nargs='+', 
                        choices=['lora', 'ia3', 'prompt', 'prefix'],
                        default=['lora', 'ia3', 'prefix'],
                        help="Base methods to compare")
    parser.add_argument("--training_variants", nargs='+', 
                        choices=['tok', 'act', 'tna', 'a2t'],
                        default=['tok', 'act', 'tna', 'a2t'],
                        help="Training variants to include in plots")
    parser.add_argument("--include_base_model", action='store_true',
                        help="Include base model in comparison")
    
    # Analysis options
    parser.add_argument("--uncertainty_mode", action='store_true',
                        help="Use uncertainty evaluation results")
    parser.add_argument("--hp_selection", type=str, choices=['performance', 'dev_loss'], default='performance',
                        help="Hyperparameter selection method: 'performance' (best mean performance) or 'dev_loss' (best mean dev loss)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="../plots/base_methods_comparison",
                        help="Output directory for plots")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model ID for plot naming")
    
    args = parser.parse_args()

    model_name = args.model_id.split('/')[-1]
    eval_suffix = "_uncertainty" if args.uncertainty_mode else ""

    # For each label type, plot everything available
    for label_type in args.label_types:
        print(f"\n==== Plotting for label_type: {label_type} ====")
        print(f"Base methods: {args.base_methods}")
        print(f"Training variants: {args.training_variants}")
        print(f"Hyperparameter selection method: {args.hp_selection}")
        
        output_base = os.path.join(args.output_dir, label_type, args.trained_dataset, 
                                  f"{args.eval_dataset}_{args.icl_source_dataset}_hp{args.hp_selection}_demos{args.icl_max_demos}{eval_suffix}")
        os.makedirs(output_base, exist_ok=True)
        all_results = {}
        all_best_hyperparams = {}

        # --- Collect results for all base methods ---
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

        # 2. Trained models for each base method
        method_mapping = {
            'tok': "Token Training",
            'act': "Activation Training", 
            'tna': "Token + Activation",
            'a2t': "Sequential A2T",
            't2a': "Sequential T2A"
        }
        
        for base_method in args.base_methods:
            print(f"\n==== Processing {base_method.upper()} base method ====")
            
            for training_variant in args.training_variants:
                # Construct model type based on base method and training variant
                if base_method == 'lora':
                    model_type = training_variant
                else:
                    model_type = f"{base_method}-{training_variant}"
                
                method_name = f"{base_method.upper()} {method_mapping.get(training_variant, training_variant)}"
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
            plot_base_methods_comparison(
                all_results, plot_path, 
                f"{plot_title}\nEval: {args.eval_dataset}, ICL Source: {args.icl_source_dataset} - max {args.icl_max_demos} demos\nBase Methods: {', '.join(args.base_methods).upper()}, Label type: {label_type}\nHP Selection: {args.hp_selection}",
                ylabel, ylim, metric_key_mean, metric_key_std, args.uncertainty_mode
            )

        # --- Save summary data and best hyperparams ---
        summary_data = {
            'config': {
                'trained_dataset': args.trained_dataset,
                'eval_dataset': args.eval_dataset,
                'icl_source_dataset': args.icl_source_dataset,
                'icl_max_demos': args.icl_max_demos,
                'base_methods': args.base_methods,
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
