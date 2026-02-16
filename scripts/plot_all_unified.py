#!/usr/bin/env python3
"""
Unified Batch Plotting Script for Distillation Project

This script generates plots across multiple configurations using the unified plotting script.
Supports all training methods and evaluation modes.
"""

import os
import sys
import argparse
import subprocess
import time
from multiprocessing import Pool
from itertools import product


def run_single_plot(args_tuple):
    """Run unified plotting for a single configuration"""
    (trained_dataset, eval_dataset, icl_source_dataset, icl_max_demos, 
     base_method, training_variants, include_base_model, uncertainty_mode, hp_selection, base_dir, output_dir, model_id, lora_r_filter, lora_type_filter) = args_tuple

    script = 'plot_unified.py'
    cmd = [
        'python', script,
        '--base_dir', str(base_dir),
        '--trained_dataset', str(trained_dataset),
        '--eval_dataset', str(eval_dataset),
        '--icl_source_dataset', str(icl_source_dataset),
        '--icl_max_demos', str(icl_max_demos),
        '--output_dir', str(output_dir),
        '--model_id', str(model_id)
    ]

    # Add base method and training variants
    cmd.extend(['--base_method', str(base_method)])
    cmd.extend(['--training_variants'] + [str(tv) for tv in training_variants])
    if include_base_model:
        cmd.append('--include_base_model')
    
    # Add hp selection
    cmd.extend(['--hp_selection', str(hp_selection)])

    # Add uncertainty mode if requested
    if uncertainty_mode:
        cmd.append('--uncertainty_mode')
    
    # Add lora filters if specified
    if lora_r_filter is not None:
        cmd.extend(['--lora_r', str(lora_r_filter)])
    if lora_type_filter is not None:
        cmd.extend(['--lora_type', str(lora_type_filter)])

    config_name = f"{trained_dataset}_{eval_dataset}_{icl_source_dataset}_{base_method}_demos{icl_max_demos}"
    if lora_r_filter is not None:
        config_name += f"_r{lora_r_filter}"
    if lora_type_filter is not None:
        config_name += f"_{lora_type_filter}"
    
    # Run the plotting
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)  # 30 min timeout
        
        if result.returncode == 0:
            if uncertainty_mode:
                config_name += "_uncertainty"
            print(f"✓ Plotting completed successfully for {config_name}")
            return True
        else:
            print(f"✗ Plotting failed for {config_name}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Plotting timed out for {config_name}")
        return False
    except Exception as e:
        print(f"✗ Error running plotting for {config_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch plotting for distillation project")
    
    # Dataset configurations
    parser.add_argument("--trained_datasets", nargs='+', default=['sciqa'],
                        help="Datasets models were trained on")
    parser.add_argument("--eval_datasets", nargs='+', default=['sciqa'],
                        help="Datasets to evaluate on")
    parser.add_argument("--icl_source_datasets", nargs='+', default=['sciqa'],
                        help="ICL source datasets")
    parser.add_argument("--icl_max_demos", nargs='+', type=int, default=[256],
                        help="Number of ICL demonstrations")
    
    # Model selection
    parser.add_argument("--base_method", type=str, choices=['lora', 'ia3', 'prompt', 'prefix'], default='lora',
                        help="Base method to use (lora, ia3, prompt, prefix)")
    parser.add_argument("--training_variants", nargs='+', 
                        choices=['tok', 'act', 'tna', 'a2t', 't2a', 'tokl'],
                        default=['tok', 'act', 'tna', 'a2t', 't2a', 'tokl'],
                        help="Training variants to include in plots")
    parser.add_argument("--include_base_model", action='store_true', default=True,
                        help="Include base model in comparison")
    
    # Analysis options
    parser.add_argument("--uncertainty_mode", action='store_true',
                        help="Include uncertainty analysis plots")
    parser.add_argument("--hp_selection", default='performance', choices=['performance', 'dev_loss'],
                        help="Hyperparameters to optimize")
    # parser.add_argument("--plot_types", nargs='+', 
    #                     choices=['accuracy', 'uncertainty', 'comparison', 'all'],
    #                     default=['all'],
    #                     help="Types of plots to generate")
    
    # System settings
    parser.add_argument("--max_parallel", type=int, default=3,
                        help="Maximum number of parallel plotting processes")
    parser.add_argument("--base_dir", type=str, default="../outputs/evaluations",
                        help="Base directory for evaluation results")
    parser.add_argument("--output_dir", type=str, default="../plots/unified",
                        help="Output directory for plots")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", choices=['meta-llama/Llama-3.2-1B', 'Qwen/Qwen3-4B-Base', 'Qwen/Qwen2.5-1.5B', 'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.1-8B'],
                        help="Model ID for plot naming")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Filter by LoRA rank (only include results with this rank)")
    parser.add_argument("--lora_type", type=str, default='qko',
                        help="Filter by LoRA type (only include results with this type, e.g., 'qkv', 'qko')")
    
    args = parser.parse_args()
    
    # Generate all plotting configurations
    plot_experiments = []
    
    # Create combinations of all configurations
    dataset_combinations = list(product(
        args.trained_datasets,
        args.eval_datasets,
        args.icl_source_datasets,
        args.icl_max_demos
    ))
    
    
    for (trained_ds, eval_ds, icl_source_ds, icl_demos) in dataset_combinations:
        plot_experiment = (
            trained_ds,           # trained_dataset
            eval_ds,              # eval_dataset  
            icl_source_ds,        # icl_source_dataset
            icl_demos,            # icl_max_demos
            args.base_method,     # base_method
            args.training_variants, # training_variants
            args.include_base_model, # include_base_model
            args.uncertainty_mode,     # uncertainty_mode
            args.hp_selection,     # hp_selection
            # args.plot_types,      # plot_types
            args.base_dir,        # base_dir
            args.output_dir,      # output_dir
            args.model_id,        # model_id
            args.lora_r,          # lora_r_filter
            args.lora_type        # lora_type_filter
        )
        plot_experiments.append(plot_experiment)

    print(f"\nGenerated {len(plot_experiments)} plotting experiments")
    
    if not plot_experiments:
        print("No experiments to run!")
        return

    # Show first few experiments as examples
    print("\nFirst 3 experiments:")
    for i, exp in enumerate(plot_experiments[:3]):
        uncertainty_suffix = " (uncertainty)" if exp[7] else " (standard)"
        base_model_suffix = " + base" if exp[6] else ""
        print(f"  {i+1}. {exp[0]} -> {exp[1]} (ICL: {exp[2]}, demos: {exp[3]}, base: {exp[4]}){base_model_suffix}{uncertainty_suffix}")

    input("\nPress Enter to start plotting...")

    # Run plotting in parallel
    print(f"Running {len(plot_experiments)} plotting tasks on {args.max_parallel} processes...")
    
    start_time = time.time()
    
    with Pool(processes=args.max_parallel) as pool:
        results = pool.map(run_single_plot, plot_experiments)
    
    end_time = time.time()
    
    # Summary
    successful = sum(results)
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"PLOTTING SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(plot_experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(plot_experiments)*100:.1f}%")
    print(f"Total time: {end_time - start_time:.1f} seconds")
    print(f"{'='*60}")
    
    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'trained_datasets': args.trained_datasets,
            'eval_datasets': args.eval_datasets,
            'icl_source_datasets': args.icl_source_datasets,
            'icl_max_demos': args.icl_max_demos,
            'base_method': args.base_method,
            'training_variants': args.training_variants,
            'include_base_model': args.include_base_model,
            'uncertainty_mode': args.uncertainty_mode,
            'hp_selection': args.hp_selection,
            'max_parallel': args.max_parallel,
            'base_dir': args.base_dir,
            'output_dir': args.output_dir,
            'model_id': args.model_id,
            'lora_r': args.lora_r,
            'lora_type': args.lora_type
        },
        'total_experiments': len(plot_experiments),
        'successful': successful,
        'failed': failed,
        'success_rate': successful/len(plot_experiments)*100,
        'total_time_seconds': end_time - start_time
    }
    
    summary_path = f"../outputs/plotting_summaries/unified_batch_plotting_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 