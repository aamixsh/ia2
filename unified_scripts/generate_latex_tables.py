#!/usr/bin/env python3
"""
Generate LaTeX tables from aggregated results CSV

Creates tables in the exact format specified with ICL, SFT Only, IA2 Only, and IA2->SFT columns
for each n_value, showing accuracy and ECE metrics with standard deviations.
"""

import pandas as pd
import argparse
import os
import numpy as np

llama_model_mapping = {
    "agnews": "Llama-3.2-1B",
    "sst2": "Llama-3.2-1B",
    "sciq_remap": "Llama-3.2-3B",
    "qasc_remap": "Llama-3.2-3B",
    "finsen": "Llama-3.2-1B",
    "poems": "Llama-3.2-1B",
    "bbcnews": "Llama-3.2-1B",
    "gsm8k": "Llama-3.2-1B-Instruct",
    "gsm8ks": "Llama-3.2-1B-Instruct",
    "hmath_algebra": "Llama-3.2-1B-Instruct",
    "sciqa": "Llama-3.2-1B-Instruct",
}

qwen_model_mapping = {
    "agnews": "Qwen3-4B-Base",
    "sst2": "Qwen3-4B-Base",
    "sciq_remap": "Qwen3-4B-Base",
    "qasc_remap": "Qwen3-4B-Base",
    "finsen": "Qwen3-4B-Base",
    "poems": "Qwen3-4B-Base",
    "bbcnews": "Qwen3-4B-Base",
    "strategytf": "Qwen3-4B-Base",
    "gsm8k": "Qwen3-4B-Base",
    "gsm8ks": "Qwen3-4B-Base",
    "hmath_algebra": "Qwen3-4B-Base",
    "sciqa": "Qwen3-4B-Base",
}

# Global variables will be set in main() based on command line arguments
label_type = None
metric_type = None
model_mapping = None
uncertainty_mode = None

dataset_names = {
    "agnews": "AGN",
    "sst2": "SST2",
    "sciq_remap": "SciQr",
    "qasc_remap": "QASCr",
    "finsen": "FinS",
    "poems": "PoemS",
    "bbcnews": "BBCN",
    "strategytf": "STF",
    "gsm8k": "GSM8K",
    "gsm8ks": "GSM8Ks",
    "hmath_algebra": "HMathA",
    "sciqa": "SciQ",
}

def load_and_filter_data(csv_path, base_method):
    """Load CSV and filter based on uncertainty_mode and base_method"""
    df = pd.read_csv(csv_path)
    
    # Filter based on uncertainty_mode and base_method
    uncertainty_filter = True if uncertainty_mode == 'on' else False
    filtered_df = df[(df['uncertainty_mode'] == uncertainty_filter) & 
                        (df['base_method'] == base_method) &
                        (df['label_type'] == label_type)]
    
    return filtered_df

def extract_metrics(df, training_method, metric_prefix):
    """Extract accuracy and ECE metrics for a given training method and metric prefix"""
    method_df = df[df['training_method'] == training_method]
    
    results = {}
    for _, row in method_df.iterrows():
        n_val = row['n_value']
        eval_dataset = row['eval_dataset']
        if eval_dataset not in model_mapping:
            continue
        trained_dataset = row['trained_dataset']
        metric_name = row['metric_name']
        metric_value = row['metric_value']
        label_type_in_row = row['label_type']
        if label_type_in_row != label_type:
            continue
        model_name = row['model_name']
        # print(model_name, eval_dataset)
        if model_name != model_mapping[eval_dataset]:
            continue

        # Create key for this combination
        key = (trained_dataset, eval_dataset, n_val)
        if key not in results:
            results[key] = {}

        if uncertainty_mode == 'on':
            # Store metric values
            if metric_name == f'{metric_prefix}_accuracy_{metric_type}_mean':
                results[key]['acc_mean'] = metric_value
            elif metric_name == f'{metric_prefix}_accuracy_{metric_type}_std':
                results[key]['acc_std'] = metric_value
            elif metric_name == f'{metric_prefix}_ece_{metric_type.split('_')[0]}_mean':
                results[key]['ece_mean'] = metric_value
            elif metric_name == f'{metric_prefix}_ece_{metric_type.split('_')[0]}_std':
                results[key]['ece_std'] = metric_value

        else:
            if metric_name == f'{metric_prefix}_accuracy_mean':
                results[key]['acc_mean'] = metric_value
            elif metric_name == f'{metric_prefix}_accuracy_std':
                results[key]['acc_std'] = metric_value
    
    return results

def format_metric(mean_val, std_val, is_best=False, is_acc=True):
    """Format a single metric with mean and standard deviation"""
    if pd.isna(mean_val) or pd.isna(std_val):
        return "N/A"
    if is_best:
        if is_acc:
            return f"\\textbf{{{mean_val:04.1f}}} \\tiny{{({std_val:04.1f})}}"
        else:
            return f"\\textbf{{{mean_val:04.2f}}} \\tiny{{({std_val:04.2f})}}"
    else:
        if is_acc:
            return f"{mean_val:04.1f} \\tiny{{({std_val:04.1f})}}"
        else:
            return f"{mean_val:04.2f} \\tiny{{({std_val:04.2f})}}"

def generate_latex_table(n_value, results, training_methods, model_type, use_icl_for_best, base_method):
    """Generate LaTeX table for a specific n_value with dynamic columns based on selected methods"""
    
    # Get all unique dataset combinations for this n_value
    all_keys = set()
    for method_results in results.values():
        all_keys.update([k for k in method_results.keys() if k[2] == n_value])
    
    if not all_keys:
        return ""
    
    # Group by trained_dataset to create multirow structure
    grouped_data = {}
    for trained_dataset, eval_dataset, n_val in all_keys:
        if trained_dataset not in grouped_data:
            grouped_data[trained_dataset] = []
        grouped_data[trained_dataset].append((trained_dataset, eval_dataset, n_val))
    
    # Sort by trained_dataset, then eval_dataset
    for trained_dataset in grouped_data:
        grouped_data[trained_dataset].sort(key=lambda x: x[1])
    
    sorted_trained_datasets = sorted(grouped_data.keys())
    
    # Determine number of columns based on uncertainty mode and selected methods
    num_methods = len(training_methods)
    if uncertainty_mode == 'on':
        cols_per_method = 2  # acc and ece
    else:
        cols_per_method = 1  # only acc
    
    total_cols = 2 + (num_methods * cols_per_method)  # 2 for source and eval
    
    # Method display names
    method_names = {
        'w_o_icl': 'w/o ICL',
        'icl': 'ICL',
        'sft': 'SFT only',
        'ia2': '\\act only',
        'ia2_sft': '\\act $\\rightarrow$ SFT',
        'tna': '\\act + SFT'
    }
    
    latex = f"\\begin{{table*}}[ht]\n"
    latex += f"    \\centering\n"
    latex += f"    \\small\n"
    latex += f"    \\resizebox{{\\linewidth}}{{!}}{{\n"
    # latex += f"\\begin{{tabular}}{{\n"
    latex += f"\\begin{{tabular}}{{"
    
    # Build column specification
    # latex += f"P{{3cm}}\n"  # Source column
    latex += "cc"  # Source/Eval column
    
    for i, method in enumerate(training_methods):
        # latex += f"P{{3cm}}\n"  # acc/ece column
        if uncertainty_mode == 'on':
            # Two columns per method: acc and ece
            latex += "cc"  # acc/ece column
        else:
            # One column per method: only acc
            latex += "c"  # acc column
    
    latex += f"}}\n"
    latex += f"    \\toprule\n"
    
    # Header row 1: Dataset and Adaptation Method
    latex += f"    \\multicolumn{{2}}{{c}}{{ Dataset }} &\n"
    latex += f"     \\multicolumn{{{total_cols - 2}}}{{c}}{{ Adaptation Method }} \\\\\n"
    latex += f"     \\cmidrule(lr){{1-{total_cols}}}\n"
    
    # Header row 2: Source, Eval, and method names
    latex += f"      \\multirow{{2}}{{*}}{{Source}} & \\multirow{{2}}{{*}}{{Eval}}"
    for method in training_methods:
        if uncertainty_mode == 'on':
            latex += f" & \\multicolumn{{2}}{{c}}{{{method_names[method]}}}"
        else:
            latex += f" & {method_names[method]}"
    latex += f" \\\\\n"
    
    # Header row 3: acc and ece labels
    latex += f"      &"
    for method in training_methods:
        if uncertainty_mode == 'on':
            latex += f" & acc $\\uparrow$ & ece $\\downarrow$"
        else:
            latex += f" & acc $\\uparrow$"
    latex += f" \\\\\n"
    
    latex += f"     \\midrule\n"
    latex += f"     \\rule{{0pt}}{{2ex}}\n"

    num_rows = 0
    
    for i, trained_dataset in enumerate(sorted_trained_datasets):
        eval_combinations = grouped_data[trained_dataset]
        trained_name = dataset_names.get(trained_dataset, trained_dataset)
        
        for j, (trained_dataset, eval_dataset, n_val) in enumerate(eval_combinations):
            # Get results for each method
            key = (trained_dataset, eval_dataset, n_val)
            
            # Collect all metric values for comparison
            all_acc_values = []
            all_ece_values = []
            method_metrics = {}
            
            for method in training_methods:
                method_results = results.get(method, {})
                method_data = method_results.get(key, {})
                
                acc_mean = method_data.get('acc_mean')
                acc_std = method_data.get('acc_std')
                ece_mean = method_data.get('ece_mean')
                ece_std = method_data.get('ece_std')
                
                method_metrics[method] = {
                    'acc_mean': acc_mean,
                    'acc_std': acc_std,
                    'ece_mean': ece_mean,
                    'ece_std': ece_std
                }

                if (method in ['icl', 'w_o_icl'] and use_icl_for_best) or (method not in ['icl', 'w_o_icl']):
                    all_acc_values.append(acc_mean)
                    if uncertainty_mode == 'on':
                        all_ece_values.append(ece_mean)
                else:
                    all_acc_values.append(-np.inf)
                    if uncertainty_mode == 'on':
                        all_ece_values.append(np.inf)

            # Find best values
            acc_values_clean = [v for v in all_acc_values if not pd.isna(v)]
            ece_values_clean = [v for v in all_ece_values if not pd.isna(v)]

            if len(acc_values_clean) in [0, 1]:
                # Only ICL results, or no results. skip.
                continue
            if 'w_o_icl' in training_methods and len(acc_values_clean) in [0, 2]:
                # Only w_o_icl and icl results, or no results. skip.
                continue
            
            best_acc_idx = None
            best_ece_idx = None
            
            if acc_values_clean:
                best_acc_idx = all_acc_values.index(max(acc_values_clean))
            if ece_values_clean:
                best_ece_idx = all_ece_values.index(min(ece_values_clean))

            # Format metrics with best indicators
            formatted_metrics = {}
            for idx, method in enumerate(training_methods):
                metrics = method_metrics[method]
                formatted_metrics[method] = {
                    'acc': format_metric(metrics['acc_mean'], metrics['acc_std'], 
                                       is_best=(best_acc_idx == idx)),
                    'ece': format_metric(metrics['ece_mean'], metrics['ece_std'], 
                                       is_best=(best_ece_idx == idx), is_acc=False)
                }
            
            # Add row to table
            eval_name = dataset_names.get(eval_dataset, eval_dataset)
            
            # Add asterisk for cross-domain evaluation
            if eval_dataset != trained_dataset:
                eval_name += "$^*$"
            
            if j == 0:
                # First row for this trained dataset - use multirow
                row_content = f"    \\multirow{{{len(eval_combinations)}}}{{*}}{{{trained_name}}} & {eval_name}"
            else:
                # Subsequent rows - no multirow for source
                row_content = f"    & {eval_name}"
            
            # Add method columns
            for method in training_methods:
                if uncertainty_mode == 'on':
                    row_content += f" & {formatted_metrics[method]['acc']} & {formatted_metrics[method]['ece']}"
                else:
                    row_content += f" & {formatted_metrics[method]['acc']}"
            
            row_content += f" \\\\\n"
            latex += row_content
            num_rows += 1
        # Add midrule between different trained datasets (except after the last one)
        if i < len(sorted_trained_datasets) - 1:
            latex += f"    \\midrule\n"
    
    if num_rows == 0:
        return None
    
    latex += f"    \\bottomrule\n"
    latex += f"    \n"
    latex += f"    \\end{{tabular}}}}\n"

    model_family = "Qwen3-4B-Base"
    if model_type == 'llama':
       model_family = 'Llama-3.2'

    metric_type_str = "top-1 token" if metric_type == 'top1' else "top label-set token"
    base_method_str = "LoRA" if base_method == 'lora' else "IA3"
    label_type_str = "Ground Truth" if label_type == 'ground_truth' else "ICL outputs"

    # Dynamic caption based on uncertainty mode
    if uncertainty_mode == 'on':
        caption = f"    Performance report for $N={n_value}$ on {model_family} models trained using {base_method_str}. Tokens used for training: {label_type_str}. Showing accuracy (acc, higher is better) and Expected Calibration Error (ece, lower is better). Best training method shown in bold. Metrics based on {metric_type_str}.\n"
    else:
        caption = f"    Performance report for $N={n_value}$ on {model_family} models trained using {base_method_str}. Tokens used for training: {label_type_str}. Showing accuracy (acc, higher is better). Best training method shown in bold.\n"
    
    caption += f"    Numbers in parentheses show standard deviations across runs. Datasets marked with $^*$ are OOD evaluations. See details in~\\autoref{{sec:exp}}.\n"
    
    latex += f"    \\caption{{\n"
    latex += caption
    latex += f"    }}\n"
    latex += f"    \\label{{tab:results_n{n_value}}}\n"
    latex += f"\\end{{table*}}\n\n"
    
    return latex

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from aggregated results")
    parser.add_argument("--csv_path", type=str, default="../plots/aggregated/all_results.csv",
                        help="Path to the aggregated results CSV file")
    parser.add_argument("--output_dir", type=str, default="../plots/latex_tables",
                        help="Output directory for LaTeX files")
    parser.add_argument("--uncertainty_mode", type=str, choices=['on', 'off'], default='on',
                        help="Uncertainty mode: 'on' for multi-token, 'off' for single-token")
    parser.add_argument("--base_method", type=str, choices=['lora', 'ia3'], default='lora',
                        help="Base method: 'lora' or 'ia3'")
    parser.add_argument("--label_type", type=str, choices=['ground_truth', 'icl_outputs'], default='icl_outputs',
                        help="Label type: 'ground_truth' or 'icl_outputs'")
    parser.add_argument("--metric_type", type=str, choices=['top1', 'label_set'], default='top1',
                        help="Metric type: 'top1' or 'label_set'")
    parser.add_argument("--model_type", type=str, choices=['llama', 'qwen'], default='llama',
                        help="Model type: 'llama' or 'qwen'")
    parser.add_argument("--use_icl_for_best", action='store_true', default=False,
                        help="Use ICL for best evaluation")
    parser.add_argument("--training_methods", nargs='+', 
                        choices=['w_o_icl', 'icl', 'sft', 'ia2', 'ia2_sft', 'tna'], 
                        default=['icl', 'sft', 'ia2', 'ia2_sft', 'tna'],
                        help="Training methods to include in table")
    
    args = parser.parse_args()
    
    # Set global variables based on command line arguments
    global label_type, metric_type, model_mapping, qwen_model_mapping, llama_model_mapping, uncertainty_mode
    label_type = args.label_type
    metric_type = args.metric_type
    uncertainty_mode = args.uncertainty_mode
    
    # Select model mapping based on model type
    if args.model_type == 'llama':
        model_mapping = llama_model_mapping
    else:
        model_mapping = qwen_model_mapping
    
    # Automatically add w_o_icl when uncertainty_mode is off
    if uncertainty_mode == 'off' and 'w_o_icl' not in args.training_methods:
        args.training_methods.insert(0, 'w_o_icl')  # Insert at the beginning
    
    print(f"Using {args.model_type} model mapping")
    print(f"Uncertainty mode: {uncertainty_mode}")
    print(f"Label type: {label_type}")
    print(f"Metric type: {metric_type}")
    print(f"Training methods: {args.training_methods}")
    
    print("Loading and filtering data...")
    df = load_and_filter_data(args.csv_path, args.base_method)
    print(f"Loaded {len(df)} rows after filtering")
    
    # Define method mappings
    method_mappings = {
        'w_o_icl': ('Base Model', 'without_icl'),
        'icl': ('Base Model', 'with_icl'),
        'sft': (f'{args.base_method} Token Training', 'without_icl'),
        'ia2': (f'{args.base_method} Activation Training', 'without_icl'),
        'ia2_sft': (f'{args.base_method} Sequential (act→tok) Training', 'without_icl'),
        'tna': (f'{args.base_method} Token + Activation Training', 'without_icl')
    }
    
    # Extract results for selected methods
    results = {}
    for method in args.training_methods:
        if method in method_mappings:
            training_method, metric_prefix = method_mappings[method]
            print(f"Extracting {method.upper()} results...")
            results[method] = extract_metrics(df, training_method, metric_prefix)
    
    # Get unique n_values
    n_values = sorted(df['n_value'].unique())
    print(f"Found n_values: {n_values}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate tables for each n_value
    for n_val in n_values:
        print(f"Generating table for N={n_val}...")
        latex_table = generate_latex_table(n_val, results, args.training_methods, args.model_type, args.use_icl_for_best, args.base_method)
        
        if latex_table:
            # Generate dynamic filename
            uncertainty_suffix = "st" if uncertainty_mode == 'on' else "mt"
            label_suffix = "gt" if label_type == 'ground_truth' else "io"
            if uncertainty_mode == 'on':
                filename = f"{args.base_method}_{args.model_type}_{uncertainty_suffix}_{args.metric_type.split('_')[0]}_{n_val}_{label_suffix}.tex"
            else:
                filename = f"{args.base_method}_{args.model_type}_{uncertainty_suffix}_{n_val}_{label_suffix}.tex"
            output_file = os.path.join(args.output_dir, filename)
            
            with open(output_file, 'w') as f:
                f.write(latex_table)
            print(f"Saved table to {output_file}")
        else:
            print(f"No data found for N={n_val}")
    
    # # Generate combined file with all tables
    # uncertainty_suffix = "mt" if uncertainty_mode == 'on' else "st"
    # label_suffix = "gt" if label_type == 'ground_truth' else "io"
    # combined_filename = f"{args.model_type}_{uncertainty_suffix}_all_{label_suffix}.tex"
    # combined_file = os.path.join(args.output_dir, combined_filename)
    
    # with open(combined_file, 'w') as f:
    #     f.write("\\documentclass{article}\n")
    #     f.write("\\usepackage{booktabs}\n")
    #     f.write("\\usepackage{multirow}\n")
    #     f.write("\\usepackage{array}\n")
    #     f.write("\\usepackage{amsmath}\n")
    #     f.write("\\usepackage{graphicx}\n")
    #     f.write("\\begin{document}\n\n")
        
    #     for n_val in n_values:
    #         latex_table = generate_latex_table(n_val, results, args.training_methods, args.model_type)
    #         if latex_table:
    #             f.write(latex_table)
        
    #     f.write("\\end{document}\n")
    
    # print(f"Saved combined file to {combined_file}")
    print("Done!")

if __name__ == "__main__":
    main()