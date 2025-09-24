#!/usr/bin/env python3
"""
Unified Subspace Overlap Analysis Script

This script analyzes subspace overlap between different LoRA training approaches:
- tok: Traditional token-based fine-tuning
- act: Activation-based training
- tna: Token + Activation training
- a2t: Sequential training (act→tok)

Features:
1. Argparse-based configuration
2. Serial processing with efficient model loading (base model loaded once)
3. Hyperparameter selection based on performance or dev loss
4. Comprehensive overlap analysis across layers and attention heads
5. Standard deviation calculation for error bars

Uses the same naming conventions as evaluate_batch_optimized.py
"""

import os
import json
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import itertools
import glob
from transformers import AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import time
from functools import partial
import re
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import multiprocessing as mp

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Import evaluation results and hyperparameter functions from plotting scripts
from plot_unified import (
    process_trained_model_results_full,
    find_best_hyperparameters_by_performance,
    find_best_hyperparameters_by_dev_loss,
    load_dev_loss_statistics
)

# Import get_model_name from utils for exact model naming
from utils import get_model_name


@dataclass
class SubspaceAnalysisConfig:
    """Configuration for subspace overlap analysis run"""
    model_id: str
    trained_dataset: str
    eval_dataset: str
    icl_source: str
    icl_demos: int
    label_type: str
    hp_selection: str
    lora_type: str
    lora_r: int
    lora_alpha: int
    num_generated_tokens: int
    device: str
    output_dir: str
    uncertainty: bool = False


def _base_storage_dir(config: SubspaceAnalysisConfig) -> str:
    return "../outputs/subspace"


def get_bases_storage_path(config: SubspaceAnalysisConfig, n_examples: int, run_idx: int, model_type: str) -> str:
    base_dir = _base_storage_dir(config)
    path = f"{base_dir}/{config.model_id.split('/')[-1]}/{config.eval_dataset}/{config.icl_source}_{config.icl_demos}/{config.label_type}/{n_examples}_{run_idx}/{model_type}/bases"
    return path


def get_overlap_storage_path(config: SubspaceAnalysisConfig, n_examples: int, run_idx: int) -> str:
    base_dir = _base_storage_dir(config)
    path = f"{base_dir}/{config.model_id.split('/')[-1]}/{config.eval_dataset}/{config.icl_source}_{config.icl_demos}/{config.label_type}/{n_examples}_{run_idx}/overlaps"
    return path


def get_hyperparam_storage_path(config: SubspaceAnalysisConfig) -> str:
    base_dir = _base_storage_dir(config)
    path = f"{base_dir}/{config.model_id.split('/')[-1]}/{config.eval_dataset}/{config.icl_source}_{config.icl_demos}/{config.label_type}/hyperparameters"
    return path


def save_bases(bases: Dict, config: SubspaceAnalysisConfig, n_examples: int, run_idx: int, model_type: str) -> str:
    storage_path = get_bases_storage_path(config, n_examples, run_idx, model_type)
    os.makedirs(storage_path, exist_ok=True)
    file_path = os.path.join(storage_path, "bases.pkl")
    # Move tensors to cpu and float for portability
    safe_bases = {}
    for layer_idx, head_map in bases.items():
        safe_bases[layer_idx] = {}
        for head_type, tensor in head_map.items():
            safe_bases[layer_idx][head_type] = tensor.detach().cpu() if tensor is not None else None
    with open(file_path, 'wb') as f:
        pickle.dump(safe_bases, f)
    return file_path


def load_bases(config: SubspaceAnalysisConfig, n_examples: int, run_idx: int, model_type: str) -> Optional[Dict]:
    storage_path = get_bases_storage_path(config, n_examples, run_idx, model_type)
    file_path = os.path.join(storage_path, "bases.pkl")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading bases from {file_path}: {e}")
        return None


def save_overlap_results(overlaps: Dict, config: SubspaceAnalysisConfig, n_examples: int, run_idx: int) -> str:
    storage_path = get_overlap_storage_path(config, n_examples, run_idx)
    os.makedirs(storage_path, exist_ok=True)
    file_path = os.path.join(storage_path, "overlaps.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(overlaps, f)
    return file_path


def load_overlap_results(config: SubspaceAnalysisConfig, n_examples: int, run_idx: int) -> Optional[Dict]:
    storage_path = get_overlap_storage_path(config, n_examples, run_idx)
    file_path = os.path.join(storage_path, "overlaps.pkl")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading overlaps from {file_path}: {e}")
        return None


def save_hyperparameters(hyperparams: Dict, config: SubspaceAnalysisConfig) -> str:
    storage_path = get_hyperparam_storage_path(config)
    os.makedirs(storage_path, exist_ok=True)
    file_path = os.path.join(storage_path, "selected_hyperparameters.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(hyperparams, f)
    return file_path


def load_hyperparameters(config: SubspaceAnalysisConfig) -> Optional[Dict]:
    storage_path = get_hyperparam_storage_path(config)
    file_path = os.path.join(storage_path, "selected_hyperparameters.pkl")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading hyperparameters from {file_path}: {e}")
        return None


def check_complete_analysis_exists(config: SubspaceAnalysisConfig, n_examples: int, run_idx: int) -> bool:
    # Require hyperparameters plus per-run overlaps
    if load_hyperparameters(config) is None:
        return False
    if load_overlap_results(config, n_examples, run_idx) is None:
        return False
    return True

def get_lora_subspace_bases(peft_adapter_path, base_model, device, torch_dtype, lora_rank):
    """Loads a PEFT adapter and computes the subspace basis U for each lora_B matrix via SVD"""
    if not peft_adapter_path or not os.path.isdir(peft_adapter_path):
        return None

    bases = {}  # {layer_idx: {'q': Uq, 'k': Uk, 'v': Uv, 'o': Uo}}

    try:
        # Load the PEFT model/adapter using the provided base model
        model = PeftModel.from_pretrained(base_model, peft_adapter_path, torch_dtype=torch_dtype)
        model = model.to(device)
        model.eval()

        # Extract lora_B weights and compute bases
        for name, module in model.named_modules():
            if 'lora_B' in name and hasattr(module, 'weight'):
                # Extract layer_idx, and head_type from name
                match = re.search(r'layers\.(\d+)\.self_attn\.([qkov]_proj)\.lora_B', name)
                if match:
                    layer_idx = int(match.group(1))
                    head_type = match.group(2)[0] # q, k, v, or o

                    lora_B = module.weight.data.float() # Use float32 for SVD stability
                    
                    try:
                        # SVD: B = U S Vh. The basis is the first r columns of U.
                        U, S, Vh = torch.linalg.svd(lora_B, full_matrices=False)
                        basis_U = U[:, :lora_rank]

                        if layer_idx not in bases:
                            bases[layer_idx] = {}
                        bases[layer_idx][head_type] = basis_U
                    except Exception as e:
                        print(f"Error during SVD for {name} in {peft_adapter_path}: {e}")
                        if layer_idx not in bases: bases[layer_idx] = {}
                        bases[layer_idx][head_type] = None # Mark as failed

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error loading model or computing bases for {peft_adapter_path}: {e}")
        return None

    if not bases:
        return None
    
    return bases


def calculate_subspace_overlap(U1, U2, rank):
    """Calculates subspace overlap: (1/r) * ||U1^T * U2||_F^2"""
    if U1 is None or U2 is None or rank is None or rank <= 0:
        return None
    if U1.shape[0] != U2.shape[0] or U1.shape[1] != rank or U2.shape[1] != rank:
         return None

    overlap_matrix = torch.matmul(U1.T, U2)
    frobenius_norm_sq = torch.norm(overlap_matrix, p='fro')**2
    overlap = frobenius_norm_sq.item() / rank
    # Clamp overlap between 0 and 1 due to potential numerical precision issues
    return max(0.0, min(1.0, overlap))


def process_overlap_job(args_tuple):
    """Process a single overlap job with caching support. Expects pre-extracted CPU weights or cached bases."""
    (config_dict, n_examples, run_idx, model_weights_by_type, lora_r, reuse_cache, force_recompute, device) = args_tuple
    cfg = SubspaceAnalysisConfig(**config_dict)

    # Reuse overlaps if available
    if reuse_cache and not force_recompute:
        cached = load_overlap_results(cfg, n_examples, run_idx)
        if cached is not None:
            return {
                'num_examples': n_examples,
                'run_idx': run_idx,
                'pair_averages': cached['pair_averages'],
                'layer_head_overlaps': cached['layer_head_overlaps']
            }

    required_models = ['tok', 'act', 'tna', 'a2t']

    # Compute or load bases for each model from cached or from provided weights
    bases = {}
    for model_type in required_models:
        # Try cache first
        cached_bases = load_bases(cfg, n_examples, run_idx, model_type) if (reuse_cache and not force_recompute) else None
        if cached_bases is not None:
            bases[model_type] = cached_bases
            continue
        # Compute SVD bases from provided CPU weights
        weights = model_weights_by_type.get(model_type)
        if weights is None:
            return None
        computed_bases = {}
        for layer_idx, head_map in weights.items():
            if layer_idx not in computed_bases:
                computed_bases[layer_idx] = {}
            for head_type, lora_B in head_map.items():
                if lora_B is None:
                    computed_bases[layer_idx][head_type] = None
                    continue
                try:
                    # Move to target device for fast SVD, then bring U back to CPU
                    mat = lora_B.detach().cpu().float().to(device)
                    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                    U_cpu = U[:, :lora_r].detach().to('cpu')
                    computed_bases[layer_idx][head_type] = U_cpu
                except Exception as e:
                    print(f"SVD error (layer {layer_idx}, head {head_type}) for {model_type}: {e}")
                    computed_bases[layer_idx][head_type] = None
        bases[model_type] = computed_bases
        # Save for reuse
        save_bases(computed_bases, cfg, n_examples, run_idx, model_type)

    # Validate
    if not all(bases.get(m) for m in required_models):
        return None

    # Calculate pairwise overlaps
    layer_head_overlaps = {}
    for layer_idx, head_data in bases['tok'].items():
        for head_type in head_data.keys():
            U_tok = bases.get('tok', {}).get(layer_idx, {}).get(head_type)
            U_act = bases.get('act', {}).get(layer_idx, {}).get(head_type)
            U_tna = bases.get('tna', {}).get(layer_idx, {}).get(head_type)
            U_a2t = bases.get('a2t', {}).get(layer_idx, {}).get(head_type)

            pairs_to_check = {
                'tok_vs_act': (U_tok, U_act),
                'tok_vs_tna': (U_tok, U_tna),
                'tok_vs_a2t': (U_tok, U_a2t),
                'act_vs_tna': (U_act, U_tna),
                'act_vs_a2t': (U_act, U_a2t),
                'tna_vs_a2t': (U_tna, U_a2t),
            }

            for pair_name, (U1, U2) in pairs_to_check.items():
                overlap = calculate_subspace_overlap(U1, U2, lora_r)
                if overlap is not None:
                    if pair_name not in layer_head_overlaps:
                        layer_head_overlaps[pair_name] = {}
                    if layer_idx not in layer_head_overlaps[pair_name]:
                        layer_head_overlaps[pair_name][layer_idx] = {}
                    if head_type not in layer_head_overlaps[pair_name][layer_idx]:
                        layer_head_overlaps[pair_name][layer_idx][head_type] = []
                    layer_head_overlaps[pair_name][layer_idx][head_type].append(overlap)

    pair_averages = {}
    for pair_name, layer_data in layer_head_overlaps.items():
        all_overlaps = []
        for layer_idx, head_map in layer_data.items():
            for head_type, overlaps in head_map.items():
                all_overlaps.extend(overlaps)
        pair_averages[pair_name] = (np.mean(all_overlaps) if all_overlaps else None)

    # Save per-run overlaps
    save_overlap_results({'pair_averages': pair_averages, 'layer_head_overlaps': layer_head_overlaps}, cfg, n_examples, run_idx)

    return {
        'num_examples': n_examples,
        'run_idx': run_idx,
        'pair_averages': pair_averages,
        'layer_head_overlaps': layer_head_overlaps
    }


def plot_overlap_results(results, output_dir, eval_dataset, icl_source, label_type, hp_selection, model_id):
    """Create comprehensive plots for overlap analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define consistent markers and colors for head types
    head_markers = {'q': 'o', 'k': 's', 'v': '^', 'o': 'D'}
    head_colors = {'q': 'blue', 'k': 'red', 'v': 'green', 'o': 'orange'}
    
    # Get all unique pair names - using the actual model type names from evaluate_batch_optimized
    all_pair_names = ['tok_vs_act', 'tok_vs_tna', 'tok_vs_a2t', 'act_vs_tna', 'act_vs_a2t', 'tna_vs_a2t']
    
    # ============ PLOT 1: Overall average overlaps ============
    plt.figure(figsize=(12, 8))
    
    for pair_name in all_pair_names:
        n_values = []
        means = []
        stds = []
        
        for n_examples in sorted(results.keys()):
            result = results[n_examples]
            if result and pair_name in result['pair_averages']:
                n_values.append(n_examples)
                # Handle new structure with mean and std
                if isinstance(result['pair_averages'][pair_name], dict):
                    means.append(result['pair_averages'][pair_name]['mean'])
                    stds.append(result['pair_averages'][pair_name]['std'])
                else:
                    # Backward compatibility for old structure
                    means.append(result['pair_averages'][pair_name])
                    # Calculate std from layer/head overlaps
                    all_overlaps = []
                    if pair_name in result['layer_head_overlaps']:
                        for layer_data in result['layer_head_overlaps'][pair_name].values():
                            for overlaps in layer_data.values():
                                all_overlaps.extend(overlaps)
                    stds.append(np.std(all_overlaps) if all_overlaps else 0.0)
        
        if n_values:
            # Use proper model type names for labels
            label_map = {
                'tok_vs_act': 'Token vs Activation',
                'tok_vs_tna': 'Token vs Token+Activation', 
                'tok_vs_a2t': 'Token vs Sequential A2T',
                'act_vs_tna': 'Activation vs Token+Activation',
                'act_vs_a2t': 'Activation vs Sequential A2T',
                'tna_vs_a2t': 'Token+Activation vs Sequential A2T'
            }
            label = label_map.get(pair_name, pair_name.replace('_vs_', ' vs '))
            plt.errorbar(n_values, means, yerr=stds, 
                       marker='o', linestyle='-', label=label, 
                       capsize=3, capthick=1)

    plt.xlabel("Number of Training Examples")
    plt.ylabel("Average Subspace Overlap")
    plt.title(f"LoRA Subspace Overlap vs. Training Examples\nDataset: {eval_dataset}, ICL Source: {icl_source}, Label Type: {label_type}, Model: {model_id.split('/')[-1]}")
    plt.xscale('log')
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()

    plot_filename = f"overlap_overall_{eval_dataset}_{icl_source}_{model_id.split('/')[-1]}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Overall plot saved to {plot_path}")

    # ============ PLOT 2: By head type ============
    for pair_name in all_pair_names:
        plt.figure(figsize=(12, 8))
        
        # Get all unique head types
        all_head_types = set()
        for n_examples, result in results.items():
            if result and pair_name in result['layer_head_overlaps']:
                for layer_data in result['layer_head_overlaps'][pair_name].values():
                    all_head_types.update(layer_data.keys())
        
        for head_type in sorted(all_head_types):
            head_means = []
            head_errors = []
            n_values = []
            
            for n_examples in sorted(results.keys()):
                result = results[n_examples]
                if result and pair_name in result['layer_head_overlaps']:
                    # Collect all overlaps for this head type across all layers
                    head_overlaps = []
                    for layer_data in result['layer_head_overlaps'][pair_name].values():
                        if head_type in layer_data:
                            head_overlaps.extend(layer_data[head_type])
                    
                    if head_overlaps:
                        head_means.append(np.mean(head_overlaps))
                        head_errors.append(np.std(head_overlaps))
                        n_values.append(n_examples)
            
            if head_means:
                marker = head_markers.get(head_type, 'o')
                color = head_colors.get(head_type, 'black')
                plt.errorbar(n_values, head_means, yerr=head_errors,
                           marker=marker, linestyle='-', label=f'Head {head_type}', 
                           capsize=3, capthick=1, color=color)

        plt.xlabel("Number of Training Examples")
        plt.ylabel("Average Subspace Overlap")
        # Use proper model type names for titles
        title_map = {
            'tok_vs_act': 'Token vs Activation',
            'tok_vs_tna': 'Token vs Token+Activation', 
            'tok_vs_a2t': 'Token vs Sequential A2T',
            'act_vs_tna': 'Activation vs Token+Activation',
            'act_vs_a2t': 'Activation vs Sequential A2T',
            'tna_vs_a2t': 'Token+Activation vs Sequential A2T'
        }
        title = title_map.get(pair_name, pair_name.replace('_vs_', ' vs '))
        plt.title(f"LoRA Subspace Overlap by Head Type: {title}\nDataset: {eval_dataset}, ICL Source: {icl_source}, Label Type: {label_type}, Model: {model_id.split('/')[-1]}")
        plt.xscale('log')
        plt.ylim(0, 1.05)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='best')
        plt.tight_layout()

        plot_filename = f"overlap_by_head_{pair_name}_{eval_dataset}_{icl_source}_{model_id.split('/')[-1]}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Head type plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified subspace overlap analysis")
    
    # Data configuration
    parser.add_argument("--trained_datasets", nargs='+', default=["agnews"],
                        help="Datasets the models were trained on")
    parser.add_argument("--eval_datasets", nargs='+', default=["agnews", "bbcnews"],
                        help="Datasets to evaluate on")
    parser.add_argument("--icl_sources", nargs='+', default=["agnews"],
                        help="ICL source datasets")
    parser.add_argument("--label_types", nargs='+', default=["icl_outputs", "ground_truth"],
                        choices=["ground_truth", "icl_outputs"],
                        help="Label types used during training")
    parser.add_argument("--num_generated_tokens", type=int, default=1,
                        help="Number of generated tokens during training")
    parser.add_argument("--uncertainty", action="store_true", default=False,
                        help="Whether to use uncertainty")
    
    # Model configuration
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Base",
                        help="Base model ID")
    parser.add_argument("--lora_type", type=str, default="qko",
                        help="LoRA type")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=8,
                        help="LoRA alpha")
    
    # Evaluation configuration
    # Note: eval_datasets and icl_sources are declared above to mirror activation script
    parser.add_argument("--icl_demos", type=int, default=256,
                        help="Number of ICL demos")
    parser.add_argument("--base_dir", type=str, default="../outputs/evaluations",
                        help="Base directory for evaluation results")
    parser.add_argument("--num_train_examples", nargs='+', type=int, default=[2,4,8,16],
                        help="List of N (training examples) to include if available")
    parser.add_argument("--run_indices", nargs='+', type=int, default=[0,1,2,3,4],
                        help="Run indices to include")
    
    # Analysis options
    parser.add_argument("--hp_selection", type=str, choices=['performance', 'dev_loss'], default='performance',
                        help="Hyperparameter selection method")

    parser.add_argument("--device", type=str, default="cuda:3",
                        help="Device to use for computation")
    parser.add_argument("--reuse_cache", action="store_true", default=True,
                        help="Reuse previously calculated bases/overlaps")
    parser.add_argument("--force_recompute", action="store_true", default=False,
                        help="Force recomputation even if cache exists")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Number of parallel processes to use")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="../plots/subspace_analysis",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Setup device and dtype
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16
    
    # Generate all configuration combinations (mirror activation script)
    configs = []
    for trained_dataset in args.trained_datasets:
        for eval_dataset in args.eval_datasets:
            for label_type in args.label_types:
                for icl_source in args.icl_sources:
                    configs.append((trained_dataset, eval_dataset, label_type, icl_source))

    total_combos = 0
    for trained_dataset, eval_dataset, label_type, icl_source in configs:
            # Build config for storage/caching per combo
            config = SubspaceAnalysisConfig(
                model_id=args.model_id,
                trained_dataset=trained_dataset,
                eval_dataset=eval_dataset,
                icl_source=icl_source,
                icl_demos=args.icl_demos,
                label_type=label_type,
                hp_selection=args.hp_selection,
                lora_type=args.lora_type,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                num_generated_tokens=args.num_generated_tokens,
                device=args.device,
                output_dir=args.output_dir,
                uncertainty=args.uncertainty
            )

            # Step 1: Find best hyperparameters for each training variant and N using shared utilities
            print("Finding best hyperparameters for each training variant...")
            all_selected_models = {}
            all_best_hyperparams = {}
            for model_type in ['tok', 'act', 'tna', 'a2t']:
                print(f"\nProcessing {model_type}...")
                method_data, best_hps, available_metrics = process_trained_model_results_full(
                    model_id=args.model_id,
                    base_dir=args.base_dir,
                    model_type=model_type,
                    trained_dataset=trained_dataset,
                    eval_dataset=eval_dataset,
                    icl_source=icl_source,
                    icl_demos=args.icl_demos,
                    uncertainty_mode=args.uncertainty,
                    label_type=label_type,
                    hp_selection=args.hp_selection
                )

                if not method_data:
                    print(f"No evaluation results found for {model_type}")
                    continue

                all_selected_models[model_type] = method_data
                all_best_hyperparams[model_type] = best_hps
                print(f"Selected {len(method_data)} best configurations for {model_type}")

            if not all_selected_models:
                print("No models found for any training variant! Skipping.")
                continue

            # Persist selected hyperparameters for reuse checks and reproducibility
            try:
                save_hyperparameters(all_best_hyperparams, config)
                print("Saved selected hyperparameters for this configuration.")
            except Exception as e:
                print(f"Warning: failed to save hyperparameters cache: {e}")

            # Step 2: Determine available N and intersect with requested
            all_n_values = set()
            for _, method_data in all_selected_models.items():
                all_n_values.update(method_data.keys())
            available_n_values = sorted(list(all_n_values))
            chosen_n_values = [n for n in args.num_train_examples if n in available_n_values]
            if not chosen_n_values:
                print(f"No requested N values present. Available: {available_n_values}. Skipping.")
                continue
            print(f"Using N values: {chosen_n_values}")

            # Step 3: Generate overlap jobs for corresponding runs
            overlap_jobs = []
            preload_results = []
            for n_examples in tqdm(chosen_n_values, desc="Generating overlap jobs"):
                # Ensure all variants have this N
                if not all(n_examples in all_selected_models.get(mt, {}) for mt in ['tok','act','tna','a2t']):
                    continue
                for run_idx in args.run_indices:
                    # If overlaps exist already, skip
                    if args.reuse_cache and not args.force_recompute and check_complete_analysis_exists(config, n_examples, run_idx):
                        print(f"Overlaps already exist for N={n_examples}, run={run_idx}. Reusing for plotting.")
                        cached = load_overlap_results(config, n_examples, run_idx)
                        if cached is not None:
                            preload_results.append({
                                'num_examples': n_examples,
                                'run_idx': run_idx,
                                'pair_averages': cached.get('pair_averages', {}),
                                'layer_head_overlaps': cached.get('layer_head_overlaps', {})
                            })
                        continue
                    # Build CPU lora_B weights for each model type in main process
                    model_weights_by_type = {}
                    lora_r = args.lora_r
                    # Load base model on CPU once
                    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
                    base_model = base_model.to('cpu')
                    base_model.eval()
                    for model_type in ['tok', 'act', 'tna', 'a2t']:
                        best_hps = all_best_hyperparams[model_type][n_examples]
                        metric_name = 'without_icl_accuracy_top1' if args.num_generated_tokens == 1 else 'without_icl_accuracy'
                        lr_value = best_hps[metric_name][0] if model_type == 'tna' else best_hps[metric_name]
                        ce_wt = best_hps[metric_name][1] if model_type == 'tna' else None
                        model_name = get_model_name(
                            training_method=model_type,
                            model_name=args.model_id.split('/')[-1],
                            lora_type=args.lora_type,
                            lora_r=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            num_generated_tokens=args.num_generated_tokens,
                            num_train_examples=n_examples,
                            lr=lr_value,
                            run_idx=run_idx,
                            label_type=label_type if model_type in ['tok', 'a2t'] else None,
                            ce_loss_weight=ce_wt
                        )
                        adapter_path = f"../outputs/{model_type}/{trained_dataset}/{model_name}"
                        weights_map = {}
                        if os.path.isdir(adapter_path):
                            try:
                                model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.bfloat16)
                                model = model.to('cpu')
                                model.eval()
                                for name, module in model.named_modules():
                                    if 'lora_B' in name and hasattr(module, 'weight'):
                                        match = re.search(r'layers\.(\d+)\.self_attn\.([qkov]_proj)\.lora_B', name)
                                        if match:
                                            layer_idx = int(match.group(1))
                                            head_type = match.group(2)[0]
                                            if layer_idx not in weights_map:
                                                weights_map[layer_idx] = {}
                                            weights_map[layer_idx][head_type] = module.weight.detach().cpu()
                            except Exception as e:
                                print(f"Failed to load adapter or extract weights from {adapter_path}: {e}")
                            finally:
                                try:
                                    del model
                                except Exception:
                                    pass
                                torch.cuda.empty_cache()
                        model_weights_by_type[model_type] = weights_map if weights_map else None
                    # cleanup base model
                    del base_model
                    torch.cuda.empty_cache()

                    # Pass device string to subprocess; they will use it for GPU SVD
                    overlap_jobs.append((config.__dict__, n_examples, run_idx, model_weights_by_type, lora_r, args.reuse_cache, args.force_recompute, args.device))

            print(f"Generated {len(overlap_jobs)} overlap jobs")

            # Step 4: Process jobs in parallel
            results = []
            print(f"Starting parallel processing with {args.num_processes} processes...")
            if args.num_processes == 1:
                for job in tqdm(overlap_jobs, desc="Processing overlap jobs"):
                    res = process_overlap_job(job)
                    if res is not None:
                        results.append(res)
            else:
                with mp.Pool(processes=args.num_processes) as pool:
                    for res in tqdm(pool.imap_unordered(process_overlap_job, overlap_jobs), total=len(overlap_jobs), desc="Processing overlap jobs"):
                        if res is not None:
                            results.append(res)

            valid_results = [r for r in preload_results + results if r is not None]
            print(f"Valid results: {len(valid_results)}/{len(preload_results) + len(results)}")
            if not valid_results:
                print("No valid results found for this combo. Skipping.")
                continue

            # Step 5: Aggregate per combo
            aggregated_results = {}
            for result in valid_results:
                n_examples = result['num_examples']
                run_idx = result['run_idx']
                if n_examples not in aggregated_results:
                    aggregated_results[n_examples] = {
                        'pair_averages_by_run': {},
                        'layer_head_overlaps_by_run': {},
                        'runs': set()
                    }
                aggregated_results[n_examples]['runs'].add(run_idx)
                for pair_name, avg in result['pair_averages'].items():
                    if pair_name not in aggregated_results[n_examples]['pair_averages_by_run']:
                        aggregated_results[n_examples]['pair_averages_by_run'][pair_name] = {}
                    aggregated_results[n_examples]['pair_averages_by_run'][pair_name][run_idx] = avg
                for pair_name, layer_data in result['layer_head_overlaps'].items():
                    if pair_name not in aggregated_results[n_examples]['layer_head_overlaps_by_run']:
                        aggregated_results[n_examples]['layer_head_overlaps_by_run'][pair_name] = {}
                    if run_idx not in aggregated_results[n_examples]['layer_head_overlaps_by_run'][pair_name]:
                        aggregated_results[n_examples]['layer_head_overlaps_by_run'][pair_name][run_idx] = {}
                    for layer_idx, head_data in layer_data.items():
                        if layer_idx not in aggregated_results[n_examples]['layer_head_overlaps_by_run'][pair_name][run_idx]:
                            aggregated_results[n_examples]['layer_head_overlaps_by_run'][pair_name][run_idx][layer_idx] = {}
                        for head_type, overlaps in head_data.items():
                            aggregated_results[n_examples]['layer_head_overlaps_by_run'][pair_name][run_idx][layer_idx][head_type] = overlaps

            final_results = {}
            for n_examples, data in aggregated_results.items():
                final_results[n_examples] = {
                    'pair_averages': {},
                    'layer_head_overlaps': {},
                    'num_runs': len(data['runs'])
                }
                for pair_name, run_data in data['pair_averages_by_run'].items():
                    values = list(run_data.values())
                    if values:
                        final_results[n_examples]['pair_averages'][pair_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
                for pair_name, run_data in data['layer_head_overlaps_by_run'].items():
                    final_results[n_examples]['layer_head_overlaps'][pair_name] = {}
                    all_layer_indices = set()
                    for ridx, layer_map in run_data.items():
                        all_layer_indices.update(layer_map.keys())
                    for layer_idx in all_layer_indices:
                        final_results[n_examples]['layer_head_overlaps'][pair_name][layer_idx] = {}
                        all_head_types = set()
                        for ridx, layer_map in run_data.items():
                            if layer_idx in layer_map:
                                all_head_types.update(layer_map[layer_idx].keys())
                        for head_type in all_head_types:
                            all_overlaps = []
                            for ridx, layer_map in run_data.items():
                                if layer_idx in layer_map and head_type in layer_map[layer_idx]:
                                    all_overlaps.extend(layer_map[layer_idx][head_type])
                            if all_overlaps:
                                final_results[n_examples]['layer_head_overlaps'][pair_name][layer_idx][head_type] = all_overlaps

            print(f"Final results for {len(final_results)} N values")

            # Step 6: Plots per combo
            output_dir = os.path.join(args.output_dir, trained_dataset, label_type)
            plot_overlap_results(final_results, output_dir, eval_dataset, icl_source, label_type, args.hp_selection, args.model_id)

            # Step 7: Save results per combo (only subspace overlaps between methods)
            results_data = {
                'config': {
                    'model_id': args.model_id,
                    'dataset': trained_dataset,
                    'eval_dataset': eval_dataset,
                    'icl_source': icl_source,
                    'icl_demos': args.icl_demos,
                    'label_type': label_type,
                    'hp_selection': args.hp_selection,
                    'lora_type': args.lora_type,
                    'lora_r': args.lora_r,
                    'lora_alpha': args.lora_alpha,
                    'num_generated_tokens': args.num_generated_tokens,
                    'device': args.device,
                    'output_dir': args.output_dir
                },
                'overlap_results': {}
            }
            for n_examples, data in final_results.items():
                # Keep only pair_averages for compactness
                results_data['overlap_results'][n_examples] = data['pair_averages']
            results_path = os.path.join(output_dir, f"subspace_analysis_results_{trained_dataset}_{eval_dataset}_{icl_source}_{args.model_id.split('/')[-1]}.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            print(f"Results saved to: {results_path}")
            total_combos += 1

    if total_combos == 0:
        print("No analyses produced any results.")
    else:
        print(f"Completed analyses for {total_combos} dataset/label_type combinations.")


if __name__ == "__main__":
    main()
