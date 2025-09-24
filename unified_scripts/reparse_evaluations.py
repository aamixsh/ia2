#!/usr/bin/env python3
"""
Re-parse Evaluation Results Script

This script takes existing evaluation results, applies a new parsing function to the 
generated text, and recalculates metrics. It preserves the original directory structure
but saves to a new output directory.

Usage:
    python reparse_evaluations.py --model_types base --eval_datasets gsm8k --output_dir ../outputs_reparsed --parse_function custom_parse
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from collections import defaultdict
import importlib.util
import inspect

# Add the current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import parse_answer_gsm8k, parse_answer_sciqa, parse_answer_boxed, construct_results_path, construct_base_without_icl_path


def calculate_metrics(results, uncertainty_analysis=False, label_tokens=None, num_generated_tokens=1, top_k=10, parse_answer_func=None):
    """Calculate evaluation metrics from results."""
    metrics = {}
    
    for eval_type in ['without_icl', 'with_icl']:
        if uncertainty_analysis:
            # Uncertainty analysis metrics
            correct_top1 = 0
            correct_label_set = 0
            valid_predictions = 0
            total = len(results[eval_type])
            
            # For single token generation
            if num_generated_tokens == 1:
                top1_entropies = []
                top1_uncertainties = []
                label_entropies = []
                label_uncertainties = []
                top1_confidences = []
                label_confidences = []
                top1_correct = []
                label_correct = []
                
                for pred in results[eval_type]:
                    if pred['prob_info'] is not None and pred['prob_info']['top_k_tokens']:
                        valid_predictions += 1
                        true_label = str(pred['parsed_answer'])
                        
                        # Top-1 accuracy
                        predicted_top1 = pred['prob_info']['top_k_tokens'][0].strip()
                        if predicted_top1 == true_label:
                            correct_top1 += 1
                            top1_correct.append(1)
                        else:
                            top1_correct.append(0)
                        
                        # Label set accuracy (if available)
                        if label_tokens and pred['prob_info']['label_probs']:
                            label_tokens_clean = [token.strip() for token in label_tokens]
                            if true_label in label_tokens_clean and np.all(np.array([pred['prob_info']['label_probs'][true_label] >= pred['prob_info']['label_probs'][label] for label in label_tokens_clean if label != true_label])):
                                correct_label_set += 1
                                label_correct.append(1)
                            else:
                                label_correct.append(0)
                        
                        # Calculate entropy and uncertainty for top-1
                        top_k_probs = pred['prob_info']['top_k_probs']
                        if len(top_k_probs) > 0:
                            probs_array = np.array(top_k_probs)
                            # Normalize probabilities (in case they don't sum to 1 due to top-k selection)
                            probs_array = probs_array / probs_array.sum()
                            
                            # Entropy: H = -sum(p * log(p))
                            entropy = -np.sum(probs_array * np.log(probs_array + 1e-10))
                            top1_entropies.append(entropy)
                            
                            # Uncertainty: 1 - top1 probability
                            top1_prob = probs_array[0]
                            uncertainty = 1 - top1_prob
                            top1_uncertainties.append(uncertainty)
                            top1_confidences.append(top1_prob)
                        
                        # Calculate entropy and uncertainty for label set
                        if label_tokens and pred['prob_info']['label_probs']:
                            label_probs = list(pred['prob_info']['label_probs'].values())
                            if len(label_probs) > 0:
                                label_probs_array = np.array(label_probs)
                                # Normalize probabilities using softmax
                                label_probs_array = np.exp(label_probs_array) / np.sum(np.exp(label_probs_array))
                                
                                # Entropy
                                entropy = -np.sum(label_probs_array * np.log(label_probs_array + 1e-10))
                                label_entropies.append(entropy)
                                
                                # Uncertainty: 1 - top label probability
                                top_label_prob = np.max(label_probs_array)
                                uncertainty = 1 - top_label_prob
                                label_uncertainties.append(uncertainty)
                                label_confidences.append(top_label_prob)
                
                if valid_predictions > 0:
                    metrics[f'{eval_type}_accuracy_top1'] = (correct_top1 / valid_predictions) * 100
                    
                    if label_tokens:
                        metrics[f'{eval_type}_accuracy_label_set'] = (correct_label_set / valid_predictions) * 100
                    
                    if top1_entropies:
                        metrics[f'{eval_type}_entropy_top1_mean'] = np.mean(top1_entropies)
                        metrics[f'{eval_type}_entropy_top1_std'] = np.std(top1_entropies)
                        metrics[f'{eval_type}_uncertainty_top1_mean'] = np.mean(top1_uncertainties)
                        metrics[f'{eval_type}_uncertainty_top1_std'] = np.std(top1_uncertainties)
                    
                    if label_entropies:
                        metrics[f'{eval_type}_entropy_label_mean'] = np.mean(label_entropies)
                        metrics[f'{eval_type}_entropy_label_std'] = np.std(label_entropies)
                        metrics[f'{eval_type}_uncertainty_label_mean'] = np.mean(label_uncertainties)
                        metrics[f'{eval_type}_uncertainty_label_std'] = np.std(label_uncertainties)
                    
                    # Calculate Expected Calibration Error for top1
                    if top1_confidences and top1_correct:
                        ece_top1 = calculate_ece(top1_confidences, top1_correct)
                        metrics[f'{eval_type}_ece_top1'] = ece_top1
                    
                    # Calculate Expected Calibration Error for label set
                    if label_confidences and label_correct:
                        ece_label = calculate_ece(label_confidences, label_correct)
                        metrics[f'{eval_type}_ece_label'] = ece_label
            
            # For multi-token generation
            else:
                # Collect entropy across all token positions for top-k
                all_entropies_topk = []
                # Standard accuracy metrics
                correct = 0
                valid_predictions = 0

                total = len(results[eval_type])

                for pred in results[eval_type]:
                    if pred['prob_info'] is not None and 'prob_matrix' in pred['prob_info'] and pred['generated_answer'] is not None:
                        valid_predictions += 1
                        true_answer = pred.get('parsed_answer', pred.get('answer', ''))
                        if str(true_answer) == str(pred['generated_answer']).strip():
                            correct += 1
                        
                        # Calculate metrics for each token position
                        for token_idx in range(len(pred['prob_info']['prob_matrix'])):
                            prob_matrix = pred['prob_info']['prob_matrix'][token_idx]
                            if len(prob_matrix) > 0:
                                # Top-k metrics
                                topk_probs = prob_matrix[:top_k]
                                topk_probs_array = np.array(topk_probs)
                                if topk_probs_array.sum() > 0:
                                    topk_probs_array = topk_probs_array / topk_probs_array.sum()
                                    entropy = -np.sum(topk_probs_array * np.log(topk_probs_array + 1e-10))
                                    all_entropies_topk.append(entropy)
                
                if valid_predictions > 0:
                    # Calculate average metrics across all token positions
                    if all_entropies_topk:
                        metrics[f'{eval_type}_entropy_topk_mean'] = np.mean(all_entropies_topk)
                        metrics[f'{eval_type}_entropy_topk_std'] = np.std(all_entropies_topk)
                        # Store k in meta info
                        metrics[f'{eval_type}_topk_value'] = top_k

                    accuracy = (correct / valid_predictions) * 100 if valid_predictions > 0 else 0
                    metrics[f'{eval_type}_accuracy'] = accuracy
            
            metrics[f'{eval_type}_valid_predictions'] = valid_predictions
            metrics[f'{eval_type}_total_examples'] = total
            
        else:
            # Standard accuracy metrics
            correct = 0
            valid_predictions = 0
            total = len(results[eval_type])
            for pred in results[eval_type]:
                if pred['generated_answer'] is not None:
                    valid_predictions += 1
                    true_answer = pred.get('parsed_answer', pred.get('answer', ''))
                    if str(true_answer) == str(pred['generated_answer']).strip():
                        correct += 1
            accuracy = (correct / total) * 100 if total > 0 else 0
            metrics[f'{eval_type}_accuracy'] = accuracy
            metrics[f'{eval_type}_valid_predictions'] = valid_predictions
            metrics[f'{eval_type}_total_examples'] = total

    # Calculate deltas and agreement
    if uncertainty_analysis:
        if 'with_icl_accuracy_top1' in metrics and 'without_icl_accuracy_top1' in metrics:
            metrics['accuracy_delta_top1'] = metrics['with_icl_accuracy_top1'] - metrics['without_icl_accuracy_top1']
            
            if 'with_icl_accuracy_label_set' in metrics:
                metrics['accuracy_delta_label_set'] = metrics['with_icl_accuracy_label_set'] - metrics['without_icl_accuracy_label_set']
        
        # Calculate entropy and uncertainty deltas
        if 'with_icl_entropy_top1_mean' in metrics and 'without_icl_entropy_top1_mean' in metrics:
            metrics['entropy_delta_top1'] = metrics['with_icl_entropy_top1_mean'] - metrics['without_icl_entropy_top1_mean']
            metrics['uncertainty_delta_top1'] = metrics['with_icl_uncertainty_top1_mean'] - metrics['without_icl_uncertainty_top1_mean']
        
        if 'with_icl_entropy_label_mean' in metrics and 'without_icl_entropy_label_mean' in metrics:
            metrics['entropy_delta_label'] = metrics['with_icl_entropy_label_mean'] - metrics['without_icl_entropy_label_mean']
            metrics['uncertainty_delta_label'] = metrics['with_icl_uncertainty_label_mean'] - metrics['without_icl_uncertainty_label_mean']
        
        # Calculate ECE deltas
        if 'with_icl_ece_top1' in metrics and 'without_icl_ece_top1' in metrics:
            metrics['ece_delta_top1'] = metrics['with_icl_ece_top1'] - metrics['without_icl_ece_top1']
        
        if 'with_icl_ece_label' in metrics and 'without_icl_ece_label' in metrics:
            metrics['ece_delta_label'] = metrics['with_icl_ece_label'] - metrics['without_icl_ece_label']
        
        # For multi-token, calculate entropy delta
        if 'with_icl_entropy_topk_mean' in metrics and 'without_icl_entropy_topk_mean' in metrics:
            metrics['entropy_delta_topk'] = metrics['with_icl_entropy_topk_mean'] - metrics['without_icl_entropy_topk_mean']
    else:
        if 'with_icl_accuracy' in metrics and 'without_icl_accuracy' in metrics:
            metrics['accuracy_delta'] = metrics['with_icl_accuracy'] - metrics['without_icl_accuracy']

    # Calculate agreement rates
    agreement_count = 0
    both_correct = 0
    both_incorrect = 0  
    only_icl_correct = 0
    only_without_icl_correct = 0
    comparable_examples = 0
    
    for with_icl_pred, without_icl_pred in zip(results['with_icl'], results['without_icl']):
        if uncertainty_analysis:
            # Handle single token vs multi-token scenarios
            if num_generated_tokens == 1:
                if (with_icl_pred['prob_info'] is not None and without_icl_pred['prob_info'] is not None and
                    with_icl_pred['prob_info']['top_k_tokens'] and without_icl_pred['prob_info']['top_k_tokens']):
                    comparable_examples += 1
                    true_label = str(with_icl_pred['parsed_answer'])
                    with_icl_output = with_icl_pred['prob_info']['top_k_tokens'][0].strip()
                    without_icl_output = without_icl_pred['prob_info']['top_k_tokens'][0].strip()
                    if with_icl_output == without_icl_output: 
                        agreement_count += 1
                    with_icl_correct = (with_icl_output == true_label)
                    without_icl_correct = (without_icl_output == true_label)
                    if with_icl_correct and without_icl_correct: 
                        both_correct += 1
                    elif not with_icl_correct and not without_icl_correct: 
                        both_incorrect += 1
                    elif with_icl_correct and not without_icl_correct: 
                        only_icl_correct += 1
                    elif not with_icl_correct and without_icl_correct: 
                        only_without_icl_correct += 1
            else:
                # For multi-token, compare the generated answer
                if (with_icl_pred['prob_info'] is not None and without_icl_pred['prob_info'] is not None and
                    'generated_text' in with_icl_pred['prob_info'] and 'generated_text' in without_icl_pred['prob_info']):
                    comparable_examples += 1
                    true_label = str(with_icl_pred['parsed_answer'])
                    with_icl_output = parse_answer_func(with_icl_pred['prob_info']['generated_text']).strip()
                    without_icl_output = parse_answer_func(without_icl_pred['prob_info']['generated_text']).strip()
                    if with_icl_output == without_icl_output: 
                        agreement_count += 1
                    with_icl_correct = (with_icl_output == true_label)
                    without_icl_correct = (without_icl_output == true_label)
                    if with_icl_correct and without_icl_correct: 
                        both_correct += 1
                    elif not with_icl_correct and not without_icl_correct: 
                        both_incorrect += 1
                    elif with_icl_correct and not without_icl_correct: 
                        only_icl_correct += 1
                    elif not with_icl_correct and without_icl_correct: 
                        only_without_icl_correct += 1
        else:
            if with_icl_pred['generated_answer'] is not None and without_icl_pred['generated_answer'] is not None:
                comparable_examples += 1
                true_label = str(with_icl_pred.get('parsed_answer', with_icl_pred.get('answer', '')))
                with_icl_output = str(with_icl_pred['generated_answer']).strip()
                without_icl_output = str(without_icl_pred['generated_answer']).strip()
                if with_icl_output == without_icl_output: 
                    agreement_count += 1
                with_icl_correct = (with_icl_output == true_label)
                without_icl_correct = (without_icl_output == true_label)
                if with_icl_correct and without_icl_correct: 
                    both_correct += 1
                elif not with_icl_correct and not without_icl_correct: 
                    both_incorrect += 1
                elif with_icl_correct and not without_icl_correct: 
                    only_icl_correct += 1
                elif not with_icl_correct and without_icl_correct: 
                    only_without_icl_correct += 1
    
    if comparable_examples > 0:
        metrics['agreement_rate'] = (agreement_count / comparable_examples) * 100
        metrics['both_correct'] = both_correct
        metrics['both_incorrect'] = both_incorrect
        metrics['only_icl_correct'] = only_icl_correct
        metrics['only_without_icl_correct'] = only_without_icl_correct
        metrics['comparable_examples'] = comparable_examples
        metrics['both_correct_pct'] = (both_correct / comparable_examples) * 100
        metrics['both_incorrect_pct'] = (both_incorrect / comparable_examples) * 100
        metrics['only_icl_correct_pct'] = (only_icl_correct / comparable_examples) * 100
        metrics['only_without_icl_correct_pct'] = (only_without_icl_correct / comparable_examples) * 100

    return metrics


def calculate_ece(confidences, corrects, n_bins=15):
    """Calculate Expected Calibration Error using 1 vs rest method."""
    confidences = np.array(confidences)
    corrects = np.array(corrects)
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    confidences = confidences[sorted_indices]
    corrects = corrects[sorted_indices]
    
    # Create bins
    bin_size = len(confidences) // n_bins
    ece = 0.0
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(confidences)
        
        if start_idx < end_idx:
            bin_confidences = confidences[start_idx:end_idx]
            bin_corrects = corrects[start_idx:end_idx]
            
            # Calculate bin accuracy and confidence
            bin_accuracy = np.mean(bin_corrects)
            bin_confidence = np.mean(bin_confidences)
            
            # Calculate bin weight
            bin_weight = len(bin_confidences) / len(confidences)
            
            # Add to ECE
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece


def find_evaluation_files(args):
    """Find all evaluation files that match the specified criteria."""
    files_found = []
    
    # Base directory for evaluations
    base_dir = "../outputs/evaluations"
    
    for model_type in args.model_types:
        for eval_dataset in args.eval_datasets:
            # Construct search path
            if model_type == 'base':
                # Base model evaluations
                search_dir = os.path.join(base_dir, "base", eval_dataset)
                if args.uncertainty_analysis:
                    search_dir = os.path.join(base_dir, "base_uncertainty", eval_dataset)
            else:
                # Trained model evaluations
                for trained_dataset in args.trained_datasets:
                    search_dir = os.path.join(base_dir, model_type, trained_dataset)
                    if args.uncertainty_analysis:
                        search_dir = os.path.join(base_dir, f"{model_type}_uncertainty", trained_dataset)
            
            if not os.path.exists(search_dir):
                continue
            
            # Find all JSON files in the directory
            for filename in os.listdir(search_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(search_dir, filename)
                    
                    # Load the file to check if it matches our criteria
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                        if data.get('evaluation_config', {}).get('eval_dataset_name') != eval_dataset:
                            continue

                        if not args.rerun and os.path.exists(os.path.join(args.output_dir, os.path.relpath(file_path, "../outputs"))):
                            continue

                        files_found.append(file_path)
    
    return files_found


def reparse_evaluation_file(file_path, parse_answer_func, output_dir):
    """Re-parse a single evaluation file with the new parsing function."""
    print(f"Processing: {file_path}")
    
    try:
        # Load existing results
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract necessary information
        results = data['results']
        evaluation_config = data.get('evaluation_config', {})
        
        # Get parameters for metrics calculation
        uncertainty_analysis = evaluation_config.get('uncertainty_analysis', False)
        num_generated_tokens = evaluation_config.get('num_generated_tokens_eval', 1)
        top_k = evaluation_config.get('top_k', 10)
        eval_dataset_name = evaluation_config.get('eval_dataset_name', '')
        
        # Load label information if needed
        label_tokens = None
        if uncertainty_analysis and num_generated_tokens == 1:
            label_tokens = evaluation_config.get('label_tokens')
        
        # Re-parse generated text for both with_icl and without_icl
        for eval_type in ['with_icl', 'without_icl']:
            if eval_type in results:
                for pred in results[eval_type]:
                    if 'full_generated_text' in pred:
                        # Apply new parsing function
                        new_generated_answer = parse_answer_func(pred['full_generated_text'])
                        pred['generated_answer'] = new_generated_answer
        
        # Recalculate metrics
        new_metrics = calculate_metrics(
            results, 
            uncertainty_analysis=uncertainty_analysis,
            label_tokens=label_tokens,
            num_generated_tokens=num_generated_tokens,
            top_k=top_k,
            parse_answer_func=parse_answer_func
        )
        
        # Update the data with new metrics
        data['metrics'] = new_metrics
        
        # Add information about the re-parsing
        data['reparsing_info'] = {
            'original_file': file_path,
            'parse_function_used': parse_answer_func.__name__,
            'reparsing_timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        # Create output path preserving directory structure
        rel_path = os.path.relpath(file_path, "../outputs")
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save updated results
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Re-parsed and saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Re-parse evaluation results with a new parsing function")
    
    # Model selection (same as evaluation script)
    parser.add_argument("--model_types", nargs='+', 
                        choices=['tok', 'act', 'a2t', 'tna', 't2a', 'base', 
                                'ia3-tok', 'ia3-act', 'ia3-tna', 'ia3-a2t', 'ia3-t2a',
                                'prompt-tok', 'prompt-act', 'prompt-tna', 'prompt-a2t', 'prompt-t2a',
                                'prefix-tok', 'prefix-act', 'prefix-tna', 'prefix-a2t', 'prefix-t2a'],
                        # default=['base'],
                        default=['base', 'tok', 'act', 'tna', 'a2t', 't2a'],
                        help="Types of models to re-parse")
    parser.add_argument("--trained_datasets", nargs='+', default=['hmath_algebra'],
                        help="Datasets the models were trained on")
    
    # Evaluation settings (same as evaluation script)
    parser.add_argument("--eval_datasets", nargs='+', default=['hmath_algebra'],
                        help="Datasets to re-parse")
    
    # Analysis options (same as evaluation script)
    parser.add_argument("--uncertainty_analysis", action='store_true',
                        help="Re-parse uncertainty analysis results")

    parser.add_argument("--rerun", action='store_true',
                        help="Rerun files even if they already exist")
    
    # Re-parsing specific arguments
    parser.add_argument("--output_dir", type=str, default="../outputs_reparsed",
                        help="Output directory for re-parsed results")
    parser.add_argument("--dry_run", action='store_true',
                        help="Show what files would be processed without actually processing them")
    
    args = parser.parse_args()

    parse_answer_func = None
    if args.trained_datasets[0] == 'hmath_algebra':
        parse_answer_func = parse_answer_boxed
    elif args.trained_datasets[0] == 'sciqa' or args.trained_datasets[0] == 'strategyreason':
        parse_answer_func = parse_answer_sciqa
    elif args.trained_datasets[0] == 'gsm8k':
        parse_answer_func = parse_answer_gsm8k
    else:
        print(f"No parsing function found for {args.trained_datasets[0]}")
        return

    # Find evaluation files
    print("Finding evaluation files...")
    files_to_process = find_evaluation_files(args)
    
    if not files_to_process:
        print("No evaluation files found matching the criteria!")
        return
    
    print(f"Found {len(files_to_process)} files to process")
    
    if args.dry_run:
        print("\nFiles that would be processed:")
        for file_path in files_to_process:
            print(f"  {file_path}")
        return
    
    # Process files
    print(f"\nProcessing {len(files_to_process)} files...")
    successful = 0
    failed = 0
    
    for file_path in tqdm(files_to_process, desc="Re-parsing files"):
        if reparse_evaluation_file(file_path, parse_answer_func, args.output_dir):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RE-PARSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(files_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(files_to_process)*100:.1f}%")
    print(f"Output directory: {args.output_dir}")
    print(f"Parse function: {parse_answer_func.__name__}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
