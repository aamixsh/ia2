#!/usr/bin/env python3
"""
Simple script to run training and evaluation experiments serially.
Define experiments in the experiments list below and run them one by one.
"""

import subprocess
import sys
import os

# Define your experiments here as a list of dictionaries
# Each dictionary contains the script to run and its parameters
experiments = [
    # {
    #     'script': 'evaluate_batch_optimized.py',
    #     'params': [
    #         '--model_types', 'tok', 'act', 'tna', 'a2t',
    #         '--model_id', 'meta-llama/Llama-3.2-1B-Instruct',
    #         '--trained_datasets', 'gsm8k',
    #         '--eval_datasets', 'gsm8k', 'gsm8ks',
    #         '--icl_source_datasets', 'gsm8k',
    #         '--num_gen_tokens_eval', '200',
    #         '--num_tokens', '200',
    #         '--num_examples', '2', '4', '8', '16', '32', '64', '128',
    #         '--lrs', '1e-4', '3e-4', '5e-4', '1e-3', '3e-3', '5e-3',
    #         '--ce_loss_weights', '0.001', '0.01', '0.05', '0.5',
    #         '--batch_size', '500',
    #         '--gpus', '0',
    #     ]
    # },
    # {
    #     'script': 'train_all_unified.py',
    #     'params': [
    #         '--training_methods', 'tok', 'act', 'tna', 'a2t',
    #         '--datasets', 'sciqa',
    #         '--model_id', 'meta-llama/Llama-3.2-1B-Instruct',
    #         '--num_generated_tokens', '200',
    #         '--num_train_examples', '2', '4', '8', '16',
    #         '--lrs', '1e-4', '3e-4', '1e-3',
    #         '--ce_loss_weights', '0.001', '0.01', '0.05', '0.5',
    #         '--max_parallel', '5',
    #         '--gpus', '0'
    #     ]
    # },

    # {
    #     'script': 'train_all_unified.py',
    #     'params': [
    #         '--training_methods', 'tok', 'act', 'tna', 'a2t',
    #         '--datasets', 'sst2',
    #         '--model_id', 'meta-llama/Llama-3.2-1B',
    #         '--num_generated_tokens', '1',
    #         '--num_labelled_samples', '2',
    #         '--num_unlabelled_samples', '126',
    #         '--max_permutations', '1',
    #         '--num_train_examples', '2',
    #         '--lrs', '1e-4', '3e-4', '1e-3',
    #         '--ce_loss_weights', '0.001', '0.01', '0.05', '0.5',
    #         '--max_parallel', '8',
    #         '--gpus', '0',
    #         '--ldr_mode'
    #     ]
    # },
    # {
    #     'script': 'evaluate_batch_optimized.py',
    #     'params': [
    #         '--model_types', 'tok', 'act', 'tna', 'a2t',
    #         '--model_id', 'meta-llama/Llama-3.2-1B',
    #         '--trained_datasets', 'sst2',
    #         '--eval_datasets', 'sst2', 'finsen', 'poems',
    #         '--icl_source_datasets', 'sst2',
    #         '--num_gen_tokens_eval', '1',
    #         '--num_tokens', '1',
    #         '--num_examples', '2',
    #         '--num_labelled_samples', '2',
    #         '--num_unlabelled_samples', '126',
    #         '--max_permutations', '1',
    #         '--lrs', '1e-4', '3e-4', '5e-4', '1e-3', '3e-3', '5e-3',
    #         '--ce_loss_weights', '0.001', '0.01', '0.05', '0.5',
    #         '--batch_size', '500',
    #         '--gpus', '0',
    #         '--ldr_mode',
    #         '--uncertainty_analysis'
    #     ]
    # },
    {
        'script': 'activation_similarity_analysis.py',
        'params': [
            '--trained_dataset', 'sst2',
            '--eval_dataset', 'finsen',
            '--icl_source', 'sst2',
            '--label_type', 'ground_truth',
            '--model_id', 'meta-llama/Llama-3.2-1B',
            '--trained_datasets', 'sst2',
            '--num_generated_tokens', '1',
            '--num_tokens', '1',
            '--num_examples', '2',
            '--num_labelled_samples', '2',
            '--num_unlabelled_samples', '126',
            '--max_permutations', '1',
            '--lrs', '1e-4', '3e-4', '5e-4', '1e-3', '3e-3', '5e-3',
            '--ce_loss_weights', '0.001', '0.01', '0.05', '0.5',
            '--batch_size', '500',
            '--gpus', '0',
            '--ldr_mode',
            '--uncertainty_analysis'
        ]
    },
]

def run_experiment(script, params):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"Params: {' '.join(params)}")
    print(f"{'='*60}")
    
    cmd = ['python', script] + params
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {script} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script} failed with exit code {e.returncode}")
        return False

def main():
    """Run all experiments serially"""
    print(f"Starting {len(experiments)} experiments...")
    
    success_count = 0
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting experiment...")
        
        script = exp['script']
        params = exp['params']
        
        if run_experiment(script, params):
            success_count += 1
        
        print(f"Progress: {success_count}/{i} successful")
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Successful: {success_count}/{len(experiments)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
