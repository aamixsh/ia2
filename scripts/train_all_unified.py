#!/usr/bin/env python3
"""
Unified Batch Training Script for Distillation Project

This script runs training across multiple configurations using the unified training script.
Supports all training methods: tok, act, tna, tokl, a2t, t2a, prompt, prefix, and ldr.
"""

import os
import sys
import json
import argparse
import subprocess
import time
from multiprocessing import Pool
from itertools import product
from utils import get_model_name

def run_single_training(args_tuple):
    """Run a single training experiment using the unified training script"""
    # Handle different tuple lengths based on training method
    if '-' in args_tuple[0]:
        base_method, training_variant = args_tuple[0].split('-')
    else:
        base_method = 'lora'
        training_variant = args_tuple[0]

    if len(args_tuple) == 28:  # LoRA + IA3 method (+ tokl_top_k)
        (training_method, dataset, model_id, lora_type, lora_r, lora_alpha,
         num_generated_tokens, num_train_examples, lr, label_type, ce_loss_weight,
         gpu_id, num_runs, wandb_log, wandb_project, wandb_entity, num_train_epochs, batch_size,
         gradient_clip_val, patience, dev_eval_steps, log_gradient_norms,
         log_checkpoints, checkpoint_frequency, shuffle_demos, num_shuffles, output_dir, tokl_top_k) = args_tuple
        if base_method == 'ia3':
            ia3_type = lora_type
            lora_type = None
        else:
            ia3_type = None
        num_virtual_tokens = None
        # LDR parameters (not provided in this tuple format)
        ldr_mode = False
        num_labelled_samples = None
        num_unlabelled_samples = None
        max_permutations = None
    elif len(args_tuple) == 29:  # Prompt/Prefix methods (+ tokl_top_k)
        (training_method, dataset, model_id, lora_type, lora_r, lora_alpha,
         num_generated_tokens, num_train_examples, lr, label_type, ce_loss_weight,
         gpu_id, num_runs, wandb_log, wandb_project, wandb_entity, num_train_epochs, batch_size,
         gradient_clip_val, patience, dev_eval_steps, log_gradient_norms,
         log_checkpoints, checkpoint_frequency, shuffle_demos, num_shuffles, num_virtual_tokens, output_dir, tokl_top_k) = args_tuple
        ia3_type = None
        # LDR parameters (not provided in this tuple format)
        ldr_mode = False
        num_labelled_samples = None
        num_unlabelled_samples = None
        max_permutations = None
    elif len(args_tuple) == 32:  # LDR mode Lora + IA3 (+ tokl_top_k)
        (training_method, dataset, model_id, lora_type, lora_r, lora_alpha,
         num_generated_tokens, num_train_examples, lr, label_type, ce_loss_weight,
         gpu_id, num_runs, wandb_log, wandb_project, wandb_entity, num_train_epochs, batch_size,
         gradient_clip_val, patience, dev_eval_steps, log_gradient_norms,
         log_checkpoints, checkpoint_frequency, shuffle_demos, num_shuffles, output_dir,
         ldr_mode, num_labelled_samples, num_unlabelled_samples, max_permutations, tokl_top_k) = args_tuple
        if base_method == 'ia3':
            ia3_type = lora_type
            lora_type = None
        else:
            ia3_type = None
        num_virtual_tokens = None
    elif len(args_tuple) == 33:  # LDR mode Prompt/Prefix (+ tokl_top_k)
        (training_method, dataset, model_id, lora_type, lora_r, lora_alpha,
         num_generated_tokens, num_train_examples, lr, label_type, ce_loss_weight,
         gpu_id, num_runs, wandb_log, wandb_project, wandb_entity, num_train_epochs, batch_size,
         gradient_clip_val, patience, dev_eval_steps, log_gradient_norms,
         log_checkpoints, checkpoint_frequency, shuffle_demos, num_shuffles, num_virtual_tokens, output_dir,
         ldr_mode, num_labelled_samples, num_unlabelled_samples, max_permutations, tokl_top_k) = args_tuple
        ia3_type = None
    else:
        raise ValueError(f"Unexpected tuple length: {len(args_tuple)}")

    script = 'train_unified.py'
    cmd = [
        'python', script,
        '--training_method', str(training_method),
        '--dataset', str(dataset),
        '--model_id', str(model_id),
        '--num_generated_tokens', str(num_generated_tokens),
        '--num_train_examples', str(num_train_examples),
        '--lr', str(lr),
        '--num_runs', str(num_runs),
        '--gpu', str(gpu_id),
        '--output_dir', str(output_dir)
    ]

    # Add method-specific arguments
    if training_method in ['tok', 'act', 'tna', 'a2t', 't2a']:
        cmd.extend(['--lora_type', str(lora_type)])
        cmd.extend(['--lora_r', str(lora_r)])
        cmd.extend(['--lora_alpha', str(lora_alpha)])
    elif base_method == 'ia3':
        cmd.extend(['--ia3_type', str(ia3_type)])
    elif base_method in ['prompt', 'prefix']:
        cmd.extend(['--num_virtual_tokens', str(num_virtual_tokens)])
    
    if training_variant in ['tok', 'a2t', 'tokl'] and label_type:
        cmd.extend(['--label_type', str(label_type)])
    if training_variant == 'tokl':
        cmd.extend(['--tokl_top_k', str(tokl_top_k)])
    
    if training_variant == 'tna' and ce_loss_weight is not None:
        cmd.extend(['--ce_loss_weight', str(ce_loss_weight)])

    # Add training configuration
    if num_train_epochs:
        cmd.extend(['--num_train_epochs', str(num_train_epochs)])
    if batch_size:
        cmd.extend(['--batch_size', str(batch_size)])
        if script == 'train_batch_optimized.py':
            # Add data collection batch size (can be larger than training batch size)
            cmd.extend(['--data_collection_batch_size', str(min(batch_size * 4, 16))])
    if gradient_clip_val:
        cmd.extend(['--gradient_clip_val', str(gradient_clip_val)])
    if patience:
        cmd.extend(['--patience', str(patience)])
    if dev_eval_steps:
        cmd.extend(['--dev_eval_steps', str(dev_eval_steps)])

    # Add data options
    if shuffle_demos:
        cmd.append('--shuffle_demos')
        if num_shuffles:
            cmd.extend(['--num_shuffles', str(num_shuffles)])

    # Add LDR options
    if ldr_mode:
        cmd.append('--ldr_mode')
        if num_labelled_samples:
            cmd.extend(['--num_labelled_samples', str(num_labelled_samples)])
        if num_unlabelled_samples:
            cmd.extend(['--num_unlabelled_samples', str(num_unlabelled_samples)])
        if max_permutations:
            cmd.extend(['--max_permutations', str(max_permutations)])

    # Add logging options
    if log_gradient_norms:
        cmd.append('--log_gradient_norms')
    if log_checkpoints:
        cmd.extend(['--log_checkpoints', '--checkpoint_frequency', str(checkpoint_frequency)])

    # Add WandB logging
    if wandb_log:
        cmd.append('--wandb_log')
        if wandb_project:
            cmd.extend(['--wandb_project', str(wandb_project)])
        if wandb_entity:
            cmd.extend(['--wandb_entity', str(wandb_entity)])

    # Run the training
    try:
        exp_name = f"{training_method}_{dataset}_N{num_train_examples}_lr{lr}_run{num_runs}"
        if label_type:
            exp_name += f"_{label_type}"
        if ce_loss_weight is not None:
            exp_name += f"_cew{ce_loss_weight}"
        
        print(f"Running: {exp_name}")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"✓ Training completed successfully for {exp_name}")
            return True
        else:
            print(f"✗ Training failed for {exp_name}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Training timed out for {exp_name}")
        return False
    except Exception as e:
        print(f"✗ Error running training for {exp_name}: {e}")
        return False


def models_exist(model_id, training_method, dataset, lora_type, lora_r, lora_alpha, num_tokens, num_examples, lr, label_type, ce_weight, num_runs, num_virtual_tokens=None, output_dir=None, ldr_mode=False, num_labelled_samples=None, num_unlabelled_samples=None, max_permutations=None):
    """Check if the model already exists"""
    exists = True

    if '-' in training_method:
        base_method, training_variant = training_method.split('-', 1)
    else:
        base_method = 'lora'
        training_variant = training_method

    for run_idx in range(num_runs):
        if base_method == 'lora':
            model_name = get_model_name(training_method, model_id.split('/')[-1], lora_type, lora_r, lora_alpha, num_tokens, num_examples, lr, run_idx, label_type, ce_weight, ldr_mode=ldr_mode, num_labelled_samples=num_labelled_samples, num_unlabelled_samples=num_unlabelled_samples, max_permutations=max_permutations)
        elif base_method == 'ia3':
            model_name = get_model_name(training_method, model_id.split('/')[-1], None, None, None, num_tokens, num_examples, lr, run_idx, label_type, ce_weight, ia3_type=lora_type, ldr_mode=ldr_mode, num_labelled_samples=num_labelled_samples, num_unlabelled_samples=num_unlabelled_samples, max_permutations=max_permutations)  # lora_type is ia3_type here
        elif base_method in ['prompt', 'prefix']:
            model_name = get_model_name(training_method, model_id.split('/')[-1], None, None, None, num_tokens, num_examples, lr, run_idx, label_type, ce_weight, ia3_type=None, num_virtual_tokens=num_virtual_tokens, ldr_mode=ldr_mode, num_labelled_samples=num_labelled_samples, num_unlabelled_samples=num_unlabelled_samples, max_permutations=max_permutations)
        else:
            print(f"Unknown training method: {training_method}")
            return False

        if not os.path.exists(f"{output_dir}/{training_method}/{dataset}/{model_name}/adapter_model.safetensors"):
            exists = False
            break
    return exists


def generate_experiments(args):
    """Generate all experiment configurations"""
    all_experiments = []
    gpu_cycle = 0
    
    # Generate all combinations
    for training_method in args.training_methods:
        # Parse training method to extract base method and training variant
        if '-' in training_method:
            base_method, training_variant = training_method.split('-', 1)
        else:
            base_method = 'lora'
            training_variant = training_method
        
        for dataset in args.datasets:
            for num_tokens in args.num_generated_tokens:
                for num_examples in args.num_train_examples:
                    for lr in args.lrs:
                        # Handle different parameter sets based on training method
                        if base_method in ['lora'] or training_method in ['tok', 'tokl', 'act', 'tna', 'a2t', 't2a']:
                            # LoRA methods (keep unchanged)
                            for lora_type in args.lora_types:
                                for lora_r in args.lora_rs:
                                    for lora_alpha in args.lora_alphas:
                                        # Determine relevant label_types and ce_loss_weights for this method
                                        if training_variant in ['tok', 'a2t']:
                                            label_types_to_use = args.label_types
                                            ce_weights_to_use = [None]
                                        elif training_variant in ['tokl']:
                                            label_types_to_use = ['icl_outputs']
                                            ce_weights_to_use = [None]
                                        elif training_variant == 'tna':
                                            label_types_to_use = [None]
                                            ce_weights_to_use = args.ce_loss_weights
                                        elif training_variant == 'act':
                                            label_types_to_use = [None]
                                            ce_weights_to_use = [None]
                                        
                                        # Generate experiments for this configuration
                                        for label_type in label_types_to_use:
                                            for ce_weight in ce_weights_to_use:
                                                # Add LDR parameters if in LDR mode
                                                if args.ldr_mode:
                                                    for num_unlabelled in args.num_unlabelled_samples:
                                                        for max_perm in args.max_permutations:

                                                            if models_exist(args.model_id, training_method, dataset, lora_type, lora_r, lora_alpha, num_tokens, num_examples, lr, label_type, ce_weight, args.num_runs, output_dir=args.output_dir, ldr_mode=True, num_labelled_samples=num_examples, num_unlabelled_samples=num_unlabelled, max_permutations=max_perm):
                                                                continue
                                                            
                                                            gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
                                                            gpu_cycle += 1

                                                            if training_variant == 'tok' and label_type == 'ground_truth': # skip ground truth for tok models when in LDR mode
                                                                continue
                                                            experiment = (
                                                                training_method,         # training_method
                                                                dataset,                 # dataset
                                                                args.model_id,          # model_id
                                                                lora_type,              # lora_type
                                                                lora_r,                 # lora_r
                                                                lora_alpha,             # lora_alpha
                                                                num_tokens,             # num_generated_tokens
                                                                num_examples,           # num_train_examples
                                                                lr,                     # lr
                                                                label_type,             # label_type
                                                                ce_weight,              # ce_loss_weight
                                                                gpu_id,                 # gpu_id
                                                                args.num_runs,          # run_idx
                                                                args.wandb_log,         # wandb_log
                                                                args.wandb_project,     # wandb_project
                                                                args.wandb_entity,      # wandb_entity
                                                                args.num_train_epochs,  # num_train_epochs
                                                                args.batch_size,        # batch_size
                                                                args.gradient_clip_val, # gradient_clip_val
                                                                args.patience,          # patience
                                                                args.dev_eval_steps,    # dev_eval_steps
                                                                args.log_gradient_norms,# log_gradient_norms
                                                                args.log_checkpoints,   # log_checkpoints
                                                                args.checkpoint_frequency, # checkpoint_frequency
                                                                args.shuffle_demos,     # shuffle_demos
                                                                args.num_shuffles,       # num_shuffles
                                                                args.output_dir,         # output_dir
                                                                args.tokl_top_k,         # tokl_top_k
                                                                True,                   # ldr_mode
                                                                num_examples,           # num_labelled_samples == num_train_examples
                                                                num_unlabelled,         # num_unlabelled_samples
                                                                max_perm                # max_permutations
                                                            )
                                                            all_experiments.append(experiment)
                                                else:
                                                    if models_exist(args.model_id, training_method, dataset, lora_type, lora_r, lora_alpha, num_tokens, num_examples, lr, label_type, ce_weight, args.num_runs, output_dir=args.output_dir, ldr_mode=False, num_labelled_samples=num_examples, num_unlabelled_samples=None, max_permutations=None):
                                                        continue
                                                    
                                                    gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
                                                    gpu_cycle += 1
                                                    
                                                    experiment = (
                                                        training_method,         # training_method
                                                        dataset,                 # dataset
                                                        args.model_id,          # model_id
                                                        lora_type,              # lora_type
                                                        lora_r,                 # lora_r
                                                        lora_alpha,             # lora_alpha
                                                        num_tokens,             # num_generated_tokens
                                                        num_examples,           # num_train_examples
                                                        lr,                     # lr
                                                        label_type,             # label_type
                                                        ce_weight,              # ce_loss_weight
                                                        gpu_id,                 # gpu_id
                                                        args.num_runs,          # run_idx
                                                        args.wandb_log,         # wandb_log
                                                        args.wandb_project,     # wandb_project
                                                        args.wandb_entity,      # wandb_entity
                                                        args.num_train_epochs,  # num_train_epochs
                                                        args.batch_size,        # batch_size
                                                        args.gradient_clip_val, # gradient_clip_val
                                                        args.patience,          # patience
                                                        args.dev_eval_steps,    # dev_eval_steps
                                                        args.log_gradient_norms,# log_gradient_norms
                                                        args.log_checkpoints,   # log_checkpoints
                                                        args.checkpoint_frequency, # checkpoint_frequency
                                                        args.shuffle_demos,     # shuffle_demos
                                                        args.num_shuffles,       # num_shuffles
                                                        args.output_dir,         # output_dir
                                                        args.tokl_top_k          # tokl_top_k
                                                    )
                                                    all_experiments.append(experiment)
                        elif base_method == 'ia3':
                            # IA3 method variants
                            for ia3_type in args.ia3_types:
                                # Determine relevant label_types and ce_loss_weights for this method
                                if training_variant in ['tok', 'a2t']:
                                    label_types_to_use = args.label_types
                                    ce_weights_to_use = [None]
                                elif training_variant in ['tokl']:
                                    label_types_to_use = ['icl_outputs']
                                    ce_weights_to_use = [None]
                                elif training_variant == 'tna':
                                    label_types_to_use = [None]
                                    ce_weights_to_use = args.ce_loss_weights
                                elif training_variant == 'act':
                                    label_types_to_use = [None]
                                    ce_weights_to_use = [None]
                                
                                # Generate experiments for this configuration
                                for label_type in label_types_to_use:
                                    for ce_weight in ce_weights_to_use:
                                        
                                        # Add LDR parameters if in LDR mode
                                        if args.ldr_mode:
                                            for num_unlabelled in args.num_unlabelled_samples:
                                                for max_perm in args.max_permutations:
                                                    if models_exist(args.model_id, training_method, dataset, ia3_type, None, None, num_tokens, num_examples, lr, label_type, ce_weight, args.num_runs, output_dir=args.output_dir, ldr_mode=True, num_labelled_samples=num_examples, num_unlabelled_samples=num_unlabelled, max_permutations=max_perm):
                                                        continue
                                                    
                                                    gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
                                                    gpu_cycle += 1
                                                    if training_variant == 'tok' and label_type == 'ground_truth': # skip ground truth for tok models when in LDR mode
                                                        continue
                                                    experiment = (
                                                        training_method,         # training_method
                                                        dataset,                 # dataset
                                                        args.model_id,          # model_id
                                                        ia3_type,               # ia3_type (replaces lora_type)
                                                        None,                   # lora_r (not used)
                                                        None,                   # lora_alpha (not used)
                                                        num_tokens,             # num_generated_tokens
                                                        num_examples,           # num_train_examples
                                                        lr,                     # lr
                                                        label_type,             # label_type
                                                        ce_weight,              # ce_loss_weight
                                                        gpu_id,                 # gpu_id
                                                        args.num_runs,          # run_idx
                                                        args.wandb_log,         # wandb_log
                                                        args.wandb_project,     # wandb_project
                                                        args.wandb_entity,      # wandb_entity
                                                        args.num_train_epochs,  # num_train_epochs
                                                        args.batch_size,        # batch_size
                                                        args.gradient_clip_val, # gradient_clip_val
                                                        args.patience,          # patience
                                                        args.dev_eval_steps,    # dev_eval_steps
                                                        args.log_gradient_norms,# log_gradient_norms
                                                        args.log_checkpoints,   # log_checkpoints
                                                        args.checkpoint_frequency, # checkpoint_frequency
                                                        args.shuffle_demos,     # shuffle_demos
                                                        args.num_shuffles,      # num_shuffles
                                                        args.output_dir,        # output_dir
                                                        True,                   # ldr_mode
                                                        num_examples,           # num_labelled_samples == num_train_examples
                                                        num_unlabelled,         # num_unlabelled_samples
                                                        max_perm,               # max_permutations
                                                        args.tokl_top_k         # tokl_top_k
                                                    )
                                                    all_experiments.append(experiment)
                                        else:
                                            if models_exist(args.model_id, training_method, dataset, ia3_type, None, None, num_tokens, num_examples, lr, label_type, ce_weight, args.num_runs, output_dir=args.output_dir, ldr_mode=False, num_labelled_samples=num_examples, num_unlabelled_samples=None, max_permutations=None):
                                                continue
                                            
                                            gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
                                            gpu_cycle += 1
                                                    
                                            experiment = (
                                                training_method,         # training_method
                                                dataset,                 # dataset
                                                args.model_id,          # model_id
                                                ia3_type,               # ia3_type (replaces lora_type)
                                                None,                   # lora_r (not used)
                                                None,                   # lora_alpha (not used)
                                                num_tokens,             # num_generated_tokens
                                                num_examples,           # num_train_examples
                                                lr,                     # lr
                                                label_type,             # label_type
                                                ce_weight,              # ce_loss_weight
                                                gpu_id,                 # gpu_id
                                                args.num_runs,          # run_idx
                                                args.wandb_log,         # wandb_log
                                                args.wandb_project,     # wandb_project
                                                args.wandb_entity,      # wandb_entity
                                                args.num_train_epochs,  # num_train_epochs
                                                args.batch_size,        # batch_size
                                                args.gradient_clip_val, # gradient_clip_val
                                                args.patience,          # patience
                                                args.dev_eval_steps,    # dev_eval_steps
                                                args.log_gradient_norms,# log_gradient_norms
                                                args.log_checkpoints,   # log_checkpoints
                                                args.checkpoint_frequency, # checkpoint_frequency
                                                args.shuffle_demos,     # shuffle_demos
                                                args.num_shuffles,      # num_shuffles
                                                args.output_dir,        # output_dir
                                                args.tokl_top_k         # tokl_top_k
                                            )
                                            all_experiments.append(experiment)
                                        
                        elif base_method in ['prompt', 'prefix']:
                            # Prompt/Prefix tuning method variants
                            for num_virtual_tokens in args.num_virtual_tokens_list:
                                # Determine relevant label_types and ce_loss_weights for this method
                                if training_variant in ['tok', 'a2t']:
                                    label_types_to_use = args.label_types
                                    ce_weights_to_use = [None]
                                elif training_variant in ['tokl']:
                                    label_types_to_use = ['icl_outputs']
                                    ce_weights_to_use = [None]
                                elif training_variant == 'tna':
                                    label_types_to_use = [None]
                                    ce_weights_to_use = args.ce_loss_weights
                                elif training_variant == 'act':
                                    label_types_to_use = [None]
                                    ce_weights_to_use = [None]
                                
                                # Generate experiments for this configuration
                                for label_type in label_types_to_use:
                                    for ce_weight in ce_weights_to_use:
                                        
                                        # Add LDR parameters if in LDR mode
                                        if args.ldr_mode:
                                            for num_unlabelled in args.num_unlabelled_samples:
                                                for max_perm in args.max_permutations:
                                                    if models_exist(args.model_id, training_method, dataset, None, None, None, num_tokens, num_examples, lr, label_type, ce_weight, args.num_runs, num_virtual_tokens, output_dir=args.output_dir, ldr_mode=True, num_labelled_samples=num_examples, num_unlabelled_samples=num_unlabelled, max_permutations=max_perm):
                                                        continue
                                                    
                                                    gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
                                                    gpu_cycle += 1
                                                    if training_variant == 'tok' and label_type == 'ground_truth': # skip ground truth for tok models when in LDR mode
                                                        continue
                                                    experiment = (
                                                        training_method,         # training_method
                                                        dataset,                 # dataset
                                                        args.model_id,          # model_id
                                                        None,                   # lora_type (not used)
                                                        None,                   # lora_r (not used)
                                                        None,                   # lora_alpha (not used)
                                                        num_tokens,             # num_generated_tokens
                                                        num_examples,           # num_train_examples
                                                        lr,                     # lr
                                                        label_type,             # label_type
                                                        ce_weight,              # ce_loss_weight
                                                        gpu_id,                 # gpu_id
                                                        args.num_runs,          # run_idx
                                                        args.wandb_log,         # wandb_log
                                                        args.wandb_project,     # wandb_project
                                                        args.wandb_entity,      # wandb_entity
                                                        args.num_train_epochs,  # num_train_epochs
                                                        args.batch_size,        # batch_size
                                                        args.gradient_clip_val, # gradient_clip_val
                                                        args.patience,          # patience
                                                        args.dev_eval_steps,    # dev_eval_steps
                                                        args.log_gradient_norms,# log_gradient_norms
                                                        args.log_checkpoints,   # log_checkpoints
                                                        args.checkpoint_frequency, # checkpoint_frequency
                                                        args.shuffle_demos,     # shuffle_demos
                                                        args.num_shuffles,      # num_shuffles
                                                        num_virtual_tokens,     # num_virtual_tokens
                                                        args.output_dir,        # output_dir
                                                        True,                   # ldr_mode
                                                        num_examples,           # num_labelled_samples == num_train_examples
                                                        num_unlabelled,         # num_unlabelled_samples
                                                        max_perm,               # max_permutations
                                                        args.tokl_top_k         # tokl_top_k
                                                    )
                                                    all_experiments.append(experiment)
                                        else:
                                            if models_exist(args.model_id, training_method, dataset, None, None, None, num_tokens, num_examples, lr, label_type, ce_weight, args.num_runs, num_virtual_tokens, output_dir=args.output_dir, ldr_mode=False, num_labelled_samples=num_examples, num_unlabelled_samples=None, max_permutations=None):
                                                continue
                                            
                                            gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
                                            gpu_cycle += 1
                                                    
                                            experiment = (
                                                training_method,         # training_method
                                                dataset,                 # dataset
                                                args.model_id,          # model_id
                                                None,                   # lora_type (not used)
                                                None,                   # lora_r (not used)
                                                None,                   # lora_alpha (not used)
                                                num_tokens,             # num_generated_tokens
                                                num_examples,           # num_train_examples
                                                lr,                     # lr
                                                label_type,             # label_type
                                                ce_weight,              # ce_loss_weight
                                                gpu_id,                 # gpu_id
                                                args.num_runs,          # run_idx
                                                args.wandb_log,         # wandb_log
                                                args.wandb_project,     # wandb_project
                                                args.wandb_entity,      # wandb_entity
                                                args.num_train_epochs,  # num_train_epochs
                                                args.batch_size,        # batch_size
                                                args.gradient_clip_val, # gradient_clip_val
                                                args.patience,          # patience
                                                args.dev_eval_steps,    # dev_eval_steps
                                                args.log_gradient_norms,# log_gradient_norms
                                                args.log_checkpoints,   # log_checkpoints
                                                args.checkpoint_frequency, # checkpoint_frequency
                                                args.shuffle_demos,     # shuffle_demos
                                                args.num_shuffles,      # num_shuffles
                                                num_virtual_tokens,     # num_virtual_tokens
                                                args.output_dir,        # output_dir
                                                args.tokl_top_k         # tokl_top_k
                                            )
                                            all_experiments.append(experiment)
                        else:
                            print(f"Unknown training method: {training_method}")
                            continue
    
    return all_experiments


def main():
    parser = argparse.ArgumentParser(description="Batch training for distillation project")
    
    # Training method configuration
    parser.add_argument("--training_methods", nargs='+', 
                        choices=['tok', 'tokl', 'act', 'tna', 'a2t', 't2a', 
                                'ia3-tok', 'ia3-tokl', 'ia3-act', 'ia3-tna', 'ia3-a2t', 'ia3-t2a',
                                'prompt-tok', 'prompt-tokl', 'prompt-act', 'prompt-tna', 'prompt-a2t', 'prompt-t2a',
                                'prefix-tok', 'prefix-tokl', 'prefix-act', 'prefix-tna', 'prefix-a2t', 'prefix-t2a'],
                        # default=['ia3-tok', 'ia3-act', 'ia3-tna', 'ia3-a2t'],
                        # default=['prefix-tok', 'prefix-act', 'prefix-tna', 'prefix-a2t'],
                        default=['tokl'],
                        help="Training methods to run")
    
    # Dataset and model configuration
    parser.add_argument("--datasets", nargs='+', default=['sst2', 'finsen', 'agnews'],
                        help="Datasets to train on")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B", choices=['meta-llama/Llama-3.2-1B', 'Qwen/Qwen3-4B-Base', 'Qwen/Qwen2.5-1.5B', 'meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.2-1B-Instruct', 'google/gemma-3-270m'],
                        help="Base model ID")
    parser.add_argument("--output_dir", type=str, default="../outputs",
                        help="Output directory")
    
    # LoRA hyperparameters
    parser.add_argument("--lora_types", nargs='+', default=['qko'],
                        help="LoRA types to use")
    parser.add_argument("--lora_rs", nargs='+', type=int, default=[8],
                        help="LoRA r values")
    parser.add_argument("--lora_alphas", nargs='+', type=int, default=[8],
                        help="LoRA alpha values")

    # IA3 hyperparameters
    parser.add_argument("--ia3_types", nargs='+', default=['qko'],
                        help="IA3 types to use")
    
    # Prompt/Prefix tuning hyperparameters
    parser.add_argument("--num_virtual_tokens_list", nargs='+', type=int, default=[20],
                        help="Number of virtual tokens for prompt/prefix tuning")
    
    # Training hyperparameters
    parser.add_argument("--num_generated_tokens", nargs='+', type=int, default=[1],
                        help="Number of generated tokens")
    parser.add_argument("--num_train_examples", nargs='+', type=int, default=[2, 4, 8, 16, 32],
                        help="Number of training examples")
    # parser.add_argument("--lrs", nargs='+', type=float, default=[1e-5, 3e-5],
    parser.add_argument("--lrs", nargs='+', type=float, default=[1e-4, 3e-4, 1e-3],
    # parser.add_argument("--lrs", nargs='+', type=float, default=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    # parser.add_argument("--lrs", nargs='+', type=float, default=[1e-3, 1e-2, 1e-1],
                        help="Learning rates")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs")
    
    # Method-specific parameters
    parser.add_argument("--label_types", nargs='+',
                        choices=['ground_truth', 'icl_outputs'],
                        default=['icl_outputs', 'ground_truth'],
                        help="Label types for tok models")
    parser.add_argument("--tokl_top_k", type=str, default='all',
                        help="Top-K for tokl probability targets; 'all' to store full logits")
    # parser.add_argument("--ce_loss_weights", nargs='+', type=float, default=[0.1, 0.5, 0.7, 0.9],
    parser.add_argument("--ce_loss_weights", nargs='+', type=float, default=[0.001, 0.01, 0.05, 0.5],
                        help="CE loss weights for tna models")
    
    # Training configuration (matching train_unified.py)
    parser.add_argument("--num_train_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--gradient_clip_val", type=float, default=None,
                        help="Value for gradient clipping")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--dev_eval_steps", type=int, default=2,
                        help="Evaluate dev loss every N training steps")
    
    # Data options
    parser.add_argument("--shuffle_demos", action='store_true',
                        help="Whether to use shuffled ICL demos")
    parser.add_argument("--num_shuffles", type=int, default=5,
                        help="Number of shuffles per example")
    
    # LDR (Low Data Regime) arguments
    parser.add_argument("--ldr_mode", action='store_true', 
                       help="Enable low data regime mode with N labelled samples and M unlabelled samples")
    parser.add_argument("--num_labelled_samples", nargs='+', type=int, default=[4], 
                       help="Number of labelled samples for ICL demos (N) - only used in LDR mode")
    parser.add_argument("--num_unlabelled_samples", nargs='+', type=int, default=[12], 
                       help="Number of unlabelled samples for training (M) - only used in LDR mode")
    parser.add_argument("--max_permutations", nargs='+', type=int, default=[1], 
                       help="Maximum number of permutations of labelled samples (K) - only used in LDR mode")
    
    # Logging options
    parser.add_argument("--log_gradient_norms", action='store_true',
                        help="Log gradient norms during training")
    parser.add_argument("--log_checkpoints", action='store_true',
                        help="Save training checkpoints")
    parser.add_argument("--checkpoint_frequency", type=int, default=1,
                        help="Checkpoint frequency (steps)")
    
    # WandB configuration
    parser.add_argument("--wandb_log", default=True, action='store_true',
                        help="Log to WandB")
    parser.add_argument("--wandb_project", type=str, default="distil901",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str,
                        help="WandB entity name")
    
    # System configuration
    parser.add_argument("--max_parallel", type=int, default=2,
                        help="Maximum number of parallel training jobs")
    parser.add_argument("--gpus", nargs='+', type=int, default=[3],
                        help="GPU IDs to use")
    parser.add_argument("--sequential_last", default=True, action='store_true',
                        help="Run base models (tok, act, tna) before sequential training (a2t, t2a)")
    
    args = parser.parse_args()
    
    print(f"Batch training configuration:")
    print(f"  Training methods: {args.training_methods}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Model: {args.model_id}")
    print(f"  Parallel jobs: {args.max_parallel}")
    print(f"  GPUs: {args.gpus}")

    if args.ldr_mode:
        args.num_examples = args.num_labelled_samples
    
    # Generate all training experiments
    all_experiments = generate_experiments(args)

    # Organize experiments by execution order if sequential_first is specified
    if args.sequential_last:
        base_methods = ['tok', 'act', 'ia3-tok', 'ia3-act', 'prompt-tok', 'prompt-act', 'prefix-tok', 'prefix-act']
        sequential_methods = ['a2t', 't2a', 'tna', 'ia3-a2t', 'ia3-t2a', 'ia3-tna', 'prompt-a2t', 'prompt-t2a','prompt-tna', 'prefix-a2t', 'prefix-t2a','prefix-tna']
        
        base_experiments = [exp for exp in all_experiments if exp[0] in base_methods]
        sequential_experiments = [exp for exp in all_experiments if exp[0] in sequential_methods]
        
        if base_experiments and sequential_experiments:
            print(f"\nRunning in two phases:")
            print(f"  Phase 1: {len(base_experiments)} base model experiments")
            print(f"  Phase 2: {len(sequential_experiments)} sequential training experiments")
            experiment_phases = [base_experiments, sequential_experiments]
        else:
            experiment_phases = [all_experiments]
    else:
        experiment_phases = [all_experiments]
    
    total_experiments = sum(len(phase) for phase in experiment_phases)
    print(f"\nTotal experiments: {total_experiments}")
    
    if not total_experiments:
        print("No experiments to run!")
        return

    # Show examples
    print("\nFirst 3 experiments:")
    example_count = 0
    for phase in experiment_phases:
        for i, exp in enumerate(phase[:3]):
            if example_count >= 3:
                break
            method_name = exp[0]
            dataset = exp[1]
            num_examples = exp[7]
            lr = exp[8]
            gpu_id = exp[11]
            print(f"  {example_count+1}. {method_name} on {dataset} (N={num_examples}, lr={lr}, GPU {gpu_id})")
            example_count += 1
        if example_count >= 3:
            break

    # input("\nPress Enter to start training...")

    # Execute experiments
    total_successful = 0
    total_failed = 0
    start_time = time.time()
    
    for phase_idx, experiments in enumerate(experiment_phases):
        if len(experiment_phases) > 1:
            print(f"\n{'='*60}")
            print(f"PHASE {phase_idx + 1}: Running {len(experiments)} experiments")
            print(f"{'='*60}")
        
        phase_start = time.time()
        
        # Run experiments in parallel
        with Pool(processes=args.max_parallel) as pool:
            results = pool.map(run_single_training, experiments)
        
        phase_end = time.time()
        
        # Phase summary
        phase_successful = sum(results)
        phase_failed = len(results) - phase_successful
        total_successful += phase_successful
        total_failed += phase_failed
        
        if len(experiment_phases) > 1:
            print(f"\nPhase {phase_idx + 1} Summary:")
            print(f"  Successful: {phase_successful}/{len(experiments)}")
            print(f"  Failed: {phase_failed}")
            print(f"  Time: {phase_end - phase_start:.1f} seconds")
    
    end_time = time.time()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_successful/total_experiments*100:.1f}%")
    print(f"Total time: {end_time - start_time:.1f} seconds")
    print(f"{'='*60}")
    
    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(args),
        'total_experiments': total_experiments,
        'successful': total_successful,
        'failed': total_failed,
        'success_rate': total_successful/total_experiments*100,
        'total_time_seconds': end_time - start_time,
        'phases': len(experiment_phases)
    }
    
    summary_path = f"../outputs/training_summaries/unified_batch_training_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    print(f"Models saved to: ../outputs/")


if __name__ == "__main__":
    main() 