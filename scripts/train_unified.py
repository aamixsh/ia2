#!/usr/bin/env python3
"""
Unified Training Script for Distillation Project

This script consolidates all training methods:
1. tok: Traditional CE loss training on ground truth or ICL output tokens
2. act: MSE loss training on activations from ICL outputs  
3. tna: Combined MSE + CE loss training
4. a2t, t2a: Combined MSE + CE loss training with sequential training from existing adapters
5. tokl: Soft token imitation training
"""

import os
import sys
import json
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from pathlib import Path

# Import PEFT components
from peft import (LoraConfig, IA3Config, PromptTuningConfig, PrefixTuningConfig, 
                 get_peft_model, PeftModel, TaskType, get_peft_model_state_dict)

# Import our custom modules
from data import ICLDataset, ICLDatasetWithOutputs, ICLDatasetShuffled
from utils import parse_answer_gsm8k, parse_answer_sciqa, get_model_name, parse_answer_boxed
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, logging
logging.set_verbosity_error()


def collect_icl_outputs(model, dataset, tokenizer, device, num_generated_tokens=1):
    """Collect ICL outputs from the model for each training example"""
    icl_outputs = {}
    icl_outputs_tokenized = {}

    generation_config = GenerationConfig(
        max_new_tokens=num_generated_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.0,
        do_sample=False,
        # stop_strings=['\n\n']
    )
    
    model.eval()
    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset, desc="Collecting ICL outputs")):
            # Process with ICL to get outputs
            inputs_with_icl = {k: v.to(device) for k, v in example['inputs_with_icl'].items()}
            decode_position = example['with_icl_decode_position']

            cut_input_ids = inputs_with_icl['input_ids'][:, :decode_position + 1]
            cut_attention_mask = inputs_with_icl['attention_mask'][:, :decode_position + 1]

            outputs = model.generate(
                tokenizer=tokenizer,
                input_ids=cut_input_ids,
                attention_mask=cut_attention_mask,
                generation_config=generation_config
            )

            # Get the generated sequence after the input
            generated_sequence = outputs[0][decode_position + 1:]

            # Decode tokens and store both text and ids
            generated_tokens = [tokenizer.decode(token) for token in generated_sequence]
            generated_tokens_tokenized = generated_sequence.detach().cpu().numpy().tolist()

            if len(generated_sequence) < num_generated_tokens:
                # if generated_sequence[-1] == tokenizer.encode('\n\n', add_special_tokens=False):
                #     generated_sequence[-1] = tokenizer.eos_token_id
                if generated_sequence[-1] != tokenizer.eos_token_id:
                    generated_tokens.append(tokenizer.eos_token)
                    generated_tokens_tokenized.append(tokenizer.eos_token_id)

            icl_outputs[idx] = generated_tokens
            icl_outputs_tokenized[idx] = generated_tokens_tokenized

    torch.cuda.empty_cache()
    return icl_outputs, icl_outputs_tokenized


def collect_icl_distributions(model, dataset, tokenizer, device, num_generated_tokens=1, top_k=100):
    """Efficiently collect output distributions using generate(output_scores=True).

    If top_k == 'all' (string), stores full logits per step over the entire vocab.
    Else, stores top-k probabilities and indices per step.
    Returns:
      - generated_tokens_all: List[List[int]]
      - distributions_all: List[List[Union[Dict[str, Tensor], Tensor]]]
    """
    model.eval()
    generated_tokens_all = []
    distributions_all = []

    with torch.no_grad():
        generation_config = GenerationConfig(
            max_new_tokens=num_generated_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

        for _, example in enumerate(tqdm(dataset, desc="Collecting ICL distributions (scores)")):
            inputs_with_icl = {k: v.to(device) for k, v in example['inputs_with_icl'].items()}
            decode_position = example['with_icl_decode_position']

            outputs = model.generate(
                tokenizer=tokenizer,
                input_ids=inputs_with_icl['input_ids'][:, :decode_position + 1],
                attention_mask=inputs_with_icl['attention_mask'][:, :decode_position + 1],
                generation_config=generation_config
            )

            sequences = outputs.sequences  # [1, seq_len]
            scores = outputs.scores        # tuple(len = generated steps), each [1, vocab]

            # Extract generated token ids
            input_length = decode_position + 1
            gen_tokens = sequences[0, input_length:]
            gen_list = gen_tokens.detach().cpu().tolist()

            step_distributions = []
            if isinstance(top_k, str) and top_k == 'all':
                # Store full logits per step
                for s in scores:
                    step_distributions.append(s[0].detach().cpu())  # Tensor[vocab]
            else:
                k = min(int(top_k), scores[0].shape[-1]) if len(scores) > 0 else 0
                for s in scores:
                    logits = s[0]
                    probs = F.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, k=k)
                    step_distributions.append({
                        'indices': top_indices.detach().cpu(),
                        'probs': top_probs.detach().cpu()
                    })

            generated_tokens_all.append(gen_list)
            distributions_all.append(step_distributions)

    torch.cuda.empty_cache()
    return generated_tokens_all, distributions_all


def register_attention_hooks(model):
    """Register forward hooks on attention layers and return hook storage and hook handles"""
    attention_outputs = {}
    hooks = []

    def create_attention_hook(layer_name):
        """Create a hook function for a specific attention layer"""
        def hook(module, input, output):
            # For transformer models, attention output is typically the first element
            # output[0] contains the attention output tensor [batch_size, seq_len, hidden_size]
            try:
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    attention_outputs[layer_name] = output[0].clone()
                else:
                    attention_outputs[layer_name] = output.clone()
            except Exception as e:
                print(f"Warning: Failed to capture attention output for {layer_name}: {e}")
        return hook

    # Register hooks on all attention layers with robust detection
    hook_count = 0
    for name, module in model.named_modules():
        # More robust attention layer detection
        if ('self_attn' in name and name.endswith('self_attn')) or \
           (hasattr(module, '__class__') and 'Attention' in module.__class__.__name__):
            hook = module.register_forward_hook(create_attention_hook(name))
            hooks.append(hook)
            hook_count += 1
            # print(f"Registered hook on: {name} ({module.__class__.__name__})")
    
    if hook_count == 0:
        print("Warning: No attention layers found! Check model architecture.")
        # Fallback: try to find any module with 'attn' in name
        for name, module in model.named_modules():
            if 'attn' in name.lower():
                print(f"Found potential attention module: {name} ({module.__class__.__name__})")
    else:
        print(f"Successfully registered {hook_count} attention hooks")
    
    return attention_outputs, hooks


def remove_hooks(hooks):
    """Remove all hooks safely"""
    for hook in hooks:
        hook.remove()
    hooks.clear()


def get_sorted_attention_layers(attention_outputs):
    """Extract and sort attention layer names by layer index"""
    layer_names = [name for name in attention_outputs.keys() if 'self_attn' in name]
    return sorted(layer_names, key=lambda x: int(x.split('.')[-2]) if 'layers.' in x else 0)


def get_activations(model, dataset, tokenizer, device, num_generated_tokens=1, get_with_icl=True, get_without_icl=False, ground_truth_tokens=False):
    """Get attention layer activations during generation using forward hooks."""
    model.eval()

    activations_data = {
        'with_icl': {},
        'without_icl': {}
    }

    # Register hooks on attention layers
    attention_outputs, hooks = register_attention_hooks(model)

    try:
        with torch.no_grad():
            for idx, example in enumerate(tqdm(dataset, desc="Collecting Attention Activations & Tokens")):

                activations_data['with_icl'][idx] = {'activations': None, 'generated_tokens': []}
                activations_data['without_icl'][idx] = {'activations': None, 'generated_tokens': []}

                # --- Process with ICL (Collect targets) ---
                if 'inputs_with_icl' in example and get_with_icl:
                    inputs_with_icl = {k: v.to(device) for k, v in example['inputs_with_icl'].items()}
                    decode_position = example['with_icl_decode_position']
                    input_ids = inputs_with_icl['input_ids']
                    attention_mask = inputs_with_icl['attention_mask']

                    activation_tensor = []

                    if ground_truth_tokens:
                        num_generated_tokens = len(example['ground_truth'])

                    for token_idx in range(num_generated_tokens):
                        # Ensure input length is valid before forward pass
                        if decode_position + token_idx + 1 >= input_ids.shape[1]:
                            print(f"Warning (Act. with ICL): Input length exceeds capacity at token {token_idx}. Stopping.")
                            break

                        # Clear previous attention outputs
                        attention_outputs.clear()

                        outputs = model(input_ids=input_ids[:, :decode_position + token_idx + 1], 
                                       attention_mask=attention_mask[:, :decode_position + token_idx + 1])

                        # Extract attention outputs from hooks (sorted by layer order)
                        layer_names = get_sorted_attention_layers(attention_outputs)

                        # Store attention activations for the current generated token position
                        step_activations = torch.stack([
                            attention_outputs[layer_name][0, -1].cpu().clone() 
                            for layer_name in layer_names
                        ]).unsqueeze(1)
                        activation_tensor.append(step_activations)
                        gen_pos = decode_position + token_idx

                        if ground_truth_tokens:
                            next_token_item = example['ground_truth'][token_idx]
                        else:
                            # Update inputs for next step
                            next_token_logits = outputs.logits[:, gen_pos, :]
                            next_token = torch.argmax(next_token_logits, dim=-1)
                            next_token_item = next_token[0].item()
                            # Store the generated token
                            activations_data['with_icl'][idx]['generated_tokens'].append(next_token_item)

                            if next_token_item == tokenizer.eos_token_id: 
                                break  # Stop if EOS
                            # elif next_token_item == tokenizer.encode('\n\n', add_special_tokens=False)[0]:
                            #     activations_data['with_icl'][idx]['generated_tokens'][-1] = tokenizer.eos_token_id
                            #     break

                        if gen_pos + 1 < input_ids.shape[1]:
                            input_ids[0, gen_pos + 1] = next_token_item
                            attention_mask[0, gen_pos + 1] = 1
                        else:
                            print(f"Warning (Act. with ICL): Reached max sequence length at token {token_idx}.")
                            break

                    activation_tensor = torch.cat(activation_tensor, dim=1)
                    activations_data['with_icl'][idx]['activations'] = activation_tensor
                    
                    del inputs_with_icl, input_ids, attention_mask, outputs, next_token_logits, next_token
                    torch.cuda.empty_cache()

                # --- Process without ICL ---
                if 'inputs_no_icl' in example and get_without_icl:
                    inputs_no_icl = {k: v.to(device) for k, v in example['inputs_no_icl'].items()}
                    decode_position = example['no_icl_decode_position']
                    input_ids = inputs_no_icl['input_ids']
                    attention_mask = inputs_no_icl['attention_mask']

                    activation_tensor = []

                    if ground_truth_tokens:
                        num_generated_tokens = len(example['ground_truth'])

                    for token_idx in range(num_generated_tokens):
                        if decode_position + token_idx + 1 >= input_ids.shape[1]:
                            print(f"Warning (Act. no ICL): Input length exceeds capacity at token {token_idx}. Stopping.")
                            break

                        # Clear previous attention outputs
                        attention_outputs.clear()

                        outputs = model(input_ids=input_ids[:, :decode_position + token_idx + 1], 
                                       attention_mask=attention_mask[:, :decode_position + token_idx + 1])

                        # Extract attention outputs from hooks (sorted by layer order)
                        layer_names = get_sorted_attention_layers(attention_outputs)

                        step_activations = torch.stack([
                            attention_outputs[layer_name][0, -1].cpu().clone() 
                            for layer_name in layer_names
                        ]).unsqueeze(1)
                        activation_tensor.append(step_activations)
                        gen_pos = decode_position + token_idx

                        if ground_truth_tokens:
                            next_token_item = example['ground_truth'][token_idx]
                        else:
                            next_token_logits = outputs.logits[:, gen_pos, :]
                            next_token = torch.argmax(next_token_logits, dim=-1)
                            next_token_item = next_token[0].item()

                            activations_data['without_icl'][idx]['generated_tokens'].append(next_token_item)

                            if next_token_item == tokenizer.eos_token_id: 
                                break
                            # elif next_token_item == tokenizer.encode('\n\n', add_special_tokens=False)[0]: 
                            #     activations_data['without_icl'][idx]['generated_tokens'][-1] = tokenizer.eos_token_id
                            #     break  # Stop if end of example

                        if gen_pos + 1 < input_ids.shape[1]:
                            input_ids[0, gen_pos + 1] = next_token_item
                            attention_mask[0, gen_pos + 1] = 1
                        else:
                            print(f"Warning (Act. no ICL): Reached max sequence length at token {token_idx}.")
                            break

                    activation_tensor = torch.cat(activation_tensor, dim=1)
                    activations_data['without_icl'][idx]['activations'] = activation_tensor

                    del inputs_no_icl, input_ids, attention_mask, outputs, next_token_logits, next_token
                    torch.cuda.empty_cache()

    finally:
        # Remove all hooks
        remove_hooks(hooks)

    torch.cuda.empty_cache()
    return activations_data


def create_lora_config(lora_type, lora_r, lora_alpha):
    """Create standardized LoRA configuration"""
    target_modules = []
    if 'q' in lora_type:
        target_modules.append("q_proj")
    if 'k' in lora_type:
        target_modules.append("k_proj")
    if 'v' in lora_type:
        target_modules.append("v_proj")
    if 'o' in lora_type:
        target_modules.append("o_proj")
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,  # No dropout for consistency
        target_modules=target_modules,
        bias="none",
    )


def create_ia3_config(ia3_type):
    """Create IA3 configuration"""
    target_modules = []
    if 'q' in ia3_type:
        target_modules.append("q_proj")
    if 'k' in ia3_type:
        target_modules.append("k_proj")
    if 'v' in ia3_type:
        target_modules.append("v_proj")
    if 'o' in ia3_type:
        target_modules.append("o_proj")
    
    return IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        feedforward_modules=target_modules,
        fan_in_fan_out=False,
    )


def create_prompt_tuning_config(num_virtual_tokens):
    """Create Prompt Tuning configuration"""
    return PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init="RANDOM",
    )


def create_prefix_tuning_config(num_virtual_tokens):
    """Create Prefix Tuning configuration"""
    return PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
    )


def load_base_or_continue_model(args, device, torch_dtype, run_idx, base_model=None, base_method=None, training_variant=None, output_dir=None):
    """Load base model or continue from existing adapter"""
    if base_model is None:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
        ).to(device)

    ldr_suffix = f"_N{args.num_labelled_samples}_M{args.num_unlabelled_samples}_K{args.max_permutations}" if args.ldr_mode else ""

    if base_method == "ia3":
        ia3_type = args.ia3_type
    else:
        ia3_type = None
    if base_method in ["prompt", "prefix"]:
        num_virtual_tokens = args.num_virtual_tokens
    else:
        num_virtual_tokens = None
        
    if training_variant not in ['a2t', 't2a']:
        return base_model, False  # False indicates fresh model
    else:
        if training_variant == 'a2t':
            if base_method in ["ia3", "prompt", "prefix"]:
                method_dir = f"{base_method}-act"
            else:
                method_dir = "act"
        elif training_variant == 't2a':
            if base_method in ["ia3", "prompt", "prefix"]:
                method_dir = f"{base_method}-tok"
            else:
                method_dir = "tok"
        starting_model_output_dir = f"{output_dir}/{method_dir}/{args.dataset}"
        adapter_name = get_model_name(f"{method_dir}", args.model_id.split('/')[-1], args.lora_type, args.lora_r, args.lora_alpha, args.num_generated_tokens, args.num_train_examples, args.lr, run_idx, label_type=None, ce_loss_weight=None, ia3_type=ia3_type, num_virtual_tokens=num_virtual_tokens, ldr_mode=args.ldr_mode, num_labelled_samples=args.num_labelled_samples, num_unlabelled_samples=args.num_unlabelled_samples, max_permutations=args.max_permutations)
        continue_from = os.path.join(starting_model_output_dir, adapter_name)
        
        if not os.path.exists(continue_from):
            raise ValueError(f"PEFT model not found at {continue_from}. Please train the starting model first.")

        print(f"Loading PEFT model from {continue_from}")
        # Load PEFT model with existing adapter
        model = PeftModel.from_pretrained(base_model, continue_from, is_trainable=True)
        model.print_trainable_parameters()
        print(f"PEFT model loaded successfully!")

        return model, True  # True indicates continued model


def compute_loss(training_method, model, example, tokenizer, device, 
                 attention_outputs=None, ce_loss_weight=1.0, label_type='icl_outputs'):
    """Compute loss based on training method"""
    
    # Initialize losses
    ce_loss = torch.tensor(0.0, device=device)
    mse_loss = torch.tensor(0.0, device=device)
    
    # For TNA, we need to do a single forward pass that can compute both losses
    if training_method == "tna":
        # Use inputs compatible with both CE and MSE loss computation
        if label_type == 'ground_truth':
            inputs = example['inputs_with_ground_truth']
            target_tokens = example['ground_truth'].to(device)
        else:
            inputs = example['inputs_with_icl_outputs']
            target_tokens = example['icl_output'].to(device)
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        decode_position = example['no_icl_decode_position']
        
        # Prepare labels for CE loss
        labels = torch.full_like(input_ids, -100)
        labels[0, -len(target_tokens):] = target_tokens
        
        # Clear attention outputs for this forward pass
        if attention_outputs is not None:
            attention_outputs.clear()
        
        # Single forward pass for both losses
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        ce_loss = outputs.loss
        
        # Compute MSE loss from attention outputs if available
        if attention_outputs is not None:
            layer_names = get_sorted_attention_layers(attention_outputs)
            if layer_names:
                # Get current attention activations for the output positions (excluding the last token)
                current_activations = torch.stack([
                    attention_outputs[layer_name][0, decode_position:decode_position + len(target_tokens)] 
                    for layer_name in layer_names
                ])

                mse_loss = F.mse_loss(current_activations, example['icl_activations'].to(device)) * len(layer_names)
        
        # Combine losses
        total_loss = (1 - ce_loss_weight) * mse_loss + ce_loss_weight * ce_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'mse_loss': mse_loss
        }
    
    # Handle pure CE loss methods (tok, a2t, ia3, prompt, prefix)
    elif training_method in ["tok", "a2t", "ia3", "prompt", "prefix"]:
        if label_type == 'ground_truth':
            inputs = example['inputs_with_ground_truth']
            target_tokens = example['ground_truth'].to(device)
        else:
            inputs = example['inputs_with_icl_outputs']
            target_tokens = example['icl_output'].to(device)
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        labels = torch.full_like(input_ids, -100)
        labels[0, -len(target_tokens):] = target_tokens

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        ce_loss = outputs.loss

        return {
            'total_loss': ce_loss,
            'ce_loss': ce_loss,
            'mse_loss': mse_loss
        }

    # Handle soft token imitation (tokl)
    elif training_method in ["tokl"]:
        if label_type != 'icl_outputs':
            raise ValueError("'tokl' only supports label_type='icl_outputs'")

        if 'inputs_with_icl_outputs' not in example:
            raise ValueError("inputs_with_icl_outputs missing for tokl")
        if 'icl_distributions' not in example:
            raise ValueError("icl_distributions missing for tokl")

        inputs = example['inputs_with_icl_outputs']
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        decode_position = example['no_icl_decode_position']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab]
        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [seq_len, vocab]

        vocab_size = log_probs.shape[-1]
        step_losses = []

        distributions = example['icl_distributions']
        # Number of generated steps equals provided distributions length
        for t, dist in enumerate(distributions):
            pos = decode_position + t
            if pos >= log_probs.shape[0]:
                break

            if isinstance(dist, dict):
                top_indices = dist['indices'].to(device)
                top_probs = dist['probs'].to(device)

                k = top_indices.shape[0]
                sum_top = top_probs.sum()
                remaining = torch.clamp(1.0 - sum_top, min=0.0)
                denom = max(vocab_size - k, 1)
                uniform_rest = remaining / denom

                target = torch.full((vocab_size,), uniform_rest, device=device, dtype=log_probs.dtype)
                target.scatter_(0, top_indices, top_probs)
                kl = F.kl_div(log_probs[pos], target, reduction='sum')
                step_losses.append(kl)
            else:
                # Full logits provided for target (Tensor[vocab]); convert to probability target directly
                target_logits = dist.to(device)
                target_probs = F.softmax(target_logits, dim=-1)
                kl = F.kl_div(log_probs[pos], target_probs, reduction='sum')
                step_losses.append(kl)

        if len(step_losses) == 0:
            total = torch.tensor(0.0, device=device)
        else:
            total = torch.stack(step_losses).mean()

        return {
            'total_loss': total,
            'ce_loss': total,
            'mse_loss': torch.tensor(0.0, device=device)
        }
    
    # Handle pure MSE loss methods (act, t2a)
    elif training_method in ["act", "t2a"]:
        inputs = example['inputs_with_icl_outputs']
        decode_position = example['no_icl_decode_position']
        
        input_ids = inputs['input_ids'].to(device) # We don't need to process the last token as its activations are not used.
        attention_mask = inputs['attention_mask'].to(device)

        # Clear attention outputs for this forward pass
        if attention_outputs is not None:
            attention_outputs.clear()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute MSE loss from attention outputs if available
        if attention_outputs is not None:
            layer_names = get_sorted_attention_layers(attention_outputs)
            if layer_names:
                # Get current attention activations for the output positions (excluding the last token)
                current_activations = torch.stack([
                    attention_outputs[layer_name][0, decode_position:decode_position + len(example['icl_output'])]
                    for layer_name in layer_names
                ])

                mse_loss = F.mse_loss(current_activations, example['icl_activations'].to(device)) * len(layer_names)

        return {
            'total_loss': mse_loss,
            'ce_loss': ce_loss,
            'mse_loss': mse_loss
        }
    
    else:
        raise ValueError(f"Unknown training method: {training_method}")


def evaluate_dev_loss(training_method, model, dev_dataset, tokenizer, device, 
                     attention_outputs=None, ce_loss_weight=1.0, label_type='icl_outputs'):
    """Evaluate development loss"""
    model.eval()
    total_losses = {'total': 0.0, 'ce': 0.0, 'mse': 0.0}
    
    with torch.no_grad():
        for i in range(len(dev_dataset)):
            example = dev_dataset[i]
            losses = compute_loss(training_method, model, example, tokenizer, device,
                                attention_outputs, ce_loss_weight, label_type)
            
            total_losses['total'] += losses['total_loss'].item()
            total_losses['ce'] += losses['ce_loss'].item()
            total_losses['mse'] += losses['mse_loss'].item()
    
    # Average the losses
    n_examples = len(dev_dataset)
    for key in total_losses:
        total_losses[key] /= n_examples
    
    model.train()
    return total_losses


def train_unified_model(
    training_method,
    model,
    train_dataset,
    dev_dataset,
    tokenizer,
    device,
    num_epochs=5,
    patience=5, 
    lr=1e-4,
    batch_size=4,
    gradient_clip_val=None,
    ce_loss_weight=1.0,
    label_type='icl_outputs',
    run_info=None,
    wandb_log=False,
    model_save_path=None,
    dev_eval_steps=2,
    log_gradient_norms=False,
    log_weight_norms=False,
    log_checkpoints=False,
    checkpoint_frequency=1
):
    """Unified training function for all methods"""
    
    # Get trainable parameters from PEFT model
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    os.makedirs(model_save_path, exist_ok=True)
    if log_checkpoints:
        os.makedirs(os.path.join(model_save_path, "checkpoints"), exist_ok=True)

    # Setup for logging
    gradient_norms_log = [] if log_gradient_norms else None
    weight_norms_log = [] if log_weight_norms else None
    
    # Register attention hooks once for methods that need them
    attention_outputs = None
    hooks = []
    if training_method in ["act", "t2a", "tna"]:
        print(f"Registering attention hooks for {training_method} training...")
        attention_outputs, hooks = register_attention_hooks(model)

    # Memory efficient processing - process one example at a time
    accumulation_steps = min(batch_size, len(train_dataset))

    print(f"Training {training_method} with memory-efficient gradient accumulation (accumulation_steps={accumulation_steps})")
    print(f"Dev evaluation every {dev_eval_steps} steps")

    # Training Loop
    model.train()
    best_dev_loss = float('inf')
    best_state = None
    patience_counter = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        total_losses = {'total': 0.0, 'ce': 0.0, 'mse': 0.0}
        accumulated_examples = 0
        
        train_indices = list(range(len(train_dataset)))
        np.random.shuffle(train_indices)
        pbar = tqdm(train_indices, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, data_idx in enumerate(pbar):
            example = train_dataset[data_idx]
            
            # Compute loss based on training method
            losses = compute_loss(training_method, model, example, tokenizer, device,
                                attention_outputs, ce_loss_weight, label_type)
            
            # Scale loss for accumulation
            scaled_loss = losses['total_loss'] / accumulation_steps
            scaled_loss.backward()

            # Accumulate losses for logging
            total_losses['total'] += losses['total_loss'].item()
            total_losses['ce'] += losses['ce_loss'].item()
            total_losses['mse'] += losses['mse_loss'].item()
            accumulated_examples += 1

            # Update parameters when accumulation is complete
            if accumulated_examples % accumulation_steps == 0:
                if gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_val)
                
                global_step += 1
                optimizer.step()
                optimizer.zero_grad()
                
                # Log gradient norms if requested
                if log_gradient_norms:
                    total_grad_norm = 0.0
                    for param in trainable_params:
                        if param.grad is not None:
                            total_grad_norm += param.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    gradient_norms_log.append(total_grad_norm)
                
                # Log weight norms if requested
                if log_weight_norms:
                    total_weight_norm = 0.0
                    for param in trainable_params:
                        total_weight_norm += param.data.norm(2).item() ** 2
                    total_weight_norm = total_weight_norm ** 0.5
                    weight_norms_log.append(total_weight_norm)
                
                # Evaluate on dev set every dev_eval_steps
                if global_step % dev_eval_steps == 0 or accumulated_examples == len(train_dataset):
                    dev_losses = evaluate_dev_loss(training_method, model, dev_dataset, tokenizer, 
                                                 device, attention_outputs, ce_loss_weight, label_type)
                    
                    # Check for improvement
                    if dev_losses['total'] < best_dev_loss:
                        best_dev_loss = dev_losses['total']
                        patience_counter = 0
                        best_state = get_peft_model_state_dict(model)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping triggered after {global_step} steps.")
                            # Restore best model state
                            if best_state:
                                model.load_state_dict(best_state, strict=False)
                            # Clean up attention hooks before early return
                            if hooks:
                                print(f"Cleaning up {len(hooks)} attention hooks...")
                                remove_hooks(hooks)

                            # Update progress bar
                            avg_losses = {k: v / max(accumulated_examples, 1) for k, v in total_losses.items()}
                            
                            # Save final training loss statistics
                            save_final_loss_statistics(model_save_path, best_dev_loss, avg_losses, epoch, global_step)

                            return model
                    
                    print(f"\nStep {global_step} - Dev Loss: {dev_losses['total']:.6f} "
                          f"(CE: {dev_losses['ce']:.6f}, MSE: {dev_losses['mse']:.6f}), patience: {patience_counter}")
                    
                    # Log to wandb if enabled
                    if wandb_log:
                        wandb.log({
                            'dev_loss_total': dev_losses['total'],
                            'dev_loss_ce': dev_losses['ce'],
                            'dev_loss_mse': dev_losses['mse'],
                            'step': global_step,
                            'epoch': epoch
                        })
                
                # Save checkpoint if requested
                if log_checkpoints and global_step % checkpoint_frequency == 0 and model_save_path:
                    checkpoint_path = os.path.join(model_save_path, "checkpoints", f"step_{global_step}")
                    model.save_pretrained(checkpoint_path)
            
            # Update progress bar
            avg_losses = {k: v / max(accumulated_examples, 1) for k, v in total_losses.items()}
            pbar.set_postfix({
                'Loss': f"{avg_losses['total']:.6f}",
                'CE': f"{avg_losses['ce']:.6f}",
                'MSE': f"{avg_losses['mse']:.6f}"
            })
            
            # Clear cache
            torch.cuda.empty_cache()
        
        # End of epoch logging
        avg_losses = {k: v / len(train_dataset) for k, v in total_losses.items()}
        print(f"Epoch {epoch+1} completed. Average losses - Total: {avg_losses['total']:.6f}, "
              f"CE: {avg_losses['ce']:.6f}, MSE: {avg_losses['mse']:.6f}")
        
        if wandb_log:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss_total': avg_losses['total'],
                'train_loss_ce': avg_losses['ce'],
                'train_loss_mse': avg_losses['mse'],
            })
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Clean up attention hooks
    if hooks:
        print(f"Cleaning up {len(hooks)} attention hooks...")
        remove_hooks(hooks)
    
    # Save final training loss statistics
    save_final_loss_statistics(model_save_path, best_dev_loss, avg_losses, epoch, global_step)
    
    return model

def save_final_loss_statistics(model_save_path, best_dev_loss, avg_losses, epoch, global_step):
    # Save final training loss statistics
    if model_save_path:
        final_loss_stats = {
            'best_dev_loss': best_dev_loss,
            'current_train_loss': {
                'mse': avg_losses['mse'],
                'ce': avg_losses['ce'],
                'total': avg_losses['total']
            },
            'training_completed': True,
            'final_epoch': epoch + 1,
            'total_steps': global_step
        }
        
        loss_stats_path = os.path.join(model_save_path, "final_loss_statistics.json")
        with open(loss_stats_path, 'w') as f:
            json.dump(final_loss_stats, f, indent=2)
        print(f"Final loss statistics saved to {loss_stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Distillation Training Script")
    
    # Core arguments
    parser.add_argument("--training_method", type=str, default="tok",
                       choices=["tok", "tokl", "act", "tna", "a2t", "t2a", 
                               "ia3-tok", "ia3-tokl", "ia3-act", "ia3-tna", "ia3-a2t", "ia3-t2a",
                               "prompt-tok", "prompt-tokl", "prompt-act", "prompt-tna", "prompt-a2t", "prompt-t2a",
                               "prefix-tok", "prefix-tokl", "prefix-act", "prefix-tna", "prefix-a2t", "prefix-t2a"],
                       help="Training method to use")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Base", 
                       help="Base model ID")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    
    # LoRA configuration
    parser.add_argument("--lora_type", type=str, default='qko')
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)

    # IA3 configuration
    parser.add_argument("--ia3_type", type=str, default='qko',
                       help="IA3 configuration type (e.g., qkv)")
    
    # Prompt/Prefix tuning configuration  
    parser.add_argument("--num_virtual_tokens", type=int, default=20,
                       help="Number of virtual tokens for prompt/prefix tuning")
    
    # Training configuration
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--num_train_examples", type=int, default=4)
    parser.add_argument("--num_generated_tokens", type=int, default=1, 
                       help="Number of tokens for training")
    parser.add_argument("--tokl_top_k", type=str, default='all',
                       help="Top-K probabilities to collect for tokl targets; use 'all' to store full logits")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_clip_val", type=float, default=None, 
                       help="Value for gradient clipping")
    parser.add_argument("--patience", type=int, default=5, 
                       help="Number of epochs to wait for improvement before early stopping")
    
    # Method-specific arguments
    parser.add_argument("--ce_loss_weight", type=float, default=0.02,
                       help="Weight for CE loss in tna (0 <= weight <= 1)")
    parser.add_argument("--label_type", type=str, default='icl_outputs', 
                       choices=['ground_truth', 'icl_outputs'],
                       help="Type of labels to use for training")
    
    # Data options
    parser.add_argument("--shuffle_demos", action='store_true', 
                       help="Whether to use shuffled ICL demos")
    parser.add_argument("--num_shuffles", type=int, default=5, 
                       help="Number of shuffles per example")
    
    # LDR (Low Data Regime) arguments
    parser.add_argument("--ldr_mode", action='store_true', 
                       help="Enable low data regime mode with N labelled samples and M unlabelled samples")
    parser.add_argument("--num_labelled_samples", type=int, default=8, 
                       help="Number of labelled samples for ICL demos (N) - only used in LDR mode")
    parser.add_argument("--num_unlabelled_samples", type=int, default=1000, 
                       help="Number of unlabelled samples for training (M) - only used in LDR mode")
    parser.add_argument("--max_permutations", type=int, default=24, 
                       help="Maximum number of permutations of labelled samples (K) - only used in LDR mode")
    
    # System configuration
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", 
                       choices=["float", "bfloat16"])
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--dev_eval_steps", type=int, default=2, 
                       help="Evaluate dev loss every N training steps")
    
    # Logging arguments
    parser.add_argument("--log_gradient_norms", action='store_true',
                       help="Log gradient norms for LoRA parameters during training")
    parser.add_argument("--log_weight_norms", action='store_true',
                       help="Log weight norms for LoRA parameters during training")
    parser.add_argument("--log_checkpoints", action='store_true',
                       help="Log model checkpoints during training")
    parser.add_argument("--checkpoint_frequency", type=int, default=1,
                       help="Save checkpoint every N training steps")
    
    # Wandb arguments
    parser.add_argument("--wandb_log", action='store_true', help="Log training to wandb")
    parser.add_argument("--wandb_project", type=str, default="distil", 
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                       help="Wandb entity name (optional)")

    args = parser.parse_args()

    args.wandb_project = f"{args.wandb_project}_{args.training_method}"
    
    # Validate arguments
    if args.training_method == "tna" and not (0 <= args.ce_loss_weight <= 1):
        raise ValueError(f"ce_loss_weight must be between 0 and 1, but got {args.ce_loss_weight}")
    # Enforce label type for tokl
    if (args.training_method == 'tokl' or args.training_method.endswith('-tokl')) and args.label_type != 'icl_outputs':
        raise ValueError("tokl only supports label_type='icl_outputs'")
        
    # Set device and dtype
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float if args.torch_dtype == "float" else torch.bfloat16
    print(f"Using device: {device}")
    
    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model_name = args.model_id.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create output directory
    output_base_dir = f"{args.output_dir}/{args.training_method}/{args.dataset}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Define parse_answer_func for each dataset
    parse_answer_func = None
    if args.dataset == 'gsm8k':
        parse_answer_func = parse_answer_gsm8k
    elif args.dataset == 'sciqa' or args.dataset == 'strategyreason':
        parse_answer_func = parse_answer_sciqa
    elif 'hmath' in args.dataset:
        parse_answer_func = parse_answer_boxed
    
    print(f"Starting {args.num_runs} runs for {args.training_method} training...")
    
    for run_idx in range(args.num_runs):
        print(f"\n{'='*60}")
        print(f"Starting run {run_idx + 1}/{args.num_runs}")
        print(f"{'='*60}")

        if '-' in args.training_method:
            base_method, training_variant = args.training_method.split('-', 1)
        else:
            base_method = args.training_method
            training_variant = args.training_method
        
        # Setup wandb for this run
        wandb_run = None
        if args.wandb_log:
            ldr_suffix = f"_N{args.num_labelled_samples}_M{args.num_unlabelled_samples}_K{args.max_permutations}" if args.ldr_mode else ""
            run_name = f"{args.training_method}_{args.dataset}_{model_name}_T{args.num_generated_tokens}_N{args.num_train_examples}_lr{args.lr}_{args.lora_type}_r{args.lora_r}_a{args.lora_alpha}_run{run_idx + 1}{ldr_suffix}"
            
            if training_variant in ["a2t", "tok"]:
                run_name += f"_label{args.label_type}"
            if training_variant == "tna":
                run_name += f"_cew{args.ce_loss_weight}"
            if training_variant == "tokl":
                run_name += f"_topk{args.tokl_top_k}"

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                reinit='return_previous'
            )

        # Step 1: Load base model
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch_dtype,
        ).to(device)
        
        try:
            # --- STEP 2: Load datasets ---
            print("Loading datasets...")
            
            # Determine dataset path based on LDR mode and training method
            if args.ldr_mode:
                # Check if this is a2t with ground_truth - in this case, use original dataset
                if '-' in args.training_method:
                    base_method, training_variant = args.training_method.split('-', 1)
                else:
                    base_method = args.training_method
                    training_variant = args.training_method
                
                if training_variant == 'a2t' and args.label_type == 'ground_truth':
                    # Special case: a2t with ground_truth in LDR mode uses original dataset
                    data_cache_path = f"../data/{args.dataset}/train/{args.num_labelled_samples}_{run_idx}.json"
                    print(f"LDR mode with a2t ground_truth: Loading original dataset from {data_cache_path}")
                else:
                    # Standard LDR mode: use {N}_{M}_{K}_{run_idx}.json format
                    data_cache_path = f"../data/{args.dataset}-ldr/train/{args.num_labelled_samples}_{args.num_unlabelled_samples}_{args.max_permutations}_{run_idx}.json"
                    print(f"LDR mode: Loading dataset from {data_cache_path}")
            else:
                # Regular mode: use {num_train_examples}_{run_idx}.json format
                data_cache_path = f"../data/{args.dataset}/train/{args.num_train_examples}_{run_idx}.json"
                print(f"Regular mode: Loading dataset from {data_cache_path}")
            
            if not os.path.exists(data_cache_path):
                print(f"Dataset not found at {data_cache_path}")
                if args.ldr_mode:
                    if training_variant == 'a2t' and args.label_type == 'ground_truth':
                        print("Please create the original dataset first using create_training_datasets.py")
                    else:
                        print("Please create the LDR dataset first using create_ldr_training_datasets.py")
                else:
                    print("Please create the dataset first using create_training_datasets.py or eval_icl.py")
                continue
            
            loaded_datasets = json.load(open(data_cache_path, "r"))
            
            # Create base datasets
            if args.shuffle_demos:
                train_dataset = ICLDatasetShuffled(
                    loaded_datasets['train']['examples'],
                    tokenizer,
                    icl_demos=loaded_datasets['train']['icl_demos'],
                    num_shuffles=args.num_shuffles,
                    seed=run_idx,
                    parse_answer_func=parse_answer_func,
                    num_generated_tokens=args.num_generated_tokens
                )
                dev_dataset = ICLDatasetShuffled(
                    loaded_datasets['dev']['examples'],
                    tokenizer,
                    icl_demos=loaded_datasets['dev']['icl_demos'],
                    num_shuffles=args.num_shuffles,
                    seed=run_idx,
                    parse_answer_func=parse_answer_func,
                    num_generated_tokens=args.num_generated_tokens
                )
            else:
                train_dataset = ICLDataset(
                    loaded_datasets['train']['examples'],
                    tokenizer,
                    icl_demos=loaded_datasets['train']['icl_demos'],
                    parse_answer_func=parse_answer_func,
                    num_generated_tokens=args.num_generated_tokens
                )
                dev_dataset = ICLDataset(
                    loaded_datasets['dev']['examples'],
                    tokenizer,
                    icl_demos=loaded_datasets['dev']['icl_demos'],
                    parse_answer_func=parse_answer_func,
                    num_generated_tokens=args.num_generated_tokens
                )
            
            # --- STEP 3: Collect ICL outputs/activations from base model ---
            
            if training_variant in ["tok", "a2t"]:
                if args.label_type == 'icl_outputs':
                    print("Collecting ICL outputs for token based training (using base model)...")
                    _, train_icl_outputs = collect_icl_outputs(base_model, train_dataset, tokenizer, device, args.num_generated_tokens)
                    _, dev_icl_outputs = collect_icl_outputs(base_model, dev_dataset, tokenizer, device, args.num_generated_tokens)
                    train_dataset = ICLDatasetWithOutputs(train_dataset, icl_outputs=train_icl_outputs, training_mode='icl_outputs')
                    dev_dataset = ICLDatasetWithOutputs(dev_dataset, icl_outputs=dev_icl_outputs, training_mode='icl_outputs')
                else:
                    train_dataset = ICLDatasetWithOutputs(train_dataset, training_mode='ground_truth')
                    dev_dataset = ICLDatasetWithOutputs(dev_dataset, training_mode='ground_truth')
            elif training_variant in ["tokl"]:
                print(f"Collecting ICL top-{args.tokl_top_k} distributions for soft token imitation (using base model)...")
                train_gen_tokens, train_dists = collect_icl_distributions(
                    base_model, train_dataset, tokenizer, device,
                    num_generated_tokens=args.num_generated_tokens, top_k=args.tokl_top_k
                )
                dev_gen_tokens, dev_dists = collect_icl_distributions(
                    base_model, dev_dataset, tokenizer, device,
                    num_generated_tokens=args.num_generated_tokens, top_k=args.tokl_top_k
                )
                train_dataset = ICLDatasetWithOutputs(train_dataset, icl_outputs=train_gen_tokens, training_mode='icl_outputs', icl_distributions=train_dists)
                dev_dataset = ICLDatasetWithOutputs(dev_dataset, icl_outputs=dev_gen_tokens, training_mode='icl_outputs', icl_distributions=dev_dists)
            elif training_variant in ["act", "tna", "t2a"]:
                print("Collecting target activations (using base model)...")
                train_activations = get_activations(
                    base_model, train_dataset, tokenizer, device, 
                    num_generated_tokens=args.num_generated_tokens
                )
                train_icl_outputs = []
                train_icl_activations = []
                for i in range(len(train_dataset)):
                    train_icl_outputs.append(train_activations['with_icl'][i]['generated_tokens'])
                    train_icl_activations.append(train_activations['with_icl'][i]['activations'])
                dev_activations = get_activations(
                    base_model, dev_dataset, tokenizer, device, 
                    num_generated_tokens=args.num_generated_tokens
                )
                dev_icl_outputs = []
                dev_icl_activations = []
                for i in range(len(dev_dataset)):
                    dev_icl_outputs.append(dev_activations['with_icl'][i]['generated_tokens'])
                    dev_icl_activations.append(dev_activations['with_icl'][i]['activations'])
                train_dataset = ICLDatasetWithOutputs(train_dataset, icl_outputs=train_icl_outputs, icl_activations=train_icl_activations)
                dev_dataset = ICLDatasetWithOutputs(dev_dataset, icl_outputs=dev_icl_outputs, icl_activations=dev_icl_activations)

                print(f"Collected activations for {len(train_dataset)} train and {len(dev_dataset)} dev examples.")
            
            # --- STEP 4: Now create/load the PEFT model for training ---
            print("Loading PEFT model for training...")
            model, is_continued = load_base_or_continue_model(args, device, torch_dtype, run_idx, base_model, base_method, training_variant, output_dir=args.output_dir)
            if not is_continued:
                if base_method in ["lora"] or args.training_method in ["tok", "tokl", "act", "tna", "a2t", "t2a"]:
                    # Existing LoRA methods (keep unchanged)
                    lora_config = create_lora_config(args.lora_type, args.lora_r, args.lora_alpha)
                    model = get_peft_model(model, lora_config)
                elif base_method == "ia3":
                    # IA3 method
                    ia3_config = create_ia3_config(args.ia3_type)
                    model = get_peft_model(model, ia3_config)
                elif base_method == "prompt":
                    # Prompt Tuning method
                    prompt_config = create_prompt_tuning_config(args.num_virtual_tokens)
                    model = get_peft_model(model, prompt_config)
                elif base_method == "prefix":
                    # Prefix Tuning method
                    prefix_config = create_prefix_tuning_config(args.num_virtual_tokens)
                    model = get_peft_model(model, prefix_config)
                
                model.print_trainable_parameters()

            # --- STEP 5: Training and saving logic remains unchanged ---
            # Generate model save path
            model_save_name = get_model_name(
                args.training_method, model_name, args.lora_type, args.lora_r, 
                args.lora_alpha, args.num_generated_tokens, args.num_train_examples, 
                args.lr, run_idx, args.label_type, args.ce_loss_weight, args.ia3_type, args.num_virtual_tokens,
                args.ldr_mode, args.num_labelled_samples, args.num_unlabelled_samples, args.max_permutations,
                args.tokl_top_k if 'tokl' in args.training_method else None
            )
            model_save_path = os.path.join(output_base_dir, model_save_name)
            
            # Create run info
            run_info = {
                'dataset': args.dataset,
                'training_method': args.training_method,
                'model_name': model_name,
                'lora_type': args.lora_type,
                'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'ia3_type': args.ia3_type,
                'num_virtual_tokens': args.num_virtual_tokens,
                'num_train_examples': args.num_train_examples,
                'num_generated_tokens': args.num_generated_tokens,
                'lr': args.lr,
                'ce_loss_weight': args.ce_loss_weight if args.training_method == 'tna' else None,
                'label_type': args.label_type,
                'run_idx': run_idx,
                'output_dir': output_base_dir,
                'ldr_mode': args.ldr_mode,
                'num_labelled_samples': args.num_labelled_samples if args.ldr_mode else None,
                'num_unlabelled_samples': args.num_unlabelled_samples if args.ldr_mode else None,
                'max_permutations': args.max_permutations if args.ldr_mode else None,
                'tokl_top_k': args.tokl_top_k if 'tokl' in args.training_method else None
            }

            # Train the model
            print(f"Starting {args.training_method} training...")
            model = train_unified_model(
                training_method=training_variant,
                model=model,
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                tokenizer=tokenizer,
                device=device,
                num_epochs=args.num_train_epochs,
                patience=args.patience,
                lr=args.lr,
                batch_size=args.batch_size,
                gradient_clip_val=args.gradient_clip_val,
                ce_loss_weight=args.ce_loss_weight,
                label_type=args.label_type,
                run_info=run_info,
                wandb_log=args.wandb_log,
                model_save_path=model_save_path,
                dev_eval_steps=args.dev_eval_steps,
                log_gradient_norms=args.log_gradient_norms,
                log_weight_norms=args.log_weight_norms,
                log_checkpoints=args.log_checkpoints,
                checkpoint_frequency=args.checkpoint_frequency
            )
            
            # Save the final model
            print(f"Saving model to {model_save_path}")
            Path(model_save_path).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_save_path)
            
            # Save training config for reproducibility
            config_path = os.path.join(model_save_path, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(run_info, f, indent=2)
            
            print(f"Training completed for run {run_idx + 1}")

            # model.unload()

            if 'model' in locals():
                del model
            if 'base_model' in locals():
                del base_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in run {run_idx + 1}: {e}")
            if wandb_run is not None:
                wandb_run.finish(exit_code=1)
            continue
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            if 'base_model' in locals():
                del base_model
            torch.cuda.empty_cache()
            if wandb_run is not None:
                wandb_run.finish()
    
    print(f"\nCompleted all {args.num_runs} runs for {args.training_method} training!")


if __name__ == "__main__":
    main() 