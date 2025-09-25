#!/usr/bin/env python3
"""
Optimized Batch Evaluation Script for Distillation Project

This script addresses performance bottlenecks by:
1. Loading base model once and keeping it in memory
2. Smart PEFT adapter management (load/unload only adapters)
3. Batched inference for multiple examples
4. Optimized evaluation sequence to minimize adapter switching
5. Single script architecture for both base and trained models

Key optimizations:
- Base model loaded once per GPU
- PEFT adapters loaded/unloaded as needed
- Batched processing for faster inference
- Grouped evaluations to minimize adapter switching
- Support for both single-token (with uncertainty) and multi-token generation
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, logging
from peft import PeftModel
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
from collections import defaultdict
from types import SimpleNamespace

from data import ICLDataset
from utils import parse_answer_gsm8k, parse_answer_sciqa, parse_answer_boxed, construct_results_path, construct_base_without_icl_path

logging.set_verbosity_error()


@dataclass
class EvaluationTask:
    """Represents a single evaluation task"""
    base_output_dir: str
    model_type: str
    model_id: str
    trained_dataset: Optional[str]
    eval_dataset_name: str
    icl_source_dataset: str
    icl_max_demos: int
    num_generated_tokens_eval: int
    run_idx: int
    gpu_id: int
    
    # Model-specific parameters
    lora_type: Optional[str] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    ia3_type: Optional[str] = None
    num_virtual_tokens: Optional[int] = None
    num_generated_tokens_train: Optional[int] = None
    num_train_examples: Optional[int] = None
    lr: Optional[float] = None
    ce_loss_weight: Optional[float] = None
    label_type: Optional[str] = None
    
    # LDR (Low Data Regime) parameters
    ldr_mode: bool = False
    num_labelled_samples: Optional[int] = None
    num_unlabelled_samples: Optional[int] = None
    max_permutations: Optional[int] = None
    
    # Evaluation parameters
    uncertainty_analysis: bool = False
    top_k: int = 10
    eval_with_icl: bool = False
    merge_lora: bool = True
    
    def get_adapter_key(self) -> str:
        """Get unique key for adapter identification"""
        if self.model_type == 'base':
            return 'base'
        
        # Parse model_type to extract base method and training variant
        if '-' in self.model_type:
            base_method, training_variant = self.model_type.split('-', 1)
        else:
            base_method = 'lora'
            training_variant = self.model_type
        
        # Add LDR suffix if in LDR mode
        ldr_suffix = ""
        if self.ldr_mode and self.num_labelled_samples is not None and self.num_unlabelled_samples is not None and self.max_permutations is not None:
            ldr_suffix = f"_ldr{self.num_labelled_samples}_{self.num_unlabelled_samples}_{self.max_permutations}"
        
        if base_method in ['lora'] or self.model_type in ['tok', 'act', 'tna', 'a2t', 't2a']:
            return f"{self.model_type}_{self.trained_dataset}_{self.lora_type}_{self.lora_r}_{self.lora_alpha}_{self.num_generated_tokens_train}_{self.num_train_examples}_{self.lr}_{self.run_idx}_{self.label_type}_{self.ce_loss_weight}{ldr_suffix}"
        elif base_method == 'ia3':
            return f"{self.model_type}_{self.trained_dataset}_{self.ia3_type}_{self.num_generated_tokens_train}_{self.num_train_examples}_{self.lr}_{self.run_idx}_{self.label_type}_{self.ce_loss_weight}{ldr_suffix}"
        elif base_method in ['prompt', 'prefix']:
            return f"{self.model_type}_{self.trained_dataset}_{self.num_virtual_tokens}_{self.num_generated_tokens_train}_{self.num_train_examples}_{self.lr}_{self.run_idx}_{self.label_type}_{self.ce_loss_weight}{ldr_suffix}"
        else:
            return f"{self.model_type}_{self.trained_dataset}_{self.run_idx}{ldr_suffix}"


class ModelManager:
    """Manages model loading and caching for efficient evaluation"""
    
    def __init__(self, model_id: str, device: torch.device, torch_dtype: torch.dtype):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.base_model = None
        self.current_adapter_key = None
        self.current_model = None
        
    def load_base_model(self):
        """Load base model once"""
        if self.base_model is not None:
            del self.base_model
            torch.cuda.empty_cache()
            
        print(f"Loading base model: {self.model_id}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        print("Base model loaded successfully")
        
        return self.base_model
    
    def get_model_for_task(self, task: EvaluationTask) -> torch.nn.Module:
        """Get model for specific task, loading adapter if needed"""
        if task.model_type == 'base':
            return self.load_base_model()

        # Check if we need to load a new adapter
        adapter_key = task.get_adapter_key()
        if adapter_key != self.current_adapter_key:
            # Load new adapter
            self._load_adapter(task, adapter_key)
        
        return self.current_model
    
    def _load_adapter(self, task: EvaluationTask, adapter_key: str):
        """Load PEFT adapter for the task"""
        print(f"Loading adapter: {adapter_key}")
        
        # Ensure base model is loaded
        base_model = self.load_base_model()
        
        # Construct adapter path
        adapter_path = self._construct_adapter_path(task)
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        # Load adapter
        self.current_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Optionally merge LoRA weights
        if '-' in task.model_type:
            base_method, training_variant = task.model_type.split('-', 1)
        else:
            base_method = 'lora'
            training_variant = task.model_type

        if task.merge_lora and base_method in ['lora', 'ia3']:
            print("Merging LoRA weights...")
            self.current_model = self.current_model.merge_and_unload()
        
        self.current_adapter_key = adapter_key
        print(f"Adapter loaded successfully: {adapter_path}")
    
    def _construct_adapter_path(self, task: EvaluationTask) -> str:
        """Construct adapter path based on task parameters"""
        model_name_base = task.model_id.split('/')[-1]
        
        # Parse model_type to extract base method and training variant
        if '-' in task.model_type:
            base_method, training_variant = task.model_type.split('-', 1)
        else:
            base_method = 'lora'
            training_variant = task.model_type

        ldr_suffix = ""
        if task.ldr_mode and task.num_labelled_samples is not None and task.num_unlabelled_samples is not None and task.max_permutations is not None:
            ldr_suffix = f"_ldr{task.num_labelled_samples}_{task.num_unlabelled_samples}_{task.max_permutations}"
        
        # Handle existing LoRA methods
        if task.model_type in ['tok', 'act', 'tna', 'a2t']:
            base_name = f"{task.base_output_dir}/{task.model_type}/{task.trained_dataset}/{model_name_base}_{task.lora_type}_{task.lora_r}_{task.lora_alpha}_{task.num_generated_tokens_train}_{task.num_train_examples}_{task.lr}_{task.run_idx}{ldr_suffix}"
            
            if task.model_type in ['tok', 'a2t']:
                return f"{base_name}_{task.label_type}"
            elif task.model_type in ['act', 't2a']:
                return base_name
            elif task.model_type == 'tna':
                return f"{base_name}_{task.ce_loss_weight}"
                
        # Handle new method variants
        elif base_method == 'ia3':
            base_name = f"{task.base_output_dir}/{task.model_type}/{task.trained_dataset}/{model_name_base}_ia3_{task.ia3_type}_{task.num_generated_tokens_train}_{task.num_train_examples}_{task.lr}_{task.run_idx}{ldr_suffix}"
            if training_variant in ['tok', 'a2t']:
                return f"{base_name}_{task.label_type}"
            elif training_variant == 'act':
                return base_name
            elif training_variant == 'tna':
                return f"{base_name}_{task.ce_loss_weight}"
            
        elif base_method in ['prompt', 'prefix']:
            base_name = f"{task.base_output_dir}/{task.model_type}/{task.trained_dataset}/{model_name_base}_{base_method}_{task.num_virtual_tokens}_{task.num_generated_tokens_train}_{task.num_train_examples}_{task.lr}_{task.run_idx}{ldr_suffix}"
            if training_variant in ['tok', 'a2t']:
                return f"{base_name}_{task.label_type}"
            elif training_variant == 'act':
                return base_name
            elif training_variant == 'tna':
                return f"{base_name}_{task.ce_loss_weight}"
            
        else:
            raise ValueError(f"Unknown model_type: {task.model_type}")


def load_dataset_label_info(eval_dataset_name: str) -> Tuple[Optional[List[str]], Optional[Dict]]:
    """Load label information from dataset's main.json file."""
    dataset_path = f"../data/{eval_dataset_name}/main.json"
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset main.json not found at {dataset_path}")
        return None, None
    
    try:
        with open(dataset_path, 'r') as f:
            dataset_info = json.load(f)['metadata']
        
        max_choices = dataset_info.get('max_choices', 0)
        remap_dict = dataset_info.get('remap_dict', None)
        
        # Generate label tokens
        if max_choices > 0:
            if remap_dict is not None:
                # Use remapped tokens
                label_tokens = [remap_dict[str(i)] for i in range(max_choices)]
            else:
                # Use numeric labels
                label_tokens = [str(i) for i in range(max_choices)]
        else:
            label_tokens = []
            
        return label_tokens, remap_dict
        
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return None, None


def get_true_batched_token_probabilities(model, batch_input_ids, batch_attention_mask, decode_positions, tokenizer, device, 
                                        label_tokens=None, top_k=10) -> List[Dict]:
    """Get probabilities for top-k tokens and label tokens for true batched single token generation."""
    model.eval()
    
    with torch.no_grad():
        # Forward pass with true batched inputs
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        results = []
        for i, decode_position in enumerate(decode_positions):
            # Get logits for the last token position for this example
            last_token_logits = logits[i, decode_position, :]  # Shape: [vocab_size]

            # Apply softmax to get probabilities
            probs = F.softmax(last_token_logits, dim=-1)
            
            # Get top-k tokens and their probabilities
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            top_k_tokens = [tokenizer.decode(idx.item()) for idx in top_k_indices]
            top_k_probs = top_k_probs.cpu().tolist()
            
            # Get probabilities for label tokens
            label_probs = {}
            if label_tokens:
                for label_token in label_tokens:
                    try:
                        # Handle both single tokens and multi-token labels
                        token_ids = tokenizer.encode(label_token, add_special_tokens=False)
                        if len(token_ids) == 1:
                            token_id = token_ids[0]
                            label_probs[label_token] = probs[token_id].item()
                        else:
                            # For multi-token labels, take the first token's probability
                            token_id = token_ids[0]
                            label_probs[label_token] = probs[token_id].item()
                    except:
                        label_probs[label_token] = 0.0
            
            results.append({
                'top_k_tokens': top_k_tokens,
                'top_k_probs': top_k_probs,
                'label_probs': label_probs
            })
        
        return results


def get_true_batched_transition_scores(model, batch_input_ids, batch_attention_mask, decode_positions, tokenizer, device, 
                                      num_generated_tokens, top_k=100) -> List[Dict]:
    """Get transition scores for multiple tokens using compute_transition_scores for true batched processing."""
    model.eval()
    
    with torch.no_grad():
        # Generate with output_scores=True to get transition scores for the entire batch
        generation_config = GenerationConfig(
            max_new_tokens=num_generated_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0,
            do_sample=False,
            return_dict_in_generate=True,
            stop_strings=['\n\n'],
            output_scores=True
        )
        
        # Generate tokens and get scores for the entire batch
        generated_outputs = model.generate(
            tokenizer=tokenizer,
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            generation_config=generation_config
        )

        # Get the generated sequences and scores
        sequences = generated_outputs.sequences  # Shape: [batch_size, seq_len]
        scores = generated_outputs.scores  # Tuple of tensors, one for each generated token
        
        results = []
        for i, decode_position in enumerate(decode_positions):
            # Get the generated tokens for this example (excluding the input tokens)
            input_length = decode_position + 1
            generated_tokens = sequences[i, input_length:]

            # For each generated token position, get top-k tokens and their probabilities
            token_matrices = []
            prob_matrices = []
            
            for token_idx in range(min(num_generated_tokens, len(scores))):
                # Get logits for this token position and example
                token_logits = scores[token_idx][i]  # Shape: [vocab_size]
                
                # Apply softmax to get probabilities
                probs = F.softmax(token_logits, dim=-1)
                
                # Get top-k tokens and their probabilities
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                top_k_tokens = [tokenizer.decode(idx.item()) for idx in top_k_indices]
                top_k_probs = top_k_probs.cpu().tolist()
                
                token_matrices.append(top_k_tokens)
                prob_matrices.append(top_k_probs)
            
            # Decode the full generated sequence
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            results.append({
                'generated_text': generated_text,
                'token_matrix': token_matrices,  # Shape: [num_generated_tokens, top_k]
                'prob_matrix': prob_matrices,    # Shape: [num_generated_tokens, top_k]
            })
        
        return results


def evaluate_model_batched(model, dataset, tokenizer, device, num_generated_tokens=1, 
                          eval_with_icl=False, eval_without_icl=False, parse_answer_func=None, uncertainty_analysis=False, 
                          label_tokens=None, top_k=10, batch_size=8) -> Dict[str, List]:
    """Evaluate model performance with true batched processing."""
    model.eval()
    
    results = {
        'with_icl': [],
        'without_icl': [],
    }
    
    generation_config = GenerationConfig(
        max_new_tokens=num_generated_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.0,
        do_sample=False,
        stop_strings=['\n\n'],
    )

    # Process dataset in batches
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating model"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_examples = dataset[batch_start:batch_end]
        
        # Evaluate WITHOUT ICL
        if eval_without_icl and batch_examples[0].get('inputs_no_icl') is not None:
            try:
                # Prepare batch inputs for true batched inference
                batch_input_ids = []
                batch_attention_mask = []
                decode_positions = []
                queries = []
                full_answer_texts = []
                true_labels = []
                
                for example in batch_examples:
                    inputs_no_icl = {k: v.to(device) for k, v in example['inputs_no_icl'].items()}
                    decode_position = example['no_icl_decode_position']
                    
                    # Truncate to decode position + 1 for single token generation
                    input_ids = inputs_no_icl['input_ids'][:, :decode_position + 1]
                    attention_mask = inputs_no_icl['attention_mask'][:, :decode_position + 1]
                    
                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    decode_positions.append(decode_position)
                    
                    query = example['example']['question']
                    full_answer_text = example['example']['answer']
                    queries.append(query)
                    full_answer_texts.append(full_answer_text)
                    
                    if 'eval_answer' in example:
                        true_label = example['eval_answer']
                    else:
                        true_label = full_answer_text
                    true_labels.append(true_label)

                # Pad sequences to same length for true batched inference
                max_length = max(input_ids.shape[1] for input_ids in batch_input_ids)
                padded_input_ids = []
                padded_attention_mask = []
                padded_decode_positions = []
                
                for input_ids, attention_mask, decode_position in zip(batch_input_ids, batch_attention_mask, decode_positions):
                    pad_length = max_length - input_ids.shape[1]
                    if pad_length > 0:
                        padded_input_ids.append(torch.cat([
                            torch.full((1, pad_length), tokenizer.pad_token_id, device=device, dtype=input_ids.dtype),
                            input_ids
                        ], dim=1))
                        padded_attention_mask.append(torch.cat([
                            torch.zeros((1, pad_length), device=device, dtype=attention_mask.dtype),
                            attention_mask
                        ], dim=1))
                    else:
                        padded_input_ids.append(input_ids)
                        padded_attention_mask.append(attention_mask)
                
                # Stack into true batch tensors
                batch_input_ids = torch.cat(padded_input_ids, dim=0)  # Shape: [batch_size, seq_len]
                batch_attention_mask = torch.cat(padded_attention_mask, dim=0)  # Shape: [batch_size, seq_len]
                decode_positions = torch.full((len(batch_examples),), max_length - 1, device=device, dtype=torch.long)

                if uncertainty_analysis:
                    if num_generated_tokens == 1:
                        # True batched probability analysis for single token generation
                        prob_infos = get_true_batched_token_probabilities(
                            model, batch_input_ids, batch_attention_mask, decode_positions, tokenizer, device, 
                            label_tokens=label_tokens, top_k=top_k
                        )
                        
                        for i, prob_info in enumerate(prob_infos):
                            generated_answer = prob_info['top_k_tokens'][0].strip() if prob_info['top_k_tokens'] else ""
                            generated_text = generated_answer
                            
                            results['without_icl'].append({
                                'query': queries[i],
                                'full_answer_text': full_answer_texts[i],
                                'parsed_answer': true_labels[i],
                                'full_generated_text': generated_text,
                                'generated_answer': generated_answer,
                                'prob_info': prob_info
                            })
                    else:
                        # True batched transition scores for multi-token generation
                        prob_infos = get_true_batched_transition_scores(
                            model, batch_input_ids, batch_attention_mask, decode_positions, tokenizer, device, 
                            num_generated_tokens, top_k=top_k
                        )
                        
                        for i, prob_info in enumerate(prob_infos):
                            generated_text = prob_info['generated_text']
                            generated_answer = parse_answer_func(generated_text) if parse_answer_func else generated_text.strip()
                            
                            results['without_icl'].append({
                                'query': queries[i],
                                'full_answer_text': full_answer_texts[i],
                                'parsed_answer': true_labels[i],
                                'full_generated_text': generated_text,
                                'generated_answer': generated_answer,
                                'prob_info': prob_info
                            })
                else:
                    # True batched generation
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            tokenizer=tokenizer,
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            generation_config=generation_config
                        )

                    # Process each example in the batch
                    for i, decode_position in enumerate(decode_positions):
                        new_tokens = generated_tokens[i, decode_position + 1:]
                        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        generated_answer = parse_answer_func(generated_text) if parse_answer_func else generated_text.strip()

                        results['without_icl'].append({
                            'query': queries[i],
                            'full_answer_text': full_answer_texts[i],
                            'parsed_answer': true_labels[i],
                            'full_generated_text': generated_text,
                            'generated_answer': generated_answer,
                            'prob_info': None
                        })
                
            except Exception as e:
                print(f"Error in without_icl evaluation: {e}")
                raise

        # Evaluate WITH ICL
        if eval_with_icl and batch_examples[0].get('inputs_with_icl') is not None:
            try:
                # Prepare batch inputs for true batched inference
                batch_input_ids = []
                batch_attention_mask = []
                decode_positions = []
                queries = []
                full_answer_texts = []
                true_labels = []
                
                for example in batch_examples:
                    inputs_with_icl = {k: v.to(device) for k, v in example['inputs_with_icl'].items()}
                    decode_position = example['with_icl_decode_position']
                    
                    # Truncate to decode position + 1 for single token generation
                    input_ids = inputs_with_icl['input_ids'][:, :decode_position + 1]
                    attention_mask = inputs_with_icl['attention_mask'][:, :decode_position + 1]
                    
                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    decode_positions.append(decode_position)
                    
                    query = example['example']['question']
                    full_answer_text = example['example']['answer']
                    queries.append(query)
                    full_answer_texts.append(full_answer_text)
                    
                    if 'eval_answer' in example:
                        true_label = example['eval_answer']
                    else:
                        true_label = full_answer_text
                    true_labels.append(true_label)

                # Pad sequences to same length for true batched inference
                max_length = max(input_ids.shape[1] for input_ids in batch_input_ids)
                padded_input_ids = []
                padded_attention_mask = []
                
                for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
                    pad_length = max_length - input_ids.shape[1]
                    if pad_length > 0:
                        padded_input_ids.append(torch.cat([
                            torch.full((1, pad_length), tokenizer.pad_token_id, device=device, dtype=input_ids.dtype),
                            input_ids
                        ], dim=1))
                        padded_attention_mask.append(torch.cat([
                            torch.zeros((1, pad_length), device=device, dtype=attention_mask.dtype),
                            attention_mask
                        ], dim=1))
                    else:
                        padded_input_ids.append(input_ids)
                        padded_attention_mask.append(attention_mask)
                
                # Stack into true batch tensors
                batch_input_ids = torch.cat(padded_input_ids, dim=0)  # Shape: [batch_size, seq_len]
                batch_attention_mask = torch.cat(padded_attention_mask, dim=0)  # Shape: [batch_size, seq_len]
                decode_positions = torch.full((len(batch_examples),), max_length - 1, device=device, dtype=torch.long)

                if uncertainty_analysis:
                    if num_generated_tokens == 1:
                        # True batched probability analysis for single token generation
                        prob_infos = get_true_batched_token_probabilities(
                            model, batch_input_ids, batch_attention_mask, decode_positions, tokenizer, device, 
                            label_tokens=label_tokens, top_k=top_k
                        )
                        
                        for i, prob_info in enumerate(prob_infos):
                            generated_answer = prob_info['top_k_tokens'][0].strip() if prob_info['top_k_tokens'] else ""
                            generated_text = generated_answer
                            
                            results['with_icl'].append({
                                'query': queries[i],
                                'full_answer_text': full_answer_texts[i],
                                'parsed_answer': true_labels[i],
                                'full_generated_text': generated_text,
                                'generated_answer': generated_answer,
                                'prob_info': prob_info
                            })
                    else:
                        # True batched transition scores for multi-token generation
                        prob_infos = get_true_batched_transition_scores(
                            model, batch_input_ids, batch_attention_mask, decode_positions, tokenizer, device, 
                            num_generated_tokens, top_k=top_k
                        )
                        
                        for i, prob_info in enumerate(prob_infos):
                            generated_text = prob_info['generated_text']
                            generated_answer = parse_answer_func(generated_text) if parse_answer_func else generated_text.strip()
                            
                            results['with_icl'].append({
                                'query': queries[i],
                                'full_answer_text': full_answer_texts[i],
                                'parsed_answer': true_labels[i],
                                'full_generated_text': generated_text,
                                'generated_answer': generated_answer,
                                'prob_info': prob_info
                            })
                else:
                    # True batched generation
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            tokenizer=tokenizer,
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            generation_config=generation_config
                        )

                    # Process each example in the batch
                    for i, decode_position in enumerate(decode_positions):
                        new_tokens = generated_tokens[i, decode_position + 1:]
                        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        generated_answer = parse_answer_func(generated_text) if parse_answer_func else generated_text.strip()

                        results['with_icl'].append({
                            'query': queries[i],
                            'full_answer_text': full_answer_texts[i],
                            'parsed_answer': true_labels[i],
                            'full_generated_text': generated_text,
                            'generated_answer': generated_answer,
                            'prob_info': None
                        })
                
            except Exception as e:
                print(f"Error in with_icl evaluation: {e}")
                raise

    return results


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


def create_or_load_evaluation_dataset(eval_dataset_name, icl_source_dataset, run_idx, num_training_examples, max_demos, tokenizer, num_generated_tokens, force_rebuild=False, parse_answer_func=None):
    """
    Creates or loads a cached evaluation dataset with paired ICL demos.
    """
    # Create directory structure
    dataset_dir = f"../data"
    source_dir = f"{dataset_dir}/{icl_source_dataset}/train"
    eval_dir = f"{dataset_dir}/{eval_dataset_name}/val"
    # Note: Cache filename depends on parameters used to *create* it
    cache_dir = f"{eval_dir}/{icl_source_dataset}"
    cache_path = f"{cache_dir}/{num_training_examples}_{run_idx}.json"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if cached dataset exists
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached evaluation dataset from {cache_path}")
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        eval_data = cached_data['examples']
        icl_prompts = cached_data['icl_demos']
        print(f"Loaded {len(eval_data)} examples and {len(icl_prompts)} ICL prompt sets from cache")
        # Verify cache parameters match expectations
        cached_config = cached_data.get('config', {})
        if cached_config.get('eval_dataset') != eval_dataset_name or \
           cached_config.get('icl_source') != icl_source_dataset or \
           cached_config.get('icl_max_demos') != max_demos or \
           cached_config.get('seed') != run_idx:
            print("Warning: Cached dataset configuration mismatch. Consider using --force_rebuild_dataset if parameters have changed.")
            print(f"Expected: eval={eval_dataset_name}, source={icl_source_dataset}, max_demos={max_demos}, seed={run_idx}")
            print(f"Found: {cached_config}")
    else:
        print(f"Creating new evaluation dataset cache at {cache_path}")
        print(f"Using ICL demos from {icl_source_dataset}")
        # Load source data file
        source_data_file_path = f"{dataset_dir}/{icl_source_dataset}/main.json"
        val_data_file_path = f"{dataset_dir}/{eval_dataset_name}/main.json"
        with open(val_data_file_path, "r") as f:
            data = json.load(f)
        eval_data = data['val_examples']
        # Load ICL source dataset
        icl_file_path = f"{source_dir}/{num_training_examples}_{run_idx}.json"
        with open(icl_file_path, "r") as f:
            icl_data = json.load(f)
        available_indices = icl_data['remaining_indices']
        with open(source_data_file_path, "r") as f:
            source_data = json.load(f)
        available_demos = [source_data['train_examples'][i] for i in available_indices]
        if len(available_demos) < max_demos:
            print(f"Warning: Only {len(available_demos)} examples available for ICL demos (requested {max_demos}). Using available count.")
            current_max_demos = len(available_demos)
        else:
            current_max_demos = max_demos
        # Set random seed for reproducibility
        rng = np.random.RandomState(run_idx) # Use the provided run_idx as seed
        # Generate ICL prompts for each evaluation example
        icl_prompts = []
        for _ in eval_data:
            available_indices = list(range(len(available_demos)))
            if len(available_indices) >= current_max_demos:
                demo_indices_relative = rng.choice(available_indices, current_max_demos, replace=False)
            else: # Should not happen due to check above, but safer
                demo_indices_relative = available_indices
            # Create prompts for sampled demos
            example_prompts = []
            for relative_idx in demo_indices_relative:
                demo = available_demos[relative_idx]['question'] + str(available_demos[relative_idx]['answer'])
                example_prompts.append(demo)
            icl_prompts.append(example_prompts)
        print(f"Generated {len(icl_prompts)} ICL prompt sets with {current_max_demos} demos each (seed={run_idx})")
        # Cache the dataset
        cache_data = {
            'examples': eval_data,
            'icl_demos': icl_prompts,
            'config': {
                'eval_dataset': eval_dataset_name,
                'icl_source': icl_source_dataset,
                'icl_max_demos': current_max_demos, # Use the actual max_demos used
                'seed': run_idx # Use the actual seed used
            }
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cached evaluation dataset to {cache_path}")
    # Create the paired dataset for evaluation using the loaded/generated data
    eval_dataset = ICLDataset(
        eval_data,
        tokenizer,
        icl_demos=icl_prompts, # Pass the list of prompt lists
        num_generated_tokens=num_generated_tokens,
        parse_answer_func=parse_answer_func
    )
    return eval_dataset, cache_path


def load_base_without_icl_results(args):
    """Load base model without_icl results from separate file."""
    results_dir, results_filename = construct_base_without_icl_path(args)
    results_path = os.path.join(results_dir, results_filename + ".json")
    
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded base model without_icl results from: {results_path}")
            return data.get('results', {}).get('without_icl', []), data.get('metrics', {})
        except Exception as e:
            print(f"Error loading base model without_icl results: {e}")
            return None, None
    else:
        print(f"Base model without_icl results not found at: {results_path}")
        return None, None


def save_base_without_icl_results(args, without_icl_results, without_icl_metrics):
    """Save base model without_icl results to separate file."""
    results_dir, results_filename = construct_base_without_icl_path(args)
    results_path = os.path.join(results_dir, results_filename + ".json")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Structure saved data
    eval_specific_args = ['eval_dataset_name', 'icl_source_dataset', 'icl_max_demos', 
                         'num_generated_tokens_eval', 'uncertainty_analysis', 'top_k', 
                         'eval_with_icl', 'merge_lora', 'gpu']
    training_params = {k: v for k, v in vars(args).items() if k not in eval_specific_args}
    evaluation_config = {k: v for k, v in vars(args).items() if k in eval_specific_args}
    
    output_data = {
        'trained_model_params': training_params,
        'evaluation_config': evaluation_config,
        'metrics': without_icl_metrics,
        'results': {'without_icl': without_icl_results}
    }
    
    try:
        with open(results_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved base model without_icl results to: {results_path}")
        return True
    except Exception as e:
        print(f"Error saving base model without_icl results to {results_path}: {e}")
        return False


def find_trained_models(base_output_dir, model_type, trained_dataset, filters):
    """Find all trained models that match the specified criteria"""
    models_found = []
    
    if model_type == 'base':
        # No trained models to find for base model - return single entry
        for num_examples in filters['num_examples']:
            for run_idx in filters['run_indices']:
                models_found.append({
                    'model_type': 'base',
                    'num_tokens': filters['num_tokens'],
                    'num_examples': num_examples,
                    'run_idx': run_idx
                })
        return models_found
    
    # Directory and paring logic for each model type
    search_dir = os.path.join(base_output_dir, model_type, trained_dataset)

    if not os.path.exists(search_dir):
        print(f"Model directory not found: {search_dir}")
        return []

    # Get all model directories
    for item in os.listdir(search_dir):
        item_path = os.path.join(search_dir, item)
        if not os.path.isdir(item_path):
            continue
        # Check if it has the PEFT adapter files
        adapter_file = os.path.join(item_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            continue
        # Parse the model name to extract parameters
        try:
            parts = item.split('_')
            if model_type in ['tok', 'a2t']:
                # Format: {model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}_{label_type}[_ldrN_M_K]
                if len(parts) < 10:
                    continue
                if filters.get('ldr_mode') and len(parts) < 13:
                    continue
                model_name = parts[0]
                lora_type = parts[1]
                lora_r = int(parts[2])
                lora_alpha = int(parts[3])
                num_tokens = int(parts[4])
                num_examples = int(parts[5])
                lr = float(parts[6])
                run_idx = int(parts[7])
                
                # Check for LDR suffix
                ldr_suffix = None
                if len(parts) == 13 and parts[8].startswith('ldr'):
                    num_unlabelled = int(parts[9])
                    max_perm = int(parts[10])
                    if filters.get('ldr_mode') and filters['ldr_mode']:
                        if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                            continue
                        if (filters.get('max_permutations') and max_perm not in filters['max_permutations']):
                            continue
                    else:
                        continue # Only used in LDR mode
                    ldr_suffix = f"{num_unlabelled}_{max_perm}"
                    label_type = '_'.join(parts[11:])
                else:
                    label_type = "_".join(parts[8:])

                # Check filters
                if (filters.get('model_name') and model_name not in filters['model_name']):
                    continue
                if (filters.get('lora_types') and lora_type not in filters['lora_types']):
                    continue
                if (filters.get('lora_rs') and lora_r not in filters['lora_rs']):
                    continue
                if (filters.get('lora_alphas') and lora_alpha not in filters['lora_alphas']):
                    continue
                if (filters.get('num_tokens') and num_tokens != filters['num_tokens']):
                    continue
                if (filters.get('num_examples') and num_examples not in filters['num_examples']):
                    continue
                if (filters.get('lrs') and lr not in filters['lrs']):
                    continue
                if (filters.get('run_indices') and run_idx not in filters['run_indices']):
                    continue
                if (filters.get('label_types') and label_type not in filters['label_types']):
                    continue
                
                models_found.append({
                    'model_name': model_name,
                    'lora_type': lora_type,
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'num_tokens_train': num_tokens,
                    'num_examples_train': num_examples,
                    'lr': lr,
                    'run_idx': run_idx,
                    'label_type': label_type,
                    'ce_loss_weight': None,
                    'ldr_suffix': ldr_suffix
                })
            elif model_type in ['act', 't2a']:
                # Format: {model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}[_ldrN_M_K]
                if len(parts) < 8:
                    continue
                if filters.get('ldr_mode') and len(parts) < 11:
                    continue
                model_name = parts[0]
                lora_type = parts[1]
                lora_r = int(parts[2])
                lora_alpha = int(parts[3])
                num_tokens = int(parts[4])
                num_examples = int(parts[5])
                lr = float(parts[6])
                run_idx = int(parts[7])

                # Check for LDR
                ldr_suffix = None
                if len(parts) == 11 and parts[8].startswith('ldr'):
                    num_unlabelled = int(parts[9])
                    max_perm = int(parts[10])
                    if filters.get('ldr_mode') and filters['ldr_mode']:
                        if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                            continue
                        if (filters.get('max_permutations') and max_perm not in filters['max_permutations']):
                            continue
                    else:
                        continue # Only used in LDR mode
                    ldr_suffix = f"{num_unlabelled}_{max_perm}"

                # Check filters
                if (filters.get('model_name') and model_name not in filters['model_name']):
                    continue
                if (filters.get('lora_types') and lora_type not in filters['lora_types']):
                    continue
                if (filters.get('lora_rs') and lora_r not in filters['lora_rs']):
                    continue
                if (filters.get('lora_alphas') and lora_alpha not in filters['lora_alphas']):
                    continue
                if (filters.get('num_tokens') and num_tokens != filters['num_tokens']):
                    continue
                if (filters.get('num_examples') and num_examples not in filters['num_examples']):
                    continue
                if (filters.get('lrs') and lr not in filters['lrs']):
                    continue
                if (filters.get('run_indices') and run_idx not in filters['run_indices']):
                    continue

                models_found.append({
                    'model_name': model_name,
                    'lora_type': lora_type,
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'num_tokens_train': num_tokens,
                    'num_examples_train': num_examples,
                    'lr': lr,
                    'run_idx': run_idx,
                    'label_type': None,
                    'ce_loss_weight': None,
                    'ldr_suffix': ldr_suffix
                })
            elif model_type == 'tna':
                # Format: {model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}_{ce_weight}[_ldrN_M_K]
                if len(parts) < 9:
                    continue
                if filters.get('ldr_mode') and len(parts) < 12:
                    continue
                model_name = parts[0]
                lora_type = parts[1]
                lora_r = int(parts[2])
                lora_alpha = int(parts[3])
                num_tokens = int(parts[4])
                num_examples = int(parts[5])
                lr = float(parts[6])
                run_idx = int(parts[7])

                # Check for LDR
                ldr_suffix = None
                if len(parts) == 12 and parts[8].startswith('ldr'):
                    num_unlabelled = int(parts[9])
                    max_perm = int(parts[10])
                    if filters.get('ldr_mode') and filters['ldr_mode']:
                        if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                            continue
                        if (filters.get('max_permutations') and max_perm not in filters['max_permutations']):
                            continue
                    else:
                        continue # Only used in LDR mode
                    ldr_suffix = f"{num_unlabelled}_{max_perm}"
                    ce_loss_weight = float(parts[11])
                else:
                    ce_loss_weight = float(parts[8])
                
                # Check filters
                if (filters.get('model_name') and model_name not in filters['model_name']):
                    continue
                if (filters.get('lora_types') and lora_type not in filters['lora_types']):
                    continue
                if (filters.get('lora_rs') and lora_r not in filters['lora_rs']):
                    continue
                if (filters.get('lora_alphas') and lora_alpha not in filters['lora_alphas']):
                    continue
                if (filters.get('num_tokens') and num_tokens != filters['num_tokens']):
                    continue
                if (filters.get('num_examples') and num_examples not in filters['num_examples']):
                    continue
                if (filters.get('lrs') and lr not in filters['lrs']):
                    continue
                if (filters.get('run_indices') and run_idx not in filters['run_indices']):
                    continue
                if (filters.get('ce_loss_weights') and ce_loss_weight not in filters['ce_loss_weights']):
                    continue
                
                models_found.append({
                    'model_name': model_name,
                    'lora_type': lora_type,
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'num_tokens_train': num_tokens,
                    'num_examples_train': num_examples,
                    'lr': lr,
                    'run_idx': run_idx,
                    'label_type': None,
                    'ce_loss_weight': ce_loss_weight,
                    'ldr_suffix': ldr_suffix
                })
            elif '-' in model_type:
                # Parse new method variants
                base_method, training_variant = model_type.split('-')

                if base_method == 'ia3':
                    # Format: {model}_ia3_{ia3_type}_{tokens}_{examples}_{lr}_{run}_{suffix}
                    if len(parts) < 7:
                        continue
                    model_name = parts[0]
                    ia3_type = parts[2]  # parts[1] is "ia3"
                    num_tokens = int(parts[3])
                    num_examples = int(parts[4])
                    lr = float(parts[5])
                    run_idx = int(parts[6])
                    
                    # Determine suffix based on training variant
                    if training_variant in ['tok', 'a2t']:
                        if filters.get('ldr_mode') and filters['ldr_mode']:
                            if len(parts) < 12:
                                continue
                            num_unlabelled = int(parts[8])
                            max_perm = int(parts[9])
                            if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                                continue
                            if (filters.get('max_permutations') and max_perm not in filters['max_permutations']):
                                continue
                            ldr_suffix = f"{num_unlabelled}_{max_perm}"
                            label_type = "_".join(parts[10:])
                        else:
                            if len(parts) > 7 and 'ldr' in parts[7]:
                                continue # Only used in LDR mode
                            ldr_suffix = None
                            label_type = "_".join(parts[7:])
                        
                        ce_loss_weight = None
                    elif training_variant == 'act':
                        if filters.get('ldr_mode') and filters['ldr_mode']:
                            if len(parts) < 10:
                                continue
                            num_unlabelled = int(parts[8])
                            max_perm = int(parts[9])
                            if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                                continue
                            if (filters.get('max_permutations') and max_perm not in filters['max_permutations']):
                                continue
                            ldr_suffix = f"{num_unlabelled}_{max_perm}"
                            label_type = None
                        else:
                            if len(parts) > 7 and 'ldr' in parts[7]:
                                continue # Only used in LDR mode
                            ldr_suffix = None
                            label_type = None
                        ce_loss_weight = None
                    elif training_variant == 'tna':
                        if filters.get('ldr_mode') and filters['ldr_mode']:
                            if len(parts) < 11:
                                continue
                            num_unlabelled = int(parts[8])
                            max_perm = int(parts[9])
                            if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                                continue
                            ce_loss_weight = float(parts[10])
                            ldr_suffix = f"{num_unlabelled}_{max_perm}"
                        else:
                            if len(parts) > 7 and 'ldr' in parts[7]:
                                continue # Only used in LDR mode
                            ce_loss_weight = float(parts[7])
                            ldr_suffix = None
                        label_type = None
                    else:
                        continue

                    # Check filters
                    if (filters.get('model_name') and model_name not in filters['model_name']):
                        continue
                    if (filters.get('ia3_types') and ia3_type not in filters['ia3_types']):
                        continue
                    if (filters.get('num_tokens') and num_tokens != filters['num_tokens']):
                        continue
                    if (filters.get('num_examples') and num_examples not in filters['num_examples']):
                        continue
                    if (filters.get('lrs') and lr not in filters['lrs']):
                        continue
                    if (filters.get('run_indices') and run_idx not in filters['run_indices']):
                        continue
                    if training_variant in ['tok', 'a2t'] and (filters.get('label_types') and label_type not in filters['label_types']):
                        continue
                    if training_variant == 'tna' and (filters.get('ce_loss_weights') and ce_loss_weight not in filters['ce_loss_weights']):
                        continue

                    
                    
                    models_found.append({
                        'model_name': model_name,
                        'ia3_type': ia3_type,
                        'lora_type': None,
                        'lora_r': None,
                        'lora_alpha': None,
                        'num_tokens_train': num_tokens,
                        'num_examples_train': num_examples,
                        'lr': lr,
                        'run_idx': run_idx,
                        'label_type': label_type,
                        'ce_loss_weight': ce_loss_weight,
                        'ldr_suffix': ldr_suffix
                    })
                    
                elif base_method in ['prompt', 'prefix']:
                    # Format: {model}_{method}_{virtual_tokens}_{tokens}_{examples}_{lr}_{run}_{suffix}
                    if len(parts) < 7:
                        continue
                    model_name = parts[0]
                    method = parts[1]  # "prompt" or "prefix"
                    num_virtual_tokens = int(parts[2])
                    num_tokens = int(parts[3])
                    num_examples = int(parts[4])
                    lr = float(parts[5])
                    run_idx = int(parts[6])
                    
                    # Determine suffix based on training variant
                    if training_variant in ['tok', 'a2t']:
                        if filters.get('ldr_mode') and filters['ldr_mode']:
                            if len(parts) < 12:
                                continue
                            num_unlabelled = int(parts[8])
                            max_perm = int(parts[9])
                            if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                                continue
                            if (filters.get('max_permutations') and max_perm not in filters['max_permutations']):
                                continue
                            ldr_suffix = f"{num_unlabelled}_{max_perm}"
                            label_type = "_".join(parts[10:])
                        else:
                            ldr_suffix = None
                            label_type = "_".join(parts[7:])
                        
                        ce_loss_weight = None
                    elif training_variant == 'act':
                        if filters.get('ldr_mode') and filters['ldr_mode']:
                            if len(parts) < 10:
                                continue
                            num_unlabelled = int(parts[8])
                            max_perm = int(parts[9])
                            if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                                continue
                            if (filters.get('max_permutations') and max_perm not in filters['max_permutations']):
                                continue
                            ldr_suffix = f"{num_unlabelled}_{max_perm}"
                            label_type = None
                        else:
                            ldr_suffix = None
                            label_type = None
                        ce_loss_weight = None
                    elif training_variant == 'tna':
                        if filters.get('ldr_mode') and filters['ldr_mode']:
                            if len(parts) < 11:
                                continue
                            num_unlabelled = int(parts[8])
                            max_perm = int(parts[9])
                            if (filters.get('num_unlabelled_samples') and num_unlabelled not in filters['num_unlabelled_samples']):
                                continue
                            ce_loss_weight = float(parts[10])
                            ldr_suffix = f"{num_unlabelled}_{max_perm}"
                        else:
                            ce_loss_weight = float(parts[7])
                            ldr_suffix = None
                        label_type = None
                    else:
                        continue
                    
                    # Check filters
                    if (filters.get('model_name') and model_name not in filters['model_name']):
                        continue
                    if base_method != method:
                        continue
                    if (filters.get('num_virtual_tokens_list') and num_virtual_tokens not in filters['num_virtual_tokens_list']):
                        continue
                    if (filters.get('num_tokens') and num_tokens != filters['num_tokens']):
                        continue
                    if (filters.get('num_examples') and num_examples not in filters['num_examples']):
                        continue
                    if (filters.get('lrs') and lr not in filters['lrs']):
                        continue
                    if (filters.get('run_indices') and run_idx not in filters['run_indices']):
                        continue
                    if training_variant in ['tok', 'a2t'] and (filters.get('label_types') and label_type not in filters['label_types']):
                        continue
                    if training_variant == 'tna' and (filters.get('ce_loss_weights') and ce_loss_weight not in filters['ce_loss_weights']):
                        continue
                    
                    models_found.append({
                        'model_name': model_name,
                        'method': method,
                        'num_virtual_tokens': num_virtual_tokens,
                        'lora_type': None,
                        'lora_r': None,
                        'lora_alpha': None,
                        'num_tokens_train': num_tokens,
                        'num_examples_train': num_examples,
                        'lr': lr,
                        'run_idx': run_idx,
                        'label_type': label_type,
                        'ce_loss_weight': ce_loss_weight,
                        'ldr_suffix': ldr_suffix
                    })
        except (ValueError, IndexError) as e:
            print(f"Error parsing model name '{item}': {e}")
            continue

    print(f"Found {len(models_found)} trained models for {model_type} in {trained_dataset}")
    return models_found


def generate_evaluation_tasks(args) -> List[EvaluationTask]:
    """Generate all evaluation tasks based on arguments"""
    tasks = []
    
    # Build filter dictionary
    model_name = args.model_id.split('/')[-1]
    filters = {
        'model_name': [model_name],
        'lora_types': args.lora_types,
        'lora_rs': args.lora_rs,
        'lora_alphas': args.lora_alphas,
        'ia3_types': args.ia3_types,
        'num_virtual_tokens_list': args.num_virtual_tokens_list,
        'num_tokens': args.num_tokens,
        'num_gen_tokens_eval': args.num_gen_tokens_eval,
        'num_examples': args.num_examples,
        'lrs': args.lrs,
        'run_indices': args.run_indices,
        'label_types': args.label_types,
        'ce_loss_weights': args.ce_loss_weights,
        'ldr_mode': args.ldr_mode,
        'num_labelled_samples': args.num_labelled_samples if args.ldr_mode else None,
        'num_unlabelled_samples': args.num_unlabelled_samples if args.ldr_mode else None,
        'max_permutations': args.max_permutations if args.ldr_mode else None
    }
    
    # Find all models to evaluate
    all_models = []
    
    for model_type in args.model_types:
        if model_type == 'base':
            # Base model doesn't need trained dataset specification
            models = find_trained_models(args.base_output_dir, model_type, None, filters)
            all_models.extend([(model_type, None, models)])
        else:
            for trained_dataset in args.trained_datasets:
                models = find_trained_models(args.base_output_dir, model_type, trained_dataset, filters)
                all_models.extend([(model_type, trained_dataset, models)])
    
    # Generate all evaluation experiments
    gpu_cycle = 0
    
    # For base models, we need to handle without_icl separately
    base_without_icl_checked = set()  # Track which without_icl evaluations we've checked
    
    for model_type, trained_dataset, models in all_models:
        for model in models:
            for eval_dataset in args.eval_datasets:
                for icl_source in args.icl_source_datasets:
                    for icl_max_demos in args.icl_max_demos:
                        gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
                        gpu_cycle += 1
                        
                        if model_type == 'base':
                            # For base model, check if without_icl results exist
                            without_icl_key = f"{eval_dataset}_{icl_source}_{args.num_gen_tokens_eval}"
                            if without_icl_key not in base_without_icl_checked:
                                # Check if without_icl results exist
                                from utils import construct_base_without_icl_path
                                results_dir, results_filename = construct_base_without_icl_path(SimpleNamespace(
                                    model_id=args.model_id,
                                    eval_dataset_name=eval_dataset,
                                    icl_source_dataset=icl_source,
                                    icl_max_demos=icl_max_demos,
                                    num_generated_tokens_eval=args.num_gen_tokens_eval,
                                    uncertainty_analysis=args.uncertainty_analysis,
                                    top_k=args.top_k,
                                    eval_with_icl=False,
                                    merge_lora=args.merge_lora,
                                    gpu=gpu_id
                                ))
                                without_icl_path = os.path.join(results_dir, results_filename + ".json")
                                
                                if not os.path.exists(without_icl_path):
                                    # Need to run without_icl evaluation
                                    print(f"Adding base model without_icl evaluation for {eval_dataset}")
                                    task = EvaluationTask(
                                        base_output_dir=args.base_output_dir,
                                        model_type='base',
                                        model_id=args.model_id,
                                        trained_dataset=None,
                                        eval_dataset_name=eval_dataset,
                                        icl_source_dataset=icl_source,
                                        icl_max_demos=1,
                                        num_generated_tokens_eval=args.num_gen_tokens_eval,
                                        num_train_examples=2,
                                        run_idx=model['run_idx'],
                                        gpu_id=gpu_id,
                                        uncertainty_analysis=args.uncertainty_analysis,
                                        top_k=args.top_k,
                                        eval_with_icl=False,
                                        merge_lora=args.merge_lora
                                    )
                                    tasks.append(task)
                                
                                base_without_icl_checked.add(without_icl_key)

                            # Now check if with_icl results exist
                            dummy_args = SimpleNamespace(
                                base_output_dir=args.base_output_dir,
                                model_type=model_type,
                                model_id=args.model_id,
                                trained_dataset=None,
                                lora_type=None,
                                lora_r=None,
                                lora_alpha=None,
                                num_generated_tokens_train=None,
                                num_train_examples=model['num_examples'],
                                lr=None,
                                run_idx=model['run_idx'],
                                ce_loss_weight=None,
                                label_type=None,
                                eval_dataset_name=eval_dataset,
                                icl_source_dataset=icl_source,
                                icl_max_demos=min(icl_max_demos, model['num_examples'] - 1),
                                num_generated_tokens_eval=args.num_gen_tokens_eval,
                                uncertainty_analysis=args.uncertainty_analysis,
                                top_k=args.top_k,
                                eval_with_icl=args.eval_with_icl,
                                merge_lora=args.merge_lora,
                                gpu=gpu_id
                            )
                        else:
                            # For trained models, check if results exist
                            # Parse model_type to extract base method and training variant
                            if '-' in model_type:
                                base_method, training_variant = model_type.split('-', 1)
                            else:
                                base_method = 'lora'
                                training_variant = model_type
                            
                            # Handle different model types with their specific parameters
                            if base_method in ['lora'] or model_type in ['tok', 'act', 'tna', 'a2t', 't2a']:
                                dummy_args = SimpleNamespace(
                                    base_output_dir=args.base_output_dir,
                                    model_type=model_type,
                                    model_id=args.model_id,
                                    trained_dataset=trained_dataset,
                                    lora_type=model['lora_type'],
                                    lora_r=model['lora_r'],
                                    lora_alpha=model['lora_alpha'],
                                    num_generated_tokens_train=model['num_tokens_train'],
                                    num_train_examples=model['num_examples_train'],
                                    lr=model['lr'],
                                    run_idx=model['run_idx'],
                                    ce_loss_weight=model['ce_loss_weight'],
                                    label_type=model['label_type'],
                                    eval_dataset_name=eval_dataset,
                                    icl_source_dataset=icl_source,
                                    icl_max_demos=min(icl_max_demos, model['num_examples_train'] - 1),
                                    num_generated_tokens_eval=args.num_gen_tokens_eval,
                                    uncertainty_analysis=args.uncertainty_analysis,
                                    top_k=args.top_k,
                                    eval_with_icl=args.eval_with_icl,
                                    merge_lora=args.merge_lora,
                                    gpu=gpu_id,
                                    ldr_mode=args.ldr_mode,
                                    num_labelled_samples=model['num_examples_train'] if args.ldr_mode else None,
                                    num_unlabelled_samples=int(model['ldr_suffix'].split('_')[0]) if args.ldr_mode else None,
                                    max_permutations=int(model['ldr_suffix'].split('_')[1]) if args.ldr_mode else None
                                )
                            elif base_method == 'ia3':
                                dummy_args = SimpleNamespace(
                                    base_output_dir=args.base_output_dir,
                                    model_type=model_type,
                                    model_id=args.model_id,
                                    trained_dataset=trained_dataset,
                                    ia3_type=model['ia3_type'],
                                    num_generated_tokens_train=model['num_tokens_train'],
                                    num_train_examples=model['num_examples_train'],
                                    lr=model['lr'],
                                    run_idx=model['run_idx'],
                                    ce_loss_weight=model['ce_loss_weight'],
                                    label_type=model['label_type'],
                                    eval_dataset_name=eval_dataset,
                                    icl_source_dataset=icl_source,
                                    icl_max_demos=min(icl_max_demos, model['num_examples_train'] - 1),
                                    num_generated_tokens_eval=args.num_gen_tokens_eval,
                                    uncertainty_analysis=args.uncertainty_analysis,
                                    top_k=args.top_k,
                                    eval_with_icl=args.eval_with_icl,
                                    merge_lora=args.merge_lora,
                                    gpu=gpu_id,
                                    ldr_mode=args.ldr_mode,
                                    num_labelled_samples=model['num_examples_train'] if args.ldr_mode else None,
                                    num_unlabelled_samples=int(model['ldr_suffix'].split('_')[0]) if args.ldr_mode else None,
                                    max_permutations=int(model['ldr_suffix'].split('_')[1]) if args.ldr_mode else None
                                )
                            elif base_method in ['prompt', 'prefix']:
                                dummy_args = SimpleNamespace(
                                    base_output_dir=args.base_output_dir,
                                    model_type=model_type,
                                    model_id=args.model_id,
                                    trained_dataset=trained_dataset,
                                    num_virtual_tokens=model['num_virtual_tokens'],
                                    num_generated_tokens_train=model['num_tokens_train'],
                                    num_train_examples=model['num_examples_train'],
                                    lr=model['lr'],
                                    run_idx=model['run_idx'],
                                    ce_loss_weight=model['ce_loss_weight'],
                                    label_type=model['label_type'],
                                    eval_dataset_name=eval_dataset,
                                    icl_source_dataset=icl_source,
                                    icl_max_demos=min(icl_max_demos, model['num_examples_train'] - 1),
                                    num_generated_tokens_eval=args.num_gen_tokens_eval,
                                    uncertainty_analysis=args.uncertainty_analysis,
                                    top_k=args.top_k,
                                    eval_with_icl=args.eval_with_icl,
                                    merge_lora=args.merge_lora,
                                    gpu=gpu_id,
                                    ldr_mode=args.ldr_mode,
                                    num_labelled_samples=model['num_examples_train'] if args.ldr_mode else None,
                                    num_unlabelled_samples=int(model['ldr_suffix'].split('_')[0]) if args.ldr_mode else None,
                                    max_permutations=int(model['ldr_suffix'].split('_')[1]) if args.ldr_mode else None
                                )
                        
                        results_dir, results_filename = construct_results_path(dummy_args)
                        results_path = os.path.join(results_dir, results_filename + ".json")
                        if os.path.exists(results_path):
                            print(f"Skipping: results already exist at {results_path}")
                            continue

                        # Add task
                        if model_type == 'base':
                            task = EvaluationTask(
                                base_output_dir=args.base_output_dir,
                                model_type=model_type,
                                model_id=args.model_id,
                                trained_dataset=None,
                                num_train_examples=model['num_examples'],
                                lr=None,
                                run_idx=model['run_idx'],
                                ce_loss_weight=None,
                                label_type=None,
                                eval_dataset_name=eval_dataset,
                                icl_source_dataset=icl_source,
                                icl_max_demos=min(icl_max_demos, model['num_examples'] - 1),
                                num_generated_tokens_eval=args.num_gen_tokens_eval,
                                uncertainty_analysis=args.uncertainty_analysis,
                                top_k=args.top_k,
                                eval_with_icl=args.eval_with_icl,
                                merge_lora=args.merge_lora,
                                gpu_id=gpu_id,
                                ldr_mode=None,
                                num_labelled_samples=None,
                                num_unlabelled_samples=None,
                                max_permutations=None
                            )
                        elif base_method in ['lora'] or model_type in ['tok', 'act', 'tna', 'a2t', 't2a']:
                            task = EvaluationTask(
                                base_output_dir=args.base_output_dir,
                                model_type=model_type,
                                model_id=args.model_id,
                                trained_dataset=trained_dataset,
                                eval_dataset_name=eval_dataset,
                                icl_source_dataset=icl_source,
                                icl_max_demos=min(icl_max_demos, model['num_examples_train'] - 1),
                                num_generated_tokens_eval=args.num_gen_tokens_eval,
                                run_idx=model['run_idx'],
                                gpu_id=gpu_id,
                                lora_type=model['lora_type'],
                                lora_r=model['lora_r'],
                                lora_alpha=model['lora_alpha'],
                                num_generated_tokens_train=model['num_tokens_train'],
                                num_train_examples=model['num_examples_train'],
                                lr=model['lr'],
                                ce_loss_weight=model['ce_loss_weight'],
                                label_type=model['label_type'],
                                uncertainty_analysis=args.uncertainty_analysis,
                                top_k=args.top_k,
                                eval_with_icl=args.eval_with_icl,
                                merge_lora=args.merge_lora,
                                ldr_mode=args.ldr_mode,
                                num_labelled_samples=model['num_examples_train'] if args.ldr_mode else None,
                                num_unlabelled_samples=int(model['ldr_suffix'].split('_')[0]) if args.ldr_mode else None,
                                max_permutations=int(model['ldr_suffix'].split('_')[1]) if args.ldr_mode else None
                            )
                        elif base_method == 'ia3':
                            task = EvaluationTask(
                                base_output_dir=args.base_output_dir,
                                model_type=model_type,
                                model_id=args.model_id,
                                trained_dataset=trained_dataset,
                                eval_dataset_name=eval_dataset,
                                icl_source_dataset=icl_source,
                                icl_max_demos=min(icl_max_demos, model['num_examples_train'] - 1),
                                num_generated_tokens_eval=args.num_gen_tokens_eval,
                                run_idx=model['run_idx'],
                                gpu_id=gpu_id,
                                ia3_type=model['ia3_type'],
                                num_generated_tokens_train=model['num_tokens_train'],
                                num_train_examples=model['num_examples_train'],
                                lr=model['lr'],
                                ce_loss_weight=model['ce_loss_weight'],
                                label_type=model['label_type'],
                                uncertainty_analysis=args.uncertainty_analysis,
                                top_k=args.top_k,
                                eval_with_icl=args.eval_with_icl,
                                merge_lora=args.merge_lora,
                                ldr_mode=args.ldr_mode,
                                num_labelled_samples=model['num_examples_train'] if args.ldr_mode else None,
                                num_unlabelled_samples=int(model['ldr_suffix'].split('_')[0]) if args.ldr_mode else None,
                                max_permutations=int(model['ldr_suffix'].split('_')[1]) if args.ldr_mode else None
                            )
                        elif base_method in ['prompt', 'prefix']:
                            task = EvaluationTask(
                                base_output_dir=args.base_output_dir,
                                model_type=model_type,
                                model_id=args.model_id,
                                trained_dataset=trained_dataset,
                                eval_dataset_name=eval_dataset,
                                icl_source_dataset=icl_source,
                                icl_max_demos=min(icl_max_demos, model['num_examples_train'] - 1),
                                num_generated_tokens_eval=args.num_gen_tokens_eval,
                                run_idx=model['run_idx'],
                                gpu_id=gpu_id,
                                num_virtual_tokens=model['num_virtual_tokens'],
                                num_generated_tokens_train=model['num_tokens_train'],
                                num_train_examples=model['num_examples_train'],
                                lr=model['lr'],
                                ce_loss_weight=model['ce_loss_weight'],
                                label_type=model['label_type'],
                                uncertainty_analysis=args.uncertainty_analysis,
                                top_k=args.top_k,
                                eval_with_icl=args.eval_with_icl,
                                merge_lora=args.merge_lora,
                                ldr_mode=args.ldr_mode,
                                num_labelled_samples=model['num_examples_train'] if args.ldr_mode else None,
                                num_unlabelled_samples=int(model['ldr_suffix'].split('_')[0]) if args.ldr_mode else None,
                                max_permutations=int(model['ldr_suffix'].split('_')[1]) if args.ldr_mode else None
                            )
                        tasks.append(task)
    
    return tasks


def optimize_task_sequence(tasks: List[EvaluationTask]) -> List[List[EvaluationTask]]:
    """Optimize task sequence to minimize adapter switching"""
    # Group tasks by GPU
    gpu_groups = defaultdict(list)
    for task in tasks:
        gpu_groups[task.gpu_id].append(task)
    
    optimized_sequences = []
    
    for gpu_id, gpu_tasks in gpu_groups.items():
        # Group tasks by adapter key to minimize switching
        adapter_groups = defaultdict(list)
        for task in gpu_tasks:
            adapter_key = task.get_adapter_key()
            adapter_groups[adapter_key].append(task)
        
        # Sort adapter groups to prioritize base model first, then by adapter complexity
        sorted_adapters = sorted(adapter_groups.keys(), key=lambda x: (x != 'base', len(x)))
        
        # Create optimized sequence
        optimized_sequence = []
        for adapter_key in sorted_adapters:
            optimized_sequence.extend(adapter_groups[adapter_key])
        
        optimized_sequences.append(optimized_sequence)

    return optimized_sequences


def run_evaluation_on_gpu(gpu_id: int, tasks: List[EvaluationTask], args) -> List[Dict]:
    """Run evaluation tasks on a specific GPU"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    
    print(f"Starting evaluation on GPU {gpu_id}")
    
    # Load tokenizer once
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model manager
    model_manager = ModelManager(args.model_id, device, torch_dtype)
    
    # Setup parse answer function
    parse_answer_func = None
    if tasks[0].eval_dataset_name == 'gsm8k' or tasks[0].eval_dataset_name == 'gsm8ks':
        parse_answer_func = parse_answer_gsm8k
    elif tasks[0].eval_dataset_name == 'sciqa' or tasks[0].eval_dataset_name == 'strategyreason':
        parse_answer_func = parse_answer_sciqa
    elif 'hmath' in tasks[0].eval_dataset_name:
        parse_answer_func = parse_answer_boxed
    
    results = []
    
    # Process tasks in optimized sequence
    for task in tasks:
        print(f"\nProcessing task: {task.model_type} on {task.eval_dataset_name} (GPU {task.gpu_id})")

        # Load label information for uncertainty analysis
        label_tokens, remap_dict = None, None
        if task.uncertainty_analysis and task.num_generated_tokens_eval == 1:
            print("Loading label information for uncertainty analysis...")
            label_tokens, remap_dict = load_dataset_label_info(task.eval_dataset_name)
        
        try:
            # Get model for this task
            model = model_manager.get_model_for_task(task)
            
            # Load or create evaluation dataset
            print(f"Loading evaluation dataset: {task.eval_dataset_name}")
            eval_dataset_instance, loaded_cache_path = create_or_load_evaluation_dataset(
                eval_dataset_name=task.eval_dataset_name,
                icl_source_dataset=task.icl_source_dataset,
                run_idx=task.run_idx,
                num_training_examples=task.num_train_examples,
                max_demos=task.icl_max_demos,
                tokenizer=tokenizer,
                num_generated_tokens=task.num_generated_tokens_eval,
                force_rebuild=False,
                parse_answer_func=parse_answer_func
            )
            
            if eval_dataset_instance is None:
                print(f"Failed to load evaluation dataset for task: {task}")
                continue
            
            print(f"Loaded dataset instance with {len(eval_dataset_instance)} examples.")
            
            # Run evaluation
            print("Starting evaluation...")
            eval_without_icl = False
            if not task.eval_with_icl:
                eval_without_icl = True
            eval_with_icl = task.eval_with_icl
                
            evaluation_results = evaluate_model_batched(
                model, eval_dataset_instance, tokenizer, device,
                num_generated_tokens=task.num_generated_tokens_eval,
                eval_with_icl=task.eval_with_icl,
                eval_without_icl=eval_without_icl,
                parse_answer_func=parse_answer_func,
                uncertainty_analysis=task.uncertainty_analysis,
                label_tokens=label_tokens,
                top_k=task.top_k,
                batch_size=args.batch_size
            )
            
            # Calculate metrics
            print("Calculating metrics...")
            metrics = calculate_metrics(
                evaluation_results, 
                task.uncertainty_analysis, 
                label_tokens, 
                task.num_generated_tokens_eval, 
                task.top_k, 
                parse_answer_func
            )
            
            # Save results
            results_dir, results_filename = construct_results_path(task)
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, results_filename + ".json")
            
            # Structure saved data
            eval_specific_args = ['eval_dataset_name', 'icl_source_dataset', 'icl_max_demos', 
                                 'num_generated_tokens_eval', 'uncertainty_analysis', 'top_k', 
                                 'eval_with_icl', 'merge_lora', 'gpu_id']
            training_params = {k: v for k, v in task.__dict__.items() if k not in eval_specific_args}
            evaluation_config = {k: v for k, v in task.__dict__.items() if k in eval_specific_args}
            evaluation_config['cache_path_used'] = loaded_cache_path
            if task.uncertainty_analysis:
                evaluation_config['label_tokens'] = label_tokens
                evaluation_config['remap_dict'] = remap_dict
            
            # Clean up prob_info for JSON serialization
            for result in evaluation_results['with_icl']:
                if result['prob_info'] is not None and 'token_matrix' in result['prob_info']:
                    result['prob_info']['token_matrix'] = [str(token) for token in result['prob_info']['token_matrix']]
                    result['prob_info']['prob_matrix'] = [str(prob) for prob in result['prob_info']['prob_matrix']]
            for result in evaluation_results['without_icl']:
                if result['prob_info'] is not None and 'token_matrix' in result['prob_info']:
                    result['prob_info']['token_matrix'] = [str(token) for token in result['prob_info']['token_matrix']]
                    result['prob_info']['prob_matrix'] = [str(prob) for prob in result['prob_info']['prob_matrix']]
            
            output_data = {
                'trained_model_params': training_params,
                'evaluation_config': evaluation_config,
                'metrics': metrics,
                'results': evaluation_results
            }

            print('Metrics:')
            print(json.dumps(output_data['metrics'], indent=2))
            
            with open(results_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Results saved to: {results_path}")
            
            # Store result summary
            results.append({
                'task': task,
                'metrics': metrics,
                'results_path': results_path
            })
            
        except Exception as e:
            print(f"Error processing task {task}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Optimized batch evaluation script for distillation models")
    
    # Model selection
    parser.add_argument("--model_types", nargs='+', 
                        choices=['tok', 'act', 'a2t', 'tna', 't2a', 'base', 
                                'ia3-tok', 'ia3-act', 'ia3-tna', 'ia3-a2t', 'ia3-t2a',
                                'prompt-tok', 'prompt-act', 'prompt-tna', 'prompt-a2t', 'prompt-t2a',
                                'prefix-tok', 'prefix-act', 'prefix-tna', 'prefix-a2t', 'prefix-t2a'],
                        # default=['ia3-tok', 'ia3-act', 'ia3-tna', 'ia3-a2t', 'prefix-tok', 'prefix-act', 'prefix-tna', 'prefix-a2t'],
                        default=['tok', 'act', 'tna', 'a2t'],
                        # default=['base'],
                        help="Types of models to evaluate")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        choices=['meta-llama/Llama-3.2-1B', 'Qwen/Qwen3-4B-Base', 'Qwen/Qwen2.5-1.5B', 'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.1-8B', 'google/gemma-3-270m', 'meta-llama/Llama-3.2-3B'],
                        help="Base model ID")
    parser.add_argument("--trained_datasets", nargs='+', default=['gsm8k'],
                        help="Datasets the models were trained on")
    
    # Model filtering
    parser.add_argument("--lora_types", nargs='+', default=['qko'],
                        help="LoRA types to include")
    parser.add_argument("--lora_rs", nargs='+', type=int, default=[8],
                        help="LoRA r values to include")
    parser.add_argument("--lora_alphas", nargs='+', type=int, default=[8],
                        help="LoRA alpha values to include")
    parser.add_argument("--ia3_types", nargs='+', default=['qko'],
                        help="IA3 types to include")
    parser.add_argument("--num_virtual_tokens_list", nargs='+', type=int, default=[20],
                        help="Number of virtual tokens for prompt/prefix tuning")
    parser.add_argument("--num_tokens", type=int, default=200,
                        help="Number of training tokens to include")
    parser.add_argument("--num_examples", nargs='+', type=int, default=[4],
                        help="Number of training examples to include")
    # parser.add_argument("--lrs", nargs='+', type=float, default=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3],
    parser.add_argument("--lrs", nargs='+', type=float, default=[1e-4, 3e-4, 1e-3],
    # parser.add_argument("--lrs", nargs='+', type=float, default=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
                        help="Learning rates to include")
    parser.add_argument("--run_indices", nargs='+', type=int, default=[0, 1, 2, 3, 4],
                        help="Run indices to include")
    parser.add_argument("--label_types", nargs='+', 
                        choices=['ground_truth', 'icl_outputs'],
                        default=['icl_outputs', 'ground_truth'],
                        help="Label types to include (for tok models)")
    # parser.add_argument("--ce_loss_weights", nargs='+', type=float, default=[0.1, 0.5, 0.7, 0.9],
    parser.add_argument("--ce_loss_weights", nargs='+', type=float, default=[0.001, 0.01, 0.05, 0.5],
                        help="CE loss weights to include (for tna models)")
    
    # LDR (Low Data Regime) arguments
    parser.add_argument("--ldr_mode", action='store_true', 
                       help="Enable low data regime mode with N labelled samples and M unlabelled samples")
    parser.add_argument("--num_labelled_samples", nargs='+', type=int, default=[2, 4, 8], 
                       help="Number of labelled samples for ICL demos (N) - only used in LDR mode")
    parser.add_argument("--num_unlabelled_samples", nargs='+', type=int, default=[12], 
                       help="Number of unlabelled samples for training (M) - only used in LDR mode")
    parser.add_argument("--max_permutations", nargs='+', type=int, default=[1, 2], 
                       help="Maximum number of permutations of labelled samples (K) - only used in LDR mode")
    
    # Evaluation settings
    parser.add_argument("--eval_datasets", nargs='+', default=['gsm8k', 'gsm8ks'],
                        help="Datasets to evaluate on")
    parser.add_argument("--icl_source_datasets", nargs='+', default=['gsm8k'],
                        help="ICL source datasets")
    parser.add_argument("--icl_max_demos", nargs='+', type=int, default=[256],
                        help="ICL demo counts")
    parser.add_argument("--num_gen_tokens_eval", type=int, default=200,
                        help="Number of tokens to generate during evaluation")
    
    # Analysis options
    parser.add_argument("--uncertainty_analysis", action='store_true',
                        help="Enable uncertainty analysis (only when num_gen_tokens_eval=1)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K for uncertainty analysis")
    parser.add_argument("--eval_with_icl", action='store_true',
                        help="Evaluate with ICL demos")
    parser.add_argument("--merge_lora", default=True, action='store_true',
                        help="Merge LoRA weights before evaluation")
    
    # Performance settings
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Batch size for evaluation")
    parser.add_argument("--gpus", nargs='+', type=int, default=[3],
                        help="GPU IDs to use")
    parser.add_argument("--base_output_dir", type=str, default="../outputs",
                        help="Base directory for trained models")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        help="Torch dtype for the model")
    
    args = parser.parse_args()

    if args.model_types == ['base']:
        args.eval_with_icl = True

    print("Generating evaluation tasks...")
    tasks = generate_evaluation_tasks(args)
    
    if not tasks:
        print("No tasks to evaluate!")
        return
    
    print(f"Generated {len(tasks)} evaluation tasks")
    
    # Optimize task sequence
    print("Optimizing task sequence...")
    optimized_sequences = optimize_task_sequence(tasks)

    # Show first few tasks as examples
    print("\nFirst 3 tasks:")
    for i, task in enumerate(tasks[:3]):
        print(f"  {i+1}. {task.model_id.split('/')[-1]} model, eval type: {task.model_type}: {task.eval_dataset_name} on {task.icl_source_dataset} (GPU {task.gpu_id})")
    
    # input("\nPress Enter to start evaluations...")
    
    # Run evaluations
    start_time = time.time()
    all_results = []
    
    for gpu_id, gpu_tasks in zip(args.gpus, optimized_sequences):
        if gpu_tasks:
            results = run_evaluation_on_gpu(gpu_id, gpu_tasks, args)
            all_results.extend(results)
    
    end_time = time.time()
    
    # Summary
    successful = len(all_results)
    failed = len(tasks) - successful
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(tasks)*100:.1f}%")
    print(f"Total time: {end_time - start_time:.1f} seconds")
    print(f"{'='*60}")
    
    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(args),
        'total_tasks': len(tasks),
        'successful': successful,
        'failed': failed,
        'success_rate': successful/len(tasks)*100,
        'total_time_seconds': end_time - start_time,
        'results': [{'task': str(r['task']), 'metrics': r['metrics'], 'path': r['results_path']} for r in all_results]
    }
    
    summary_path = f"{args.base_output_dir}/evaluation_summaries/optimized_batch_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
