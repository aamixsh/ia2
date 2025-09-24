import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import json
from utils import parse_answer_sciqa, parse_answer_gsm8k

class ICLDataset(Dataset):
    def __init__(self, examples, tokenizer, icl_demos=None, device=None, 
                 num_generated_tokens=1, parse_answer_func=None):
        self.examples = examples
        self.tokenizer = tokenizer
        self.icl_demos = icl_demos
        self.device = device # Not using
        self.num_generated_tokens = num_generated_tokens
        self.parse_answer_func = parse_answer_func

        if not self.icl_demos:
            raise ValueError("No ICL demos provided")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice notation (e.g., dataset[:2])
            return [self[i] for i in range(*idx.indices(len(self)))]
            
        # Get the example
        example = self.examples[idx]
        # Prepare input with and without ICL
        query = example['question']
        
        # Support multi-token answers instead of just first token
        answer_tokens = self.tokenizer.encode(str(example['answer']), add_special_tokens=False)
        if self.num_generated_tokens > 1:
            answer_tokens.append(self.tokenizer.eos_token_id)
        ground_truth_tensor = torch.tensor(answer_tokens)
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)

        # Input without ICL
        inputs_no_icl = self.tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=len(query_tokens) + 1 + max(len(answer_tokens), self.num_generated_tokens)
        )

        tok_remove = inputs_no_icl['input_ids'][0, 0] != self.tokenizer.bos_token_id

        if self.icl_demos:
            # Get the paired ICL demo
            icl_demo = self.icl_demos[idx]

            icl_text = "\n\n".join(icl_demo) + "\n\n"
            icl_query = icl_text + query

            icl_query_tokens = self.tokenizer.encode(icl_query, add_special_tokens=False)

            inputs_with_icl = self.tokenizer(
                icl_query,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=len(icl_query_tokens) + 1 + max(len(answer_tokens), self.num_generated_tokens)
            )

        # Base result
        result = {
            'example': example,
            'inputs_no_icl': inputs_no_icl,
            'inputs_with_icl': inputs_with_icl if self.icl_demos else None,
            'ground_truth': ground_truth_tensor,
            'no_icl_decode_position': len(query_tokens) - int(tok_remove),
            'with_icl_decode_position': len(icl_query_tokens) - int(tok_remove) if self.icl_demos else None
        }

        # For long-form answers, also include the full text answer and parsed eval_answer
        if self.num_generated_tokens > 1:
            full_answer_text = str(example['answer'])
            eval_answer = self.parse_answer_func(full_answer_text)
            result['eval_answer'] = eval_answer

        return result



class ICLDatasetWithOutputs(ICLDataset):
    def __init__(self, base_dataset, icl_outputs=None, icl_activations=None, training_mode='icl_outputs'):
        super().__init__(base_dataset.examples, base_dataset.tokenizer, base_dataset.icl_demos, base_dataset.device, base_dataset.num_generated_tokens, base_dataset.parse_answer_func)
        self.icl_outputs = icl_outputs
        self.icl_activations = icl_activations
        self.training_mode = training_mode
        if self.training_mode not in ['icl_outputs', 'ground_truth']:
            raise ValueError("Invalid training mode")

        if self.training_mode == 'icl_outputs' and not self.icl_outputs:
            raise ValueError("No ICL outputs provided")

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        result = super().__getitem__(idx)

        # Add ICL outputs if available
        if self.icl_outputs is not None or result['ground_truth'] is not None:

            if self.training_mode == 'icl_outputs' and self.icl_outputs is not None:
                icl_output_tokens = self.icl_outputs[idx]
                result['icl_output'] = torch.tensor(icl_output_tokens)

                # Create inputs_with_icl_outputs: concatenate no_icl input with icl outputs
                icl_output_tokens = self.icl_outputs[idx]
                if isinstance(icl_output_tokens, str):
                    icl_output_tokens = self.tokenizer.encode(icl_output_tokens, add_special_tokens=False)
                elif isinstance(icl_output_tokens, torch.Tensor):
                    icl_output_tokens = icl_output_tokens.tolist()
                
                # Concatenate input with icl outputs
                combined_ids = torch.cat([result['inputs_no_icl']['input_ids'][0, :result['no_icl_decode_position'] + 1], torch.tensor(icl_output_tokens)])

                # Create the new input tensor
                inputs_with_icl_outputs = {
                    'input_ids': combined_ids.unsqueeze(0),
                    'attention_mask': torch.ones_like(combined_ids).unsqueeze(0)
                }
                result['inputs_with_icl_outputs'] = inputs_with_icl_outputs
                if self.icl_activations is not None:
                    result['icl_activations'] = self.icl_activations[idx]
            
            elif self.training_mode == 'ground_truth' and result['ground_truth'] is not None:
                # Create inputs_with_ground_truth: concatenate no_icl input with ground truth
                ground_truth_tokens = result['ground_truth']
                if isinstance(ground_truth_tokens, torch.Tensor):
                    ground_truth_tokens = ground_truth_tokens.tolist()
                
                # Concatenate input with ground truth
                combined_ground_truth_ids = torch.cat([result['inputs_no_icl']['input_ids'][0, :result['no_icl_decode_position'] + 1], torch.tensor(ground_truth_tokens)])

                # Create the new input tensor
                inputs_with_ground_truth = {
                    'input_ids': combined_ground_truth_ids.unsqueeze(0),
                    'attention_mask': torch.ones_like(combined_ground_truth_ids).unsqueeze(0)
                }
                result['inputs_with_ground_truth'] = inputs_with_ground_truth
        
        return result

class ICLDatasetShuffled(Dataset):
    def __init__(self, examples, tokenizer, icl_demos=None, device=None, num_shuffles=1, seed=42, parse_answer_func=None, num_generated_tokens=1):
        """
        Dataset that allows using each example multiple times with shuffled ICL demos.
        
        Args:
            examples: List of examples
            tokenizer: Tokenizer to use
            icl_demos: List of ICL demos for each example
            device: Device to place tensors on
            num_shuffles: Number of times to use each example with different demo orderings
            seed: Random seed for reproducibility
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.icl_demos = icl_demos
        self.device = device
        self.num_shuffles = num_shuffles
        self.rng = np.random.RandomState(seed)
        self.parse_answer_func = parse_answer_func
        self.num_generated_tokens = num_generated_tokens

        # Pre-generate shuffled demo indices for each example and shuffle
        self.shuffled_demo_indices = []
        for _ in range(num_shuffles):
            for i in range(len(examples)):
                if icl_demos is not None:
                    demo_indices = list(range(len(icl_demos[i])))
                    self.rng.shuffle(demo_indices)
                    self.shuffled_demo_indices.append((i, demo_indices))
                else:
                    raise ValueError("No ICL demos provided")
                    
    def __len__(self):
        return len(self.examples) * self.num_shuffles
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice notation (e.g., dataset[:2])
            return [self[i] for i in range(*idx.indices(len(self)))]
            
        # Get the example index and its shuffled demos
        example_idx, demo_indices = self.shuffled_demo_indices[idx]
        example = self.examples[example_idx]
        
        # Prepare input with and without ICL
        query = example['question']
        
        # Support multi-token answers instead of just first token
        answer_tokens = self.tokenizer.encode(str(example['answer']), add_special_tokens=False)
        ground_truth_tensor = torch.tensor(answer_tokens, device=self.device)
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)

        # Input without ICL
        inputs_no_icl = self.tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=len(query_tokens) + max(len(answer_tokens), self.num_generated_tokens) + 10
        )

        # Get the shuffled ICL demos
        if self.icl_demos is not None:
            icl_demo = [self.icl_demos[example_idx][i] for i in demo_indices]
            icl_text = "\n\n".join(icl_demo) + "\n\n"
            icl_query = icl_text + query

            icl_query_tokens = self.tokenizer.encode(icl_query, add_special_tokens=False)
            icl_buffer_length = max(len(answer_tokens), self.num_generated_tokens)

            inputs_with_icl = self.tokenizer(
                icl_query,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=len(icl_query_tokens) + icl_buffer_length + 10
            )

        # Base result
        result = {
            'example': example,
            'inputs_no_icl': {k: v.squeeze(0).to(self.device) for k, v in inputs_no_icl.items()},
            'inputs_with_icl': {k: v.squeeze(0).to(self.device) for k, v in inputs_with_icl.items()},
            'ground_truth': ground_truth_tensor,
            'shuffle_idx': idx // len(self.examples),  # Which shuffle this example came from
            'example_idx': example_idx  # Original index of the example
        }

        # For long-form answers, also include the full text answer and parsed eval_answer
        if self.parse_answer_func:
            full_answer_text = str(example['answer'])
            eval_answer = self.parse_answer_func(full_answer_text)
            result['full_answer_text'] = full_answer_text
            result['eval_answer'] = eval_answer

        return result
        