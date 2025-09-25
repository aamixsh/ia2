import os
import sys
import json
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import our custom modules
from data import ICLDataset, ICLDatasetShuffled
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--num_train_examples", type=int, default=64)
    parser.add_argument("--lora_type", type=str, default='qkv') # Kept for naming consistency if needed, but not used in dataset creation
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--max_icl_demos", type=int, default=48)
    parser.add_argument("--num_dev_examples", type=int, default=48)
    parser.add_argument("--shuffle_demos", action='store_true', help="Whether to use shuffled ICL demos")
    parser.add_argument("--num_shuffles", type=int, default=5, help="Number of shuffles per example when using shuffled demos")
    parser.add_argument("--fixed_demos", action='store_true', help="Whether to use a fixed set of ICL demos for all training examples")
    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    model_id = "meta-llama/Llama-3.2-1B" # Or make this an arg
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load raw datasets
    print("Loading raw datasets...")
    with open(f"../data/{args.dataset}/main.json", "r") as f:
        datasets = json.load(f)

    print(f"Dataset keys: {datasets.keys()}")
    if 'query_examples' in datasets: # Original script had 'query_examples', checking for 'train_examples' as per logic below
        print(f"Query examples count: {len(datasets['query_examples'])}")
        if len(datasets['query_examples']) > 0:
            print(f"First query example format: {type(datasets['query_examples'][0])}")
            print(f"First query example: {datasets['query_examples'][0]}")
    elif 'train_examples' in datasets:
         print(f"Train examples count: {len(datasets['train_examples'])}")

    for run_idx in range(args.num_runs):
        print(f"Processing run {run_idx + 1}/{args.num_runs}")
        num_training_samples = args.num_train_examples

        # Choose output directory based on fixed_demos flag
        if args.fixed_demos:
            output_dir = f"../data/{args.dataset}/train_fixed"
        else:
            output_dir = f"../data/{args.dataset}/train"
        
        os.makedirs(output_dir, exist_ok=True)
        
        if args.shuffle_demos:
            dataset_filepath = f"{output_dir}/shuffled_{args.num_shuffles}_{num_training_samples}_{run_idx}.json"
        else:
            dataset_filepath = f"{output_dir}/{num_training_samples}_{run_idx}.json"

        if not os.path.exists(dataset_filepath):
            if 'train_examples' not in datasets:
                print(f"Error: 'train_examples' not found in raw dataset at ../data/{args.dataset}/main.json")
                sys.exit(1)

            if datasets['train_examples']:

                all_indices = list(range(len(datasets['train_examples'])))

                rng = np.random.RandomState(run_idx)
                rng.shuffle(all_indices)

                train_indices = all_indices[:num_training_samples]
                num_dev_samples = min(args.num_dev_examples, num_training_samples // 2)
                dev_indices = all_indices[num_training_samples:num_training_samples + num_dev_samples]
                remaining_indices = all_indices[num_training_samples + num_dev_samples:] # Not used in this script directly

                train_examples = [datasets['train_examples'][i] for i in train_indices]
                dev_examples = [datasets['train_examples'][i] for i in dev_indices]
                # remaining_examples = [datasets['train_examples'][i] for i in remaining_indices] # Not used

                print(f"Selected {len(train_examples)} examples for training")
                print(f"Selected {len(dev_examples)} examples for dev")

                print("Creating ICL demo lists...")
                icl_prompts = []
                # Ensure max_demos calculation is safe
                if not train_examples:
                    print("Warning: No training examples to create ICL demos from.")
                    max_demos = 0
                else:
                    max_demos = min(args.max_icl_demos, len(train_examples) -1 if len(train_examples) > 1 else 0)

                if args.fixed_demos:
                    # Create a single fixed set of ICL demos for all training examples
                    print("Creating fixed ICL demo set...")
                    
                    # Use remaining examples (not used for training or dev) as demo sources
                    available_demo_indices = remaining_indices.copy()
                    
                    if len(available_demo_indices) == 0:
                        print("Warning: No remaining examples available for ICL demos. Using training examples instead.")
                        # Fall back to using training examples (excluding current example during training)
                        icl_prompts = []
                        print("Fixed demos mode requires remaining examples. Created empty demo set.")
                    elif len(available_demo_indices) > max_demos:
                        demo_indices = rng.choice(available_demo_indices, max_demos, replace=False)
                        demo_indices_list = list(demo_indices)
                        rng.shuffle(demo_indices_list)
                        
                        # Create the fixed demo set using remaining examples
                        fixed_demo_prompts = []
                        question_key = 'question' if 'question' in datasets['train_examples'][0] else 'sentence'
                        answer_key = 'answer' if 'answer' in datasets['train_examples'][0] else 'label'
                        
                        for idx in demo_indices_list:
                            demo_example = datasets['train_examples'][idx]
                            demo_q = demo_example.get(question_key, "")
                            demo_a = demo_example.get(answer_key, "")
                            demo = f"{demo_q}{str(demo_a)}"
                            fixed_demo_prompts.append(demo)
                        
                        # Store demos once, not per example
                        icl_prompts = fixed_demo_prompts
                        print(f"Created fixed ICL demo set with {len(fixed_demo_prompts)} demos from remaining examples")
                    else:
                        demo_indices_list = list(available_demo_indices)
                        rng.shuffle(demo_indices_list)
                        
                        # Create the fixed demo set using all available remaining examples
                        fixed_demo_prompts = []
                        question_key = 'question' if 'question' in datasets['train_examples'][0] else 'sentence'
                        answer_key = 'answer' if 'answer' in datasets['train_examples'][0] else 'label'
                        
                        for idx in demo_indices_list:
                            demo_example = datasets['train_examples'][idx]
                            demo_q = demo_example.get(question_key, "")
                            demo_a = demo_example.get(answer_key, "")
                            demo = f"{demo_q}{str(demo_a)}"
                            fixed_demo_prompts.append(demo)
                        
                        # Store demos once, not per example
                        icl_prompts = fixed_demo_prompts
                        print(f"Created fixed ICL demo set with {len(fixed_demo_prompts)} demos from all remaining examples")
                
                else:
                    # Original logic: create unique ICL demos for each training example
                    icl_prompts = []
                    for i, example in enumerate(train_examples):
                        available_indices = list(range(len(train_examples)))
                        if i < len(train_examples): # Ensure i is a valid index
                            available_indices.remove(i)

                        if len(available_indices) > max_demos:
                            demo_indices = rng.choice(available_indices, max_demos, replace=False)
                        else:
                            demo_indices = available_indices
                        
                        # Ensure demo_indices is a list of integers before shuffling
                        demo_indices_list = list(demo_indices)
                        rng.shuffle(demo_indices_list)

                        example_prompts = []
                        question_key = 'question' if 'question' in train_examples[0] else 'sentence' # Adapt to dataset structure
                        answer_key = 'answer' if 'answer' in train_examples[0] else 'label' # Adapt to dataset structure

                        for idx in demo_indices_list:
                            demo_q = train_examples[idx].get(question_key, "")
                            demo_a = train_examples[idx].get(answer_key, "")
                            demo = f"{demo_q}{str(demo_a)}" # Ensure answer is string
                            example_prompts.append(demo)
                        icl_prompts.append(example_prompts)

                    print(f"Created {len(icl_prompts)} ICL demo lists with {max_demos} demos each")

                # if not icl_prompts or not any(icl_prompts): # Check if icl_prompts is empty or contains only empty lists
                #     print("Warning: No ICL prompts were generated. Max ICL demo length will be 0.")
                #     max_icl_demo_length = 0
                # else:
                #     if args.fixed_demos:
                #         # icl_prompts is a single list of demos
                #         max_icl_demo_length = len(tokenizer.encode('\n\n'.join(icl_prompts))) if icl_prompts else 0
                #     else:
                #         # icl_prompts is a list of lists
                #         max_icl_demo_length = max(len(tokenizer.encode('\n\n'.join(demo))) for demo in icl_prompts if demo)


                # question_key_for_max_len = 'question' if 'question' in train_examples[0] else 'sentence'
                # if not train_examples: # Handle case with no train_examples
                #     print("Warning: No training examples for dynamic_max_length calculation.")
                #     dynamic_max_length = 50 + max_icl_demo_length # Default calculation
                # else:
                #     dynamic_max_length = max(len(tokenizer.encode(str(example.get(question_key_for_max_len, "")))) + max_icl_demo_length + 50 for example in train_examples)
                # print(f"Dynamic max length: {dynamic_max_length}")

                # if args.shuffle_demos:
                #     if args.fixed_demos:
                #         # For fixed demos, use the same demo list for all examples
                #         train_dataset = ICLDatasetShuffled(
                #             train_examples,
                #             tokenizer,
                #             icl_demos=[icl_prompts] * len(train_examples),
                #             max_length=dynamic_max_length,
                #             num_shuffles=args.num_shuffles,
                #             seed=run_idx
                #         )
                #         dev_dataset = ICLDatasetShuffled(
                #             dev_examples,
                #             tokenizer,
                #             icl_demos=[icl_prompts] * len(dev_examples),
                #             max_length=dynamic_max_length,
                #             num_shuffles=args.num_shuffles,
                #             seed=run_idx
                #         )
                #     else:
                #         train_dataset = ICLDatasetShuffled(
                #             train_examples,
                #             tokenizer,
                #             icl_demos=icl_prompts,
                #             max_length=dynamic_max_length,
                #             num_shuffles=args.num_shuffles,
                #             seed=run_idx
                #         )
                #         dev_dataset = ICLDatasetShuffled(
                #             dev_examples,
                #             tokenizer,
                #             icl_demos=icl_prompts[:len(dev_examples)] if icl_prompts else [],
                #             max_length=dynamic_max_length,
                #             num_shuffles=args.num_shuffles,
                #             seed=run_idx
                #         )
                # else:
                #     if args.fixed_demos:
                #         # For fixed demos, use the same demo list for all examples
                #         train_dataset = ICLDataset(
                #             train_examples,
                #             tokenizer,
                #             icl_demos=[icl_prompts] * len(train_examples),
                #             max_length=dynamic_max_length
                #         )
                #         dev_dataset = ICLDataset(
                #             dev_examples,
                #             tokenizer,
                #             icl_demos=[icl_prompts] * len(dev_examples),
                #             max_length=dynamic_max_length
                #         )
                #     else:
                #         train_dataset = ICLDataset(
                #             train_examples,
                #             tokenizer,
                #             icl_demos=icl_prompts,
                #             max_length=dynamic_max_length
                #         )
                #         dev_dataset = ICLDataset(
                #             dev_examples,
                #             tokenizer,
                #             icl_demos=icl_prompts[:len(dev_examples)] if icl_prompts else [],
                #             max_length=dynamic_max_length
                #         )

                print("Saving datasets...")
                if args.fixed_demos:
                    # For fixed demos, store demos once and reference them for all examples
                    datasets_to_save = {
                        'train': {
                            'examples': train_examples,
                            'icl_demos': icl_prompts  # Single list of demos for all examples
                        },
                        'dev': {
                            'examples': dev_examples,
                            'icl_demos': icl_prompts  # Same demos for dev set
                        },
                        'remaining_indices': remaining_indices,
                        'demo_type': 'fixed'  # Flag to indicate demo type
                    }
                else:
                    # For unique demos per example
                    datasets_to_save = {
                        'train': {
                            'examples': train_examples,
                            'icl_demos': icl_prompts
                        },
                        'dev': {
                            'examples': dev_examples,
                            'icl_demos': icl_prompts[:len(dev_examples)] if icl_prompts else []
                        },
                        'remaining_indices': remaining_indices,
                        'demo_type': 'unique'  # Flag to indicate demo type
                    }

            else:
                all_indices = list(range(len(datasets['val_examples'])))
                datasets_to_save = {
                    'remaining_indices': all_indices
                }

            with open(dataset_filepath, "w") as f:
                json.dump(datasets_to_save, f, indent=2)
            print(f"Saved datasets to {dataset_filepath}")

        else:
            print(f"Dataset already exists at {dataset_filepath}. Skipping creation.")
        print("-" * 30) # Separator for runs

    print("Dataset creation process completed for all runs.")


if __name__ == "__main__":
    main() 