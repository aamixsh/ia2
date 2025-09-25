import os
import sys
import json
import argparse
import subprocess
from multiprocessing import Pool, cpu_count
from itertools import product
from tqdm import tqdm

def create_dataset_task(params):
    """Run a single dataset creation task with the given parameters"""
    dataset, num_train_examples, num_runs, max_icl_demos, num_dev_examples, shuffle_demo, num_shuffles, fixed_demos = params
    
    script = 'create_training_datasets.py'
    cmd = [
        'python', script,
        '--dataset', dataset,
        '--num_train_examples', str(num_train_examples),
        '--num_runs', str(num_runs),
        '--max_icl_demos', str(max_icl_demos),
        '--num_dev_examples', str(num_dev_examples),
        '--num_shuffles', str(num_shuffles)
    ]
    if shuffle_demo:
        cmd.append('--shuffle_demos')
    if fixed_demos:
        cmd.append('--fixed_demos')
    
    # Run the command
    cmd_str = ' '.join(cmd)
    print(f"Starting dataset creation: {cmd_str}")
    try:
        # Using shell=False is generally safer, ensure all args are strings
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print(f"Successfully completed: {cmd_str}")
            # print(f"Stdout:\n{stdout}") # Optional: print stdout
            return True
        else:
            print(f"Error running: {cmd_str}")
            print(f"Return code: {process.returncode}")
            print(f"Stderr:\n{stderr}")
            if stdout:
                print(f"Stdout:\n{stdout}")
            return False
            
    except FileNotFoundError:
        print(f"Error: The script '{script}' was not found. Make sure it's in the correct path.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while running {cmd_str}: {e}")
        print(f"Stderr: {stderr}")
        if stdout:
            print(f"Stdout: {stdout}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create all training datasets based on various configurations.")
    parser.add_argument("--dataset", type=str, nargs='+', default=["hmath_algebra"], 
                        choices=["sst2", "gsm8k", "qasc", "sciq", "cosmosqa", "hellaswag", "agnews", 
                                 "qasc_simple", "sciq_simple", "cosmosqa_simple", "hellaswag_simple", "agnews_simple", "arc", "strategytf", "strategyreason"],
                        help="Base dataset name(s) to generate variations for.")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()), help="Number of parallel processes to use.")
    parser.add_argument("--num_examples", type=int, nargs='+', default=[2, 4, 8, 16, 32], 
                        help="List of numbers of training examples to use.")
    parser.add_argument("--num_runs", type=int, default=5, 
                        help="List of numbers of runs for each configuration.")
    parser.add_argument("--max_icl_demos", type=int, default=256, 
                        help="List of maximum numbers of in-context learning demonstrations.")
    parser.add_argument("--num_dev_examples", type=int, default=48, 
                        help="List of numbers of development examples to use.")
    parser.add_argument("--shuffle_demo", type=bool, default=False, 
                        help="List of boolean values indicating whether to shuffle demonstrations.")
    parser.add_argument("--num_shuffles", type=int, nargs='+', default=[2, 4, 8], 
                        help="List of numbers of shuffles to use if shuffle_demos is True.")
    parser.add_argument("--fixed_demos", type=bool, action=argparse.BooleanOptionalAction, default=False, 
                        help="List of boolean values indicating whether to use fixed demonstrations.")
    args = parser.parse_args()
    
    # Define parameter lists for dataset creation
    all_task_params = []
    for dataset, num_examples, num_shuffles_config in product(
        args.dataset, args.num_examples, args.num_shuffles
    ):
        if not args.shuffle_demo and num_shuffles_config != 2:
            # This filter ensures that for non-shuffled demos, we only generate one configuration 
            # for num_shuffles (arbitrarily 2, as in the reference run_all_experiments.py).
            # The actual value of num_shuffles doesn't matter to create_training_datasets.py when --shuffle_demos is off.
            continue
        
        # The num_dev_examples passed to the script.
        # The create_training_datasets.py script's --num_dev_examples has default 48.
        # It then calculates: num_dev_samples = min(args.num_dev_examples, num_training_samples // 2)
        # So, passing num_dev_config (e.g., 128) ensures it's capped appropriately by the inner script.

        exists = True
        for i in range(args.num_runs):
            if args.fixed_demos:
                # Fixed demos are stored in train_fixed directory
                if args.shuffle_demo:
                    dataset_filepath = f"../data/{dataset}/train_fixed/shuffled_{num_shuffles_config}_{num_examples}_{i}.json"
                else:
                    dataset_filepath = f"../data/{dataset}/train_fixed/{num_examples}_{i}.json"
            else:
                # Regular demos are stored in train directory
                if args.shuffle_demo:
                    dataset_filepath = f"../data/{dataset}/train/shuffled_{num_shuffles_config}_{num_examples}_{i}.json"
                else:
                    dataset_filepath = f"../data/{dataset}/train/{num_examples}_{i}.json"
            if not os.path.exists(dataset_filepath):
                exists = False
                break

        if exists:
            print(f"Dataset already exists at {dataset_filepath}. Skipping creation.")
            continue
        
        all_task_params.append((
            dataset, num_examples, args.num_runs, args.max_icl_demos, args.num_dev_examples, args.shuffle_demo, num_shuffles_config, args.fixed_demos
        ))

    if not all_task_params:
        print("No dataset creation tasks generated. Check configurations.")
        return

    print(f"Generated {len(all_task_params)} dataset creation tasks.")
    print("First 5 task parameters:")
    for p in all_task_params[:5]:
        print(p)
    
    # Create output directory for logs if it doesn't exist
    # The create_training_datasets.py script saves datasets in ../data/ , not ../outputs/
    # This script can save its own log/config in ../outputs/
    os.makedirs("../outputs/dataset_creation_logs", exist_ok=True)

    print(f"\nStarting {len(all_task_params)} dataset creation tasks using {args.workers} workers...")
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(create_dataset_task, all_task_params),
            total=len(all_task_params),
            desc="Creating datasets"
        ))
    
    successful_tasks = sum(1 for r in results if r)
    print(f"\nDataset Creation Summary:")
    print(f"Total tasks attempted: {len(all_task_params)}")
    print(f"Successful tasks: {successful_tasks}")
    print(f"Failed tasks: {len(all_task_params) - successful_tasks}")
    
    # Save experiment configuration
    config_log = {
        'base_dataset': args.dataset,
        'num_workers': args.workers,
        'parameter_space': {
            'datasets_list': args.dataset,
            'num_examples_list': args.num_examples,
            'num_runs_config_list': args.num_runs,
            'max_icl_demos_list': args.max_icl_demos,
            'num_dev_examples_config_list': args.num_dev_examples,
            'shuffle_demo_config_list': args.shuffle_demo,
            'num_shuffles_config_list': args.num_shuffles,
            'fixed_demos_config_list': args.fixed_demos
        },
        'tasks': [
            {
                'params': list(task_param), # Convert tuple to list for JSON
                'success': result
            }
            for task_param, result in zip(all_task_params, results)
        ]
    }
    
    log_filepath = f"../outputs/dataset_creation_logs/{args.dataset}_dataset_creation_config.json"
    with open(log_filepath, "w") as f:
        json.dump(config_log, f, indent=2)
    print(f"Dataset creation configuration and results saved to {log_filepath}")

if __name__ == "__main__":
    main() 