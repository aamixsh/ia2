import os
import json
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from remap_utils import remap_tokens

def format_question_sciqa(example, prompt_template='normal', remap_dict=None):
    """Formats a single SCiQ example into the desired string format."""
    question_text = example['question']
    correct_answer = example['correct_answer']
    reason = example['support']

    label = f"{reason}\n#### {correct_answer}"

    answer_idx = 0
        
    if prompt_template == "hard":
        formatted = f"{question_text}:::"
        formatted += f"{reason}::"
    else:
        formatted = f"Question:{question_text}\nAnswer:"

    if remap_dict is not None:
        print ('Remapping not implemented for sciqa')
        exit(1)

    return formatted, label


def sciqa(output_dir="../data/sciqa", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads SCiQ dataset, processes it for QA, and saves split datasets."""
    print("Loading SCiQ dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        train_ds = load_dataset("allenai/sciq", split='train')
        val_ds = load_dataset("allenai/sciq", split='validation')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        print ('Label remapping not implemented for sciqa')
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)  # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    answer_lengths = []
    token_lengths = []
    token_answer_lengths = []

    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, label = format_question_sciqa(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        answer_lengths.append(len(label))
        token_lengths.append(len(tokenizer.encode(formatted_q)))
        token_answer_lengths.append(len(tokenizer.encode(label)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, label = format_question_sciqa(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        answer_lengths.append(len(label))
        token_lengths.append(len(tokenizer.encode(formatted_q)))
        token_answer_lengths.append(len(tokenizer.encode(label)))
    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 'free',
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "max_answer_length": max(answer_lengths),
            "max_95p_answer_length": np.percentile(answer_lengths, 95),
            "min_answer_length": min(answer_lengths),
            "mean_answer_length": np.mean(answer_lengths),
            "median_answer_length": np.median(answer_lengths),
            "std_answer_length": np.std(answer_lengths),
            "max_token_answer_length": max(token_answer_lengths),
            "max_95p_token_answer_length": np.percentile(token_answer_lengths, 95),
            "min_token_answer_length": min(token_answer_lengths),
            "mean_token_answer_length": np.mean(token_answer_lengths),
            "median_token_answer_length": np.median(token_answer_lengths),
            "std_token_answer_length": np.std(token_answer_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")


def format_question_gsm8k(example, prompt_template='normal', remap_dict=None):
    """Formats a single GSM8K example into the desired string format."""
    text = example['question']
    label = example['answer']

    if prompt_template == "hard":
        formatted = f"{text}:::"
    else:
        formatted = f"Question:{text}\nAnswer:"

    if remap_dict is not None:
        print ('Remapping not implemented for GSM8K')
        exit(1)

    return formatted, label


def gsm8k(output_dir="../data/gsm8k", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal", symbolic=False):
    """Loads GSM8K dataset, processes it by category, and saves split datasets."""
    print("Loading GSM8K dataset...")
    try:
        if symbolic:
            output_dir = output_dir + "s"
            train_ds = load_dataset("apple/GSM-Symbolic", 'main', split='test')
            val_ds = load_dataset("apple/GSM-Symbolic", 'main', split='test')
        else:
            train_ds = load_dataset("openai/gsm8k", 'main', split='train')
            val_ds = load_dataset("openai/gsm8k", 'main', split='test')
        print(f"Dataset loaded. Total examples: {len(train_ds)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        print ('Label remapping not implemented for GSM8K')
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)  # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    answer_lengths = []
    token_lengths = []
    token_answer_lengths = []

    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, label = format_question_gsm8k(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        answer_lengths.append(len(label))
        token_lengths.append(len(tokenizer.encode(formatted_q)))
        token_answer_lengths.append(len(tokenizer.encode(label)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, label = format_question_gsm8k(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        answer_lengths.append(len(label))
        token_lengths.append(len(tokenizer.encode(formatted_q)))
        token_answer_lengths.append(len(tokenizer.encode(label)))
    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 'free',
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "max_answer_length": max(answer_lengths),
            "max_95p_answer_length": np.percentile(answer_lengths, 95),
            "min_answer_length": min(answer_lengths),
            "mean_answer_length": np.mean(answer_lengths),
            "median_answer_length": np.median(answer_lengths),
            "std_answer_length": np.std(answer_lengths),
            "max_token_answer_length": max(token_answer_lengths),
            "max_95p_token_answer_length": np.percentile(token_answer_lengths, 95),
            "min_token_answer_length": min(token_answer_lengths),
            "mean_token_answer_length": np.mean(token_answer_lengths),
            "median_token_answer_length": np.median(token_answer_lengths),
            "std_token_answer_length": np.std(token_answer_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")



def format_question_cmath(example, prompt_template='normal', remap_dict=None):
    """Formats a single camel Math example into the desired string format."""
    text = example['question']
    label = example['answer']

    if prompt_template == "hard":
        formatted = f"{text}:::"
    else:
        formatted = f"Question:{text}\nAnswer:"

    if remap_dict is not None:
        label = remap_dict[label]

    return formatted, label

# TODO: not working
def cmath(output_dir="../data/cmath", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads CMATH dataset, processes it by category, and saves split datasets."""
    print("Loading CMATH dataset...")
    try:
        full_ds = load_dataset("camel-ai/math", split='train')
        print(f"Dataset loaded. Total examples: {len(full_ds)}")

        filtered_ds = full_ds.filter(lambda x: x['topic;'] in ["Probability", "Statistics", "Combinatorics"])
        print(f"Filtered dataset size: {len(filtered_ds)}")
        unique_columns = set(filtered_ds.column_names)
        print(f"Unique column names: {unique_columns}")

        unique_topics = set(filtered_ds['topic;'])
        unique_subtopics = set(filtered_ds['sub_topic'])
        
        print(f"Unique topics: {unique_topics}")
        for topic in unique_topics:
            count = len(filtered_ds.filter(lambda x: x['topic;'] == topic))
            print(f"Number of samples for topic '{topic}': {count}")

        print(f"Unique subtopics: {unique_subtopics}")
        for subtopic in unique_subtopics:
            count = len(filtered_ds.filter(lambda x: x['sub_topic'] == subtopic))
            print(f"Number of samples for subtopic '{subtopic}': {count}")
        input()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return


def format_question_finsen(example, prompt_template='normal', remap_dict=None):
    """Formats a single Finsen example into the desired string format."""
    text = example['sentence']
    label = example['label']

    if prompt_template == "hard":
        formatted = f"{text}:::"
    else:
        formatted = f"Text:{text}\nLabel:"

    if remap_dict is not None:
        label = remap_dict[label]

    return formatted, label

def finsen(output_dir="../data/finsen", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads Finsen dataset, processes it by category, and saves split datasets."""
    print("Loading Finsen dataset...")
    try:
        dataset = load_dataset("takala/financial_phrasebank", 'sentences_50agree')
        # The dataset only has a 'train' split, so we need to split it manually
        full_ds = dataset['train']
        print(f"Dataset loaded. Total examples: {len(full_ds)}")
        
        # Split into train and validation
        full_ds = full_ds.shuffle(seed=seed)
        full_ds = full_ds.filter(lambda x: x['label'] in [0, 2])
        full_ds = full_ds.map(lambda x: {'label': 1 if x['label'] == 2 else x['label']})
        print(f"Number of examples for label 0: {len(full_ds.filter(lambda x: x['label'] == 0))}")
        print(f"Number of examples for label 1: {len(full_ds.filter(lambda x: x['label'] == 1))}")
        
        split_point = len(full_ds) - num_val_samples
        train_ds = full_ds.select(range(split_point))
        val_ds = full_ds.select(range(split_point, len(full_ds)))
        # # Select only the labels 0 and 1
        # # Change the label 2 to 1
        # train_ds = train_ds.map(lambda x: {'label': 1 if x['label'] == 2 else x['label']})
        # val_ds = val_ds.map(lambda x: {'label': 1 if x['label'] == 2 else x['label']})
        # # Count the number of examples for each label
        # print(f"Number of examples for label 0: {len(train_ds.filter(lambda x: x['label'] == 0))}")
        # print(f"Number of examples for label 1: {len(train_ds.filter(lambda x: x['label'] == 1))}")
        # print(f"Number of examples for label 0: {len(val_ds.filter(lambda x: x['label'] == 0))}")
        # print(f"Number of examples for label 1: {len(val_ds.filter(lambda x: x['label'] == 1))}")
        
        print(f"Split into {len(train_ds)} train and {len(val_ds)} validation examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = set([ex['label'] for ex in train_ds] + [ex['label'] for ex in val_ds])
        unique_labels = sorted(list(unique_labels))
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)  # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []

    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, label = format_question_finsen(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, label = format_question_finsen(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 2,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")


def format_question_bbcnews(example, prompt_template='normal', agnews_label_map=None, cut_text_length=None, remap_dict=None):
    """Formats a single BBC News example into the desired string format."""

    text = example['text']
    answer_idx = example['label']

    if agnews_label_map is not None:
        answer_idx = agnews_label_map[answer_idx]

    if cut_text_length is not None:
        text = text[:cut_text_length]
        text = text.split(" ")
        text = " ".join(text[:-1])

    if prompt_template == "hard":
        formatted = f"{text}:::"
    else:
        formatted = f"Text:{text}\nLabel:"

    if remap_dict is not None:
        answer_idx = remap_dict[answer_idx]

    return formatted, answer_idx

def bbcnews(output_dir="../data/bbcnews", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal", val_only=False):
    """Loads BBC News dataset, processes it by category, and saves split datasets."""
    print("Loading BBC News dataset...")
    try:
        if val_only:
            val_ds = load_dataset("SetFit/bbc-news", split='test')
        else:
            train_ds = load_dataset("SetFit/bbc-news", split='train')
            val_ds = load_dataset("SetFit/bbc-news", split='test')
            print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    agnews_label_map = {3: 0, 4: 0, 2: 1, 1: 2, 0: 3}
    
    if not val_only:
        train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = [0, 1, 2, 3]
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)  # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []

    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    if not val_only:
        for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
            if i == num_train_samples:
                break
            formatted_q, label = format_question_bbcnews(example, prompt_template, agnews_label_map, cut_text_length=256, remap_dict=remap_dict)
            processed_train_examples.append({
                "idx": i,
                "question": formatted_q,
                "answer": str(label),
            })
            query_lengths.append(len(formatted_q))
            token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, label = format_question_bbcnews(example, prompt_template, agnews_label_map, cut_text_length=256, remap_dict=remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 4,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")


def format_question_sst2(example, prompt_template="normal", remap_dict=None):
    """Formats a single SST2 example into the desired string format."""
    text = example['sentence']
    label = example['label']

    if prompt_template == "hard":
        formatted = f"{text}:::"
    else:
        formatted = f"Text:{text}\nLabel:"

    if remap_dict is not None:
        label = remap_dict[label]

    return formatted, label

def sst2(output_dir="../data/sst2", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads SST2 dataset, processes it by category, and saves split datasets."""
    print("Loading SST2 dataset...")
    try:
        train_ds = load_dataset("stanfordnlp/sst2", split='train')
        val_ds = load_dataset("stanfordnlp/sst2", split='validation')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = set([ex['label'] for ex in train_ds] + [ex['label'] for ex in val_ds])
        unique_labels = sorted(list(unique_labels))
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)  # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []

    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, label = format_question_sst2(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, label = format_question_sst2(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 2,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")

def format_question_poems(example, prompt_template="normal", remap_dict=None):
    """Formats a single poem sentiment example into the desired string format."""
    text = example['verse_text']
    label = example['label']

    if prompt_template == "hard":
        formatted = f"{text}:::"
    else:
        formatted = f"Text:{text}\nLabel:"

    if remap_dict is not None:
        label = remap_dict[label]

    return formatted, label

def poems(output_dir="../data/poems", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal", val_only=False):
    """Loads poem_sentiment dataset, processes it by category, and saves split datasets."""
    print("Loading poem_sentiment dataset...")
    try:
        if val_only:
            val_ds = load_dataset("google-research-datasets/poem_sentiment", split='train')
        else:
            train_ds = load_dataset("google-research-datasets/poem_sentiment", split='train')
            val_ds = load_dataset("google-research-datasets/poem_sentiment", split='validation')
            print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if not val_only:
        train_ds = train_ds.filter(lambda x: x['label'] in [0, 1]).shuffle(seed=seed)
    val_ds = val_ds.filter(lambda x: x['label'] in [0, 1]).shuffle(seed=seed)

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = [0, 1]
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)  # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []

    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    if not val_only:
        for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
            if i == num_train_samples:
                break
            formatted_q, label = format_question_poems(example, prompt_template, remap_dict)
            processed_train_examples.append({
                "idx": i,
                "question": formatted_q,
                "answer": str(label),
            })
            query_lengths.append(len(formatted_q))
            token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, label = format_question_poems(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 2,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")

def format_question_mathqa(example, prompt_template="normal", remap_dict=None):
    """Formats a single QA example into the desired string format."""
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    question_text = example['question']
    choices = example['choices']['text']
    correct_answer = label_dict[example['answerKey']]

    if prompt_template == "hard":
        formatted = f"{question_text}:::"
        for i, choice in enumerate(choices):
            formatted += f"{choice}::"
        formatted += ":"
    else:
        choices_str = ""
        for i, choice in enumerate(choices):
            if remap_dict is not None:
                choices_str += f"[{remap_dict[i]}]{choice}\n"
            else:
                choices_str += f"[{i}]{choice}\n"
        formatted = f"Question:{question_text}\nChoices:\n{choices_str}\nAnswer:"

    if remap_dict is not None:
        correct_answer = remap_dict[correct_answer]

    return formatted, correct_answer


def mathqa(output_dir="../data/mathqa", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads QASC dataset, processes it by category, and saves split datasets."""
    print("Loading QASC dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        train_ds = load_dataset("allenai/math_qa", split='validation')
        val_ds = load_dataset("allenai/math_qa", split='test')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
        print(train_ds[0])
        print(val_ds[0])
        exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)
    
    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = [0, 1, 2, 3, 4, 5, 6, 7]
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed) # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []
    
    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, correct_answer = format_question_qasc(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": example['id'],
            "question": formatted_q,
            "answer": correct_answer,
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, correct_answer = format_question_qasc(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": example['id'],
            "question": formatted_q,
            "answer": correct_answer,
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 8,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4) # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")

def format_question_qasc(example, prompt_template="normal", remap_dict=None):
    """Formats a single QA example into the desired string format."""
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    question_text = example['question']
    choices = example['choices']['text']
    correct_answer = label_dict[example['answerKey']]

    if prompt_template == "hard":
        formatted = f"{question_text}:::"
        for i, choice in enumerate(choices):
            formatted += f"{choice}::"
        formatted += ":"
    else:
        choices_str = ""
        for i, choice in enumerate(choices):
            if remap_dict is not None:
                choices_str += f"[{remap_dict[i]}]{choice}\n"
            else:
                choices_str += f"[{i}]{choice}\n"
        formatted = f"Question:{question_text}\nChoices:\n{choices_str}\nAnswer:"

    if remap_dict is not None:
        correct_answer = remap_dict[correct_answer]

    return formatted, correct_answer


def qasc(output_dir="../data/qasc", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads QASC dataset, processes it by category, and saves split datasets."""
    print("Loading QASC dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        train_ds = load_dataset("allenai/qasc", split='train')
        val_ds = load_dataset("allenai/qasc", split='validation')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)
    
    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = [0, 1, 2, 3, 4, 5, 6, 7]
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed) # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []
    
    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, correct_answer = format_question_qasc(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": example['id'],
            "question": formatted_q,
            "answer": correct_answer,
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, correct_answer = format_question_qasc(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": example['id'],
            "question": formatted_q,
            "answer": correct_answer,
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 8,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4) # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")

def format_question_sciq(example, prompt_template="normal", remap_dict=None):
    """Formats a single QA example into the desired string format."""
    question_text = example['question']
    distractor1 = example['distractor1']
    distractor2 = example['distractor2']
    distractor3 = example['distractor3']
    correct_answer = example['correct_answer']
    options = [distractor1, distractor2, distractor3, None]

    # Shuffle options
    np.random.shuffle(options)

    answer_idx = None
    for i, option in enumerate(options):
        if option is None:
            answer_idx = i
            options[i] = correct_answer
        
    if prompt_template == "hard":
        formatted = f"{question_text}:::"
        for i, option in enumerate(options):
            formatted += f"{option}::"
        formatted += ":"
    else:
        choices_str = ""
        for i, option in enumerate(options):
            if remap_dict is not None:
                choices_str += f"[{remap_dict[i]}]{option}\n"
            else:
                choices_str += f"[{i}]{option}\n"
        formatted = f"Question:{question_text}\nChoices:\n{choices_str}\nAnswer:"

    if remap_dict is not None:
        answer_idx = remap_dict[answer_idx]

    return formatted, answer_idx

def sciq(output_dir="../data/sciq", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads SCiQ dataset, processes it by category, and saves split datasets."""
    print("Loading SCiQ dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        train_ds = load_dataset("allenai/sciq", split='train')
        val_ds = load_dataset("allenai/sciq", split='validation')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    np.random.seed(seed) # for reproducibility of train/val splits

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None
    
    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = [0, 1, 2, 3]
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []
    
    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, answer_idx = format_question_sciq(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, answer_idx = format_question_sciq(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 4,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4) # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")

def format_question_arc(example, prompt_template="normal", remap_dict=None):
    """Formats a single QA example into the desired string format."""
    question_text = example['question']
    choices = example['choices']
    answerKey = example['answerKey']
    options = [choices['text'][i] for i in range(len(choices['text']))]
    keys = ['A', 'B', 'C', 'D']
    key_dict = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    answer_idx = key_dict[answerKey]

    if prompt_template == "hard":
        formatted = f"{question_text}:::"
        for i, option in enumerate(options):
            formatted += f"{option}::"
        formatted += ":"
    else:
        choices_str = ""
        for i, option in enumerate(options):
            if remap_dict is not None:
                choices_str += f"[{remap_dict[keys[i]]}]{option}\n"
            else:
                choices_str += f"[{i}]{option}\n"
        formatted = f"Question:{question_text}\nChoices:\n{choices_str}\nAnswer:"

    if remap_dict is not None:
        answer_idx = remap_dict[key_dict[answerKey]]

    return formatted, answer_idx

def arc(output_dir="../data/arc", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads ARC dataset, processes it by category, and saves split datasets."""
    print("Loading ARC dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        train_ds = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split='train')
        val_ds = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split='test')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    np.random.seed(seed) # for reproducibility of train/val splits

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None
    
    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = ['A', 'B', 'C', 'D']
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []
    
    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if example['choices']['label'] != ['A', 'B', 'C', 'D']:
            continue
        if len(processed_train_examples) == num_train_samples:
            break
        formatted_q, answer_idx = format_question_arc(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if example['choices']['label'] != ['A', 'B', 'C', 'D']:
            continue
        if len(processed_val_examples) == num_val_samples:
            break
        formatted_q, answer_idx = format_question_arc(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 4,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4) # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")


def format_question_strategytf(example, prompt_template="normal", remap_dict=None):
    """Formats a single QA example into the desired string format."""
    question_text = example['question']
    answerKey = example['answer']
    keys = [False, True]
    key_dict = {
        False: 0,
        True: 1,
    }
    answer_idx = key_dict[answerKey]

    if prompt_template == "hard":
        formatted = f"{question_text}:::"
        formatted += ":"
    else:
        formatted = f"Question:{question_text}\nAnswer:"

    if remap_dict is not None:
        answer_idx = remap_dict[key_dict[answerKey]]

    return formatted, answer_idx

def strategytf(output_dir="../data/strategytf", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads StrategyTF dataset, processes it by category, and saves split datasets."""
    print("Loading StrategyTF dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        full_ds = load_dataset("wics/strategy-qa", split='test')

        print(f"Dataset loaded. Total examples: {len(full_ds)}")
        
        # Split into train and validation
        full_ds = full_ds.shuffle(seed=seed)
        print(f"Number of examples for label false: {len(full_ds.filter(lambda x: x['answer'] == 'false'))}")
        print(f"Number of examples for label true: {len(full_ds.filter(lambda x: x['answer'] == 'true'))}")
        
        split_point = len(full_ds) - num_val_samples
        train_ds = full_ds.select(range(split_point))
        val_ds = full_ds.select(range(split_point, len(full_ds)))

        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    np.random.seed(seed) # for reproducibility of train/val splits

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None
    
    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        unique_labels = [False, True]
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []
    
    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if len(processed_train_examples) == num_train_samples:
            break
        formatted_q, answer_idx = format_question_strategytf(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if len(processed_val_examples) == num_val_samples:
            break
        formatted_q, answer_idx = format_question_strategytf(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 2,
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4) # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")


def format_question_strategyreason(example, prompt_template='normal', remap_dict=None):
    """Formats a single SCiQ example into the desired string format."""
    question_text = example['question']
    answer = str(example['answer']).lower()
    reason = " ".join(example['facts'])

    label = f"{reason}\n#### {answer}"

    if prompt_template == "hard":
        formatted = f"{question_text}:::"
        formatted += f"{reason}::"
    else:
        formatted = f"Question:{question_text}\nAnswer:"

    if remap_dict is not None:
        print ('Remapping not implemented for sciqa')
        exit(1)

    return formatted, label


def strategyreason(output_dir="../data/strategyreason", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads StrategyReason dataset, processes it for QA, and saves split datasets."""
    print("Loading StrategyReason dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        full_ds = load_dataset("wics/strategy-qa", split='test')

        print(f"Dataset loaded. Total examples: {len(full_ds)}")
        
        # Split into train and validation
        full_ds = full_ds.shuffle(seed=seed)
        print(f"Number of examples for label false: {len(full_ds.filter(lambda x: x['answer'] == 'false'))}")
        print(f"Number of examples for label true: {len(full_ds.filter(lambda x: x['answer'] == 'true'))}")
        
        split_point = len(full_ds) - num_val_samples
        train_ds = full_ds.select(range(split_point))
        val_ds = full_ds.select(range(split_point, len(full_ds)))

        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        print ('Label remapping not implemented for sciqa')
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)  # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    answer_lengths = []
    token_lengths = []
    token_answer_lengths = []

    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, label = format_question_strategyreason(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        answer_lengths.append(len(label))
        token_lengths.append(len(tokenizer.encode(formatted_q)))
        token_answer_lengths.append(len(tokenizer.encode(label)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, label = format_question_strategyreason(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        answer_lengths.append(len(label))
        token_lengths.append(len(tokenizer.encode(formatted_q)))
        token_answer_lengths.append(len(tokenizer.encode(label)))
    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 'free',
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "max_answer_length": max(answer_lengths),
            "max_95p_answer_length": np.percentile(answer_lengths, 95),
            "min_answer_length": min(answer_lengths),
            "mean_answer_length": np.mean(answer_lengths),
            "median_answer_length": np.median(answer_lengths),
            "std_answer_length": np.std(answer_lengths),
            "max_token_answer_length": max(token_answer_lengths),
            "max_95p_token_answer_length": np.percentile(token_answer_lengths, 95),
            "min_token_answer_length": min(token_answer_lengths),
            "mean_token_answer_length": np.mean(token_answer_lengths),
            "median_token_answer_length": np.median(token_answer_lengths),
            "std_token_answer_length": np.std(token_answer_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")


def format_question_hmath(example, prompt_template='normal', remap_dict=None):
    """Formats a single Hendryck Math example into the desired string format."""
    text = example['problem']
    label = example['solution']

    if prompt_template == "hard":
        formatted = f"{text}:::"
    else:
        formatted = f"Question:{text}\nAnswer:"

    if remap_dict is not None:
        print ('Remapping not implemented for Hendryck Math')
        exit(1)

    return formatted, label


def hmath(output_dir="../data/hmath", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal", subset='algebra'):
    """Loads Hendryck Math dataset, processes it by category, and saves split datasets."""
    print("Loading Hendryck Math dataset...")
    try:
        train_ds = load_dataset("EleutherAI/hendrycks_math", subset, split='train')
        val_ds = load_dataset("EleutherAI/hendrycks_math", subset, split='test')
        print(f"Dataset loaded. Total examples: {len(train_ds)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    output_dir = output_dir + "_" + subset

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        print ('Label remapping not implemented for Hendryck Math')
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed)  # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    answer_lengths = []
    token_lengths = []
    token_answer_lengths = []

    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, label = format_question_hmath(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        answer_lengths.append(len(label))
        token_lengths.append(len(tokenizer.encode(formatted_q)))
        token_answer_lengths.append(len(tokenizer.encode(label)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, label = format_question_hmath(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(label),
        })
        query_lengths.append(len(formatted_q))
        answer_lengths.append(len(label))
        token_lengths.append(len(tokenizer.encode(formatted_q)))
        token_answer_lengths.append(len(tokenizer.encode(label)))
    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": 'free',
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "max_answer_length": max(answer_lengths),
            "max_95p_answer_length": np.percentile(answer_lengths, 95),
            "min_answer_length": min(answer_lengths),
            "mean_answer_length": np.mean(answer_lengths),
            "median_answer_length": np.median(answer_lengths),
            "std_answer_length": np.std(answer_lengths),
            "max_token_answer_length": max(token_answer_lengths),
            "max_95p_token_answer_length": np.percentile(token_answer_lengths, 95),
            "min_token_answer_length": min(token_answer_lengths),
            "mean_token_answer_length": np.mean(token_answer_lengths),
            "median_token_answer_length": np.median(token_answer_lengths),
            "std_token_answer_length": np.std(token_answer_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4)  # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")

def format_question_cosmosqa(example, prompt_template="normal", remap_dict=None):
    """Formats a single CosmosQA example into the desired string format."""
    context = example['context']
    question = example['question']
    options = [example['answer0'], example['answer1'], example['answer2'], example['answer3']]
    answer_idx = example['label']

    # Move "None of the above choices ." to the last position if it exists, and update answer_idx accordingly
    none_str = "None of the above choices ."
    if none_str in options:
        none_idx = options.index(none_str)
        # Move the "None of the above choices ." to the last position
        options.append(options.pop(none_idx))
        # If the answer_idx was pointing to the moved option, update to new index (last)
        if answer_idx == none_idx:
            answer_idx = 3
        # If the answer_idx was after the moved option, decrement by 1
        elif answer_idx > none_idx:
            answer_idx -= 1

    if prompt_template == "hard":
        formatted = f"{context}::::{question}:::"
        for i, option in enumerate(options):
            formatted += f"{option}::"
        formatted += ":"
    else:
        formatted = f"Context:{context}\nQuestion:{question}\nChoices:\n"

        for i, option in enumerate(options):
            if remap_dict is not None:
                formatted += f"[{remap_dict[i]}]{option}\n"
            else:
                formatted += f"[{i}]{option}\n"
        
        formatted += f"Answer:"

    if remap_dict is not None:
        answer_idx = remap_dict[answer_idx]

    return formatted, answer_idx


def cosmosqa(output_dir="../data/cosmosqa", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads CosmosQA dataset, processes it by category, and saves split datasets."""
    print("Loading CosmosQA dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        train_ds = load_dataset("allenai/cosmos_qa", split='train')
        val_ds = load_dataset("allenai/cosmos_qa", split='validation')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    unique_labels = set([ex['label'] for ex in train_ds] + [ex['label'] for ex in val_ds])
    unique_labels = sorted(list(unique_labels))

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed) # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []
    
    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, answer_idx = format_question_cosmosqa(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, answer_idx = format_question_cosmosqa(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": len(unique_labels),
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4) # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")

def format_question_hellaswag(example, prompt_template="normal", remap_dict=None):
    """Formats a single Hellaswag example into the desired string format."""
    sentence = example['ctx']
    options = [example['endings'][0], example['endings'][1], example['endings'][2], example['endings'][3]]
    answer_idx = example['label']

    if prompt_template == "hard":
        formatted = f"{sentence}:::"
        for i, option in enumerate(options):
            formatted += f"{option}::"
        formatted += ":"

    else:
        formatted = f"Sentence:{sentence}\nContinuation:\n"

        for i, option in enumerate(options):
            if remap_dict is not None:
                formatted += f"[{remap_dict[i]}]{option}\n"
            else:
                formatted += f"[{i}]{option}\n"

        formatted += f"Answer:"

    if remap_dict is not None:
        answer_idx = remap_dict[answer_idx]

    return formatted, answer_idx


def hellaswag(output_dir="../data/hellaswag", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads Hellaswag dataset, processes it by category, and saves split datasets."""
    print("Loading Hellaswag dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        train_ds = load_dataset("Rowan/hellaswag", split='train')
        val_ds = load_dataset("Rowan/hellaswag", split='validation')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    unique_labels = set([ex['label'] for ex in train_ds] + [ex['label'] for ex in val_ds])
    unique_labels = sorted(list(unique_labels))

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed) # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []
    
    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, answer_idx = format_question_hellaswag(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, answer_idx = format_question_hellaswag(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": len(unique_labels),
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4) # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")

def format_question_agnews(example, prompt_template='normal', remap_dict=None):
    """Formats a single Agnews example into the desired string format."""

    text = example['text']
    answer_idx = example['label']

    if prompt_template == "hard":
        formatted = f"{text}:::"
    else:
        formatted = f"Text:{text}\nLabel:"

    if remap_dict is not None:
        answer_idx = remap_dict[answer_idx]

    return formatted, answer_idx


def agnews(output_dir="../data/agnews", seed=42, model_id="meta-llama/Llama-3.2-1B", num_val_samples=500, num_train_samples=2000, prompt_template="normal", label_template="normal"):
    """Loads Agnews dataset, processes it by category, and saves split datasets."""
    print("Loading Agnews dataset...")
    # Load only the test split as per the dataset structure on Hugging Face
    try:
        train_ds = load_dataset("fancyzhx/ag_news", split='train')
        val_ds = load_dataset("fancyzhx/ag_news", split='test')
        print(f"Dataset loaded. Number of train examples: {len(train_ds)}")
        print(f"Dataset loaded. Number of val examples: {len(val_ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_ds = train_ds.shuffle(seed=seed)
    val_ds = val_ds.shuffle(seed=seed)

    unique_labels = set([ex['label'] for ex in train_ds] + [ex['label'] for ex in val_ds])
    unique_labels = sorted(list(unique_labels))

    remap_dict = None

    if prompt_template == "hard":
        output_dir = output_dir + "_hard"
    if label_template == "remap":
        output_dir = output_dir + "_remap"
        vocab = remap_tokens
        np.random.seed(seed)
        remap_dict = {}
        used_tokens = set()
        for label in unique_labels:
            # Randomly select a token from the vocab that hasn't been used yet
            while True:
                token = np.random.choice(vocab)
                if token not in used_tokens:
                    used_tokens.add(token)
                    break
            remap_dict[label] = token

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    np.random.seed(seed) # for reproducibility of train/val splits

    print(f"\nProcessing train set...")

    query_lengths = []
    token_lengths = []
    
    # Format examples
    processed_train_examples = []
    processed_val_examples = []
    for i, example in enumerate(tqdm(train_ds, desc=f"Formatting train set", leave=False)):
        if i == num_train_samples:
            break
        formatted_q, answer_idx = format_question_agnews(example, prompt_template, remap_dict)
        processed_train_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    for i, example in enumerate(tqdm(val_ds, desc=f"Formatting val set", leave=False)):
        if i == num_val_samples:
            break
        formatted_q, answer_idx = format_question_agnews(example, prompt_template, remap_dict)
        processed_val_examples.append({
            "idx": i,
            "question": formatted_q,
            "answer": str(answer_idx),
        })
        query_lengths.append(len(formatted_q))
        token_lengths.append(len(tokenizer.encode(formatted_q)))

    # Save the data for the category
    output_path = os.path.join(output_dir, "main.json")
    data_dict = {
        "metadata": {
            "max_choices": len(unique_labels),
            "remap_dict": remap_dict,
            "max_query_length": max(query_lengths),
            "max_95p_query_length": np.percentile(query_lengths, 95),
            "min_query_length": min(query_lengths),
            "mean_query_length": np.mean(query_lengths),
            "median_query_length": np.median(query_lengths),
            "std_query_length": np.std(query_lengths),
            "num_train_examples": len(processed_train_examples),
            "num_val_examples": len(processed_val_examples),
            "max_token_length": max(token_lengths),
            "max_95p_token_length": np.percentile(token_lengths, 95),
            "min_token_length": min(token_lengths),
            "mean_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "std_token_length": np.std(token_lengths),
        },
        "train_examples": processed_train_examples,
        "val_examples": processed_val_examples
    }

    try:
        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=4) # Use indent for readability
        print(f"  Saved data to {output_path}")
    except Exception as e:
        print(f"  Error saving data: {e}")

    print("\nFinished processing.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "gsm8k", "gsm8ks", "poems", "qasc", "sciq", "cosmosqa", "hellaswag", "agnews", "bbcnews", "finsen", "cmath", "mathqa", "sciqa", "arc", "strategytf", "strategyreason", "hmath"])
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--subset", type=str, default="algebra", choices=["algebra", "prealgebra", "precalculus", "intermediate_algebra", "geometry", "number_theory", "counting_and_probability"])
    parser.add_argument("--num_val_samples", type=int, default=500)
    parser.add_argument("--prompt_template", type=str, default="normal", choices=["normal", "hard"])
    parser.add_argument("--label_template", type=str, default="normal", choices=["normal", "remap"])
    args = parser.parse_args()
    if args.dataset == "cmath":
        cmath(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    if args.dataset == "sst2":
        sst2(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "gsm8k":
        gsm8k(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template) 
    elif args.dataset == "gsm8ks":
        gsm8k(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template, symbolic=True) 
    elif args.dataset == "sciqa":
        sciqa(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "poems":
        poems(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template, val_only=True) 
    elif args.dataset == "mathqa":
        mathqa(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "qasc":
        qasc(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "sciq":
        sciq(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "cosmosqa":
        cosmosqa(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "hellaswag":
        hellaswag(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "agnews":
        agnews(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "bbcnews":
        bbcnews(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template, val_only=True)
    elif args.dataset == "finsen":
        finsen(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "arc":
        arc(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "strategytf":
        strategytf(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "strategyreason":
        strategyreason(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template)
    elif args.dataset == "hmath":
        hmath(num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, prompt_template=args.prompt_template, label_template=args.label_template, subset=args.subset)