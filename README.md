# IA2: In-Context Activation Alignment

A comprehensive framework for training and evaluating neural networks using the IA2 (In-Context Activation Alignment) method and related approaches for improving in-context learning performance.

## 🌟 Overview

This project implements and compares the IA2 method and related training approaches:
- **`tok`** (SFT): Supervised Fine-Tuning - Traditional cross-entropy loss training on ground truth or ICL output tokens
- **`act`** (IA2): In-Context Activation Alignment - MSE loss training to imitate ICL activations  
- **`a2t`** (IA2 → SFT): Sequential training from IA2 to SFT
- **`tna`** (IA2 + SFT): Combined IA2 and SFT training with both MSE and cross-entropy losses

The system also supports **base model evaluation** for direct comparison without any adapter training.

## 📁 Project Structure

```
unified_scripts/
├── train_unified.py              # 🚀 Unified training script (all methods)
├── train_all_unified.py          # 🔥 Batch training script
├── evaluate_batch_optimized.py   # 📊 Optimized batch evaluation script
├── plot_unified.py              # 📉 Unified plotting script
├── plot_all_unified.py          # 🎨 Batch plotting script
├── data.py                      # 📦 Dataset utilities
├── utils.py                     # 🛠️  Common utilities
├── subspace_overlap_analysis.py # 🔍 Subspace overlap analysis
└── activation_similarity_analysis.py # 📊 Activation similarity analysis
```

## 🚀 Quick Start

### 1. Training Models

**SFT (Supervised Fine-Tuning):**
```bash
python train_unified.py --training_method tok --dataset gsm8k --label_type icl_outputs --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0
```

**IA2 (In-Context Activation Alignment):**
```bash
python train_unified.py --training_method act --dataset gsm8k --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0
```

**IA2 + SFT (Combined training):**
```bash
python train_unified.py --training_method tna --dataset gsm8k --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0 --ce_loss_weight 0.002
```

**Sequential training (IA2 → SFT):**
```bash
# First train IA2 model
python train_unified.py --training_method act --dataset gsm8k --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0

# Then train SFT model (sequential training is handled automatically)
python train_unified.py --training_method a2t --dataset gsm8k --label_type icl_outputs --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0
```

**Batch training:**
```bash
# Train all methods across multiple configurations
python train_all_unified.py --training_methods tok act tna --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0

# Include sequential training with dependency validation
python train_all_unified.py --training_methods tok act --include_sequential --sequential_first --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0

# Hyperparameter sweep
python train_all_unified.py --training_methods tok --datasets gsm8k --num_train_examples 100 200 --lrs 1e-4 5e-4 1e-3 --run_indices 0 1 2 --max_parallel 4

### 2. Evaluating Models

**Batch evaluation (recommended):**
```bash
# Evaluate all model types
python evaluate_batch_optimized.py --model_types tok act tna base --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5

# Filter specific configurations
python evaluate_batch_optimized.py --model_types tok --lora_types qkv --num_examples 100 --lrs 1e-4 --run_indices 0 --uncertainty_analysis

# Uncertainty analysis
python evaluate_batch_optimized.py --model_types tok act tna base --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5 --uncertainty_analysis
```

### 3. Generating Plots

**Single configuration:**
```bash
# Standard comparison plots
python plot_unified.py --trained_dataset gsm8k --eval_dataset gsm8k --icl_source_dataset gsm8k --icl_max_demos 5 --model_types base tok act tna

# Uncertainty analysis plots
python plot_unified.py --trained_dataset gsm8k --eval_dataset gsm8k --icl_source_dataset gsm8k --icl_max_demos 5 --uncertainty_mode --plot_types all
```

**Batch plotting:**
```bash
# Generate all plots
python plot_all_unified.py --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5 --include_uncertainty --include_standard
```

## 📖 Detailed Usage

### Training Methods

| Method | Description | Required Arguments | Output Directory |
|--------|-------------|-------------------|------------------|
| `tok` | SFT: Supervised Fine-Tuning with CE loss | `--label_type` | `../outputs/tok/{dataset}/` |
| `act` | IA2: In-Context Activation Alignment with MSE loss | None | `../outputs/act/{dataset}/` |  
| `tna` | IA2 + SFT: Combined MSE + CE loss training | `--ce_loss_weight` | `../outputs/tna/{dataset}/` |
| `a2t` | Sequential: IA2 → SFT | `--continue_training a2t` | `../outputs/a2t/{dataset}/` |
| `t2a` | Sequential: SFT → IA2 | `--continue_training t2a` | `../outputs/t2a/{dataset}/` |

### Model Naming Convention

**SFT models (`tok`):**
```
{model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}_{label_type}
```
Example: `Llama-3.2-1B_qkv_8_8_1_100_0.0001_0_icl_outputs`

**IA2 models (`act`):**
```
{model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}
```
Example: `Llama-3.2-1B_qkv_8_8_1_100_0.0001_0`

**IA2 + SFT models (`tna`):**
```
{model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}_{ce_weight}
```
Example: `Llama-3.2-1B_qkv_8_8_1_100_0.0001_0_0.002`

### Evaluation Modes

#### Standard Evaluation
- Generates text completions and compares with ground truth
- Metrics: `with_icl_accuracy`, `without_icl_accuracy`, `accuracy_delta`

#### Uncertainty Analysis
- **Requirements**: `--uncertainty_analysis` and `--num_generated_tokens_eval`
- **Additional metrics**: Top-K accuracy, label-set accuracy, entropy, uncertainty
- **Output**: Probability distributions and uncertainty measures

### Directory Structure

```
outputs/
├── tok/{dataset}/                    # SFT training models
├── act/{dataset}/                    # IA2 training models  
├── tna/{dataset}/                    # IA2 + SFT training models
├── a2t/{dataset}/                    # Sequential IA2 → SFT models
├── t2a/{dataset}/                    # Sequential SFT → IA2 models
└── evaluations/
    ├── base/                         # Base model evaluations
    ├── tok/{dataset}/                # SFT model evaluations
    ├── act/{dataset}/                # IA2 model evaluations
    ├── tna/{dataset}/                # IA2 + SFT model evaluations
    ├── base_uncertainty/             # Base model uncertainty evaluations
    ├── tok_uncertainty/{dataset}/    # SFT model uncertainty evaluations
    ├── act_uncertainty/{dataset}/    # IA2 model uncertainty evaluations
    └── tna_uncertainty/{dataset}/    # IA2 + SFT model uncertainty evaluations

plots/
└── unified/{dataset}/
    └── {eval_dataset}_{icl_source}_{demos}/
        ├── {model}_accuracy_with_icl.png
        ├── {model}_accuracy_without_icl.png
        ├── {model}_top1_accuracy_with_icl.png     # Uncertainty mode
        ├── {model}_label_accuracy_with_icl.png    # Uncertainty mode  
        └── {model}_uncertainty_with_icl.png       # Uncertainty mode
```

## 🔧 Configuration Options

### Common Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--training_method` | str | Required | Training method: `tok`, `act`, `tna` |
| `--dataset` | str | `gsm8k` | Training dataset |
| `--model_id` | str | `meta-llama/Llama-3.2-1B` | Base model |
| `--lora_type` | str | `qkv` | LoRA target modules |
| `--lora_r` | int | `8` | LoRA rank |
| `--lora_alpha` | int | `8` | LoRA scaling parameter |
| `--num_generated_tokens` | int | `1` | Number of tokens to generate/train on |
| `--num_train_examples` | int | `100` | Number of training examples |
| `--lr` | float | `1e-4` | Learning rate |
| `--run_idx` | int | `0` | Run index for multiple runs |

### Method-Specific Arguments

**SFT Training (`tok`):**
- `--label_type`: `ground_truth` or `icl_outputs`

**IA2 + SFT Training (`tna`):**
- `--ce_loss_weight`: Weight for cross-entropy loss (0-1)

**Sequential Training:**
- Sequential training is handled automatically when using `a2t` or `t2a` methods

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_type` | str | Required | Model type: `tok`, `act`, `tna`, `base` |
| `--eval_dataset_name` | str | Required | Evaluation dataset |
| `--icl_source_dataset` | str | Required | ICL demonstration source |
| `--icl_max_demos` | int | Required | Number of ICL demonstrations |
| `--uncertainty_analysis` | flag | False | Enable uncertainty analysis |
| `--num_generated_tokens_eval` | int | `1` | Tokens to generate during eval |
| `--eval_with_icl` | flag | False | Include ICL in evaluation |

## 🏃‍♂️ Typical Workflows

### Complete Training → Evaluation → Plotting Pipeline

**Option 1: Individual Training**
```bash
# 1. Train models individually
python train_unified.py --training_method tok --dataset gsm8k --label_type icl_outputs --num_train_examples 100 --lr 1e-4 --run_idx 0
python train_unified.py --training_method act --dataset gsm8k --num_train_examples 100 --lr 1e-4 --run_idx 0
python train_unified.py --training_method tna --dataset gsm8k --num_train_examples 100 --lr 1e-4 --run_idx 0 --ce_loss_weight 0.002

# 2. Evaluate models
python evaluate_batch_optimized.py --model_types base tok act tna --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5

# 3. Generate plots  
python plot_unified.py --trained_dataset gsm8k --eval_dataset gsm8k --icl_source_dataset gsm8k --icl_max_demos 5 --model_types base tok act tna
```

**Option 2: Batch Training (Recommended)**
```bash
# 1. Batch train all models with parallel processing
python train_all_unified.py --training_methods tok act tna --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0 --max_parallel 3

# 2. Batch evaluate all models
python evaluate_batch_optimized.py --model_types base tok act tna --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5

# 3. Generate plots
python plot_unified.py --trained_dataset gsm8k --eval_dataset gsm8k --icl_source_dataset gsm8k --icl_max_demos 5 --model_types base tok act tna
```

### Sequential Training Workflow

```bash
# 1. Train base IA2 model
python train_unified.py --training_method act --dataset gsm8k --num_train_examples 100 --lr 1e-4 --run_idx 0

# 2. Train sequential model (a2t)
python train_unified.py --training_method a2t --dataset gsm8k --label_type icl_outputs --num_train_examples 100 --lr 1e-4 --run_idx 0

# 3. Evaluate sequential model
python evaluate_batch_optimized.py --model_types a2t --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5
```

### Uncertainty Analysis Workflow

```bash
# 1. Evaluate with uncertainty analysis
python evaluate_batch_optimized.py --model_types base tok act tna --uncertainty_analysis

# 2. Generate uncertainty plots
python plot_unified.py --trained_dataset gsm8k --eval_dataset gsm8k --icl_source_dataset gsm8k --icl_max_demos 5 --uncertainty_mode --plot_types all
```

### Hyperparameter Sweep Workflow

```bash
# 1. Comprehensive hyperparameter sweep across methods
python train_all_unified.py \
    --training_methods tok act tna \
    --datasets gsm8k \
    --num_train_examples 50 100 200 \
    --lrs 5e-5 1e-4 5e-4 1e-3 \
    --run_indices 0 1 2 \
    --ce_loss_weights 0.001 0.002 0.005 \
    --max_parallel 4 \
    --wandb_log

# 2. Evaluate all trained models
python evaluate_batch_optimized.py \
    --model_types tok act tna \
    --trained_datasets gsm8k \
    --eval_datasets gsm8k \
    --icl_source_datasets gsm8k \
    --icl_max_demos 5

# 3. Generate comparison plots
python plot_unified.py \
    --trained_dataset gsm8k \
    --eval_dataset gsm8k \
    --icl_source_dataset gsm8k \
    --icl_max_demos 5 \
    --model_types tok act tna
```

## 🎯 Key Features

### ✅ Unified Interface
- Single training script for all methods
- Consistent argument naming and conventions
- Standardized output directory structure

### ✅ Sequential Training Support  
- Seamless continuation from one method to another
- Automatic model discovery and loading
- Proper checkpoint management

### ✅ Comprehensive Evaluation
- Base model comparison
- Multiple evaluation metrics
- Uncertainty analysis with probability distributions
- Batch evaluation across configurations

### ✅ Advanced Plotting
- Method comparison visualizations
- Uncertainty analysis plots  
- Hyperparameter optimization insights
- Batch plotting across datasets

### ✅ Robust Error Handling
- Input validation and helpful error messages
- Automatic path construction and validation
- Graceful handling of missing models/data

## 🔍 Troubleshooting

### Common Issues

**Training fails with "PEFT model not found":**
- Ensure the base model exists when using `--continue_training`
- Check the model naming convention matches exactly

**Evaluation finds no models:**
- Verify the model directory structure matches the expected naming
- Check that `--trained_dataset` matches the training dataset used

**Plotting shows no data:**
- Ensure evaluation results exist in the expected directory
- Verify the dataset names and ICL configurations match between training/evaluation/plotting

### Directory Structure Verification

```bash
# Check training outputs
ls ../outputs/tok/gsm8k/
ls ../outputs/act/gsm8k/
ls ../outputs/tna/gsm8k/

# Check evaluation results
ls ../outputs/evaluations/tok/gsm8k/
ls ../outputs/evaluations/base/

# Check plots
ls ../plots/unified/gsm8k/
```

## 📊 Expected Results

The unified system enables direct comparison between:
- **Base Model**: Pretrained model performance with ICL
- **SFT**: Traditional supervised fine-tuning approach
- **IA2**: In-Context Activation Alignment for ICL behavior imitation
- **IA2 + SFT**: Combined approach leveraging both methods
- **Sequential Training**: Progressive improvement strategies

Typical performance ordering: `base < tok < act ≈ tna` with sequential training often achieving the best results.

## 🔬 About IA2

**In-Context Activation Alignment (IA2)** is a novel training method that improves in-context learning by aligning model activations with those produced during in-context learning. The method:

- **Core Idea**: Train models to produce similar internal representations (activations) as when performing in-context learning
- **Training Objective**: MSE loss between model activations and target ICL activations
- **Key Insight**: By aligning internal representations, models can better leverage in-context learning capabilities
- **Advantages**: More efficient than traditional fine-tuning, better generalization, improved ICL performance

The method is particularly effective when combined with supervised fine-tuning (SFT) or used in sequential training approaches.

## 🤝 Contributing

When adding new features:
1. Follow the unified naming conventions (`tok`/`act`/`tna`)
2. Ensure compatibility with existing directory structures
3. Add appropriate argument validation and error handling
4. Update this README with new functionality

## 📝 License

[Add your license information here]
