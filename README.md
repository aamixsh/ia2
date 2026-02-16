# IA2: ICL Activation Alignment (ICLR 2026, Rio)

This repository contains the code for the paper "IA2: Alignment with ICL activations improves SFT", a comprehensive framework for training and evaluating neural networks activation alignment.

## 🔬 About IA2

**ICL Activation Alignment (IA2)** is a novel training method that improves ICL learning by aligning model activations with those produced during ICL learning. The method:

- **Core Idea**: Train models to produce similar internal representations (activations) as when performing ICL learning
- **Training Objective**: MSE loss between model activations and target ICL activations
- **Key Insight**: By aligning internal representations, models can better leverage ICL learning capabilities
- **Advantages**: More efficient than traditional fine-tuning, better generalization, improved ICL performance

The method is particularly effective when combined with supervised fine-tuning (SFT) or used in sequential training approaches.

## 🌟 Overview

This project implements and compares the IA2 method and related training approaches. **Training variants** (objectives) are available with multiple **adapter types**: LoRA, IA3, Prompt Tuning, and Prefix Tuning. Use `{adapter}-{variant}` for non-LoRA adapters (e.g. `ia3-tok`, `prompt-act`); LoRA uses the variant name only (e.g. `tok`, `act`).

**Training variants (same names across adapters):**

| Variant | Description |
|---------|-------------|
| **`tok`** | SFT: Supervised Fine-Tuning with cross-entropy on ground truth or ICL output tokens |
| **`tokl`** | Soft-label SFT: Match ICL output *distributions* (soft labels) via KL/CE instead of hard token labels |
| **`act`** | IA2: ICL Activation Alignment — MSE loss to imitate ICL activations |
| **`tna`** | IA2 + SFT: Combined MSE and CE loss training |
| **`a2t`** | Sequential: IA2 → SFT |
| **`t2a`** | Sequential: SFT → IA2 |

**Adapter types:** LoRA (`tok`, `act`, …), IA3 (`ia3-tok`, `ia3-tokl`, `ia3-act`, …), Prompt Tuning (`prompt-tok`, `prompt-tokl`, `prompt-act`, …), Prefix Tuning (`prefix-tok`, `prefix-tokl`, `prefix-act`, …). Each adapter type may require extra args (e.g. `--ia3_type`, `--num_virtual_tokens`).

The system also supports **base model evaluation** for direct comparison without any adapter training.

## 📁 Project Structure

All scripts live in the `scripts/` directory. Run commands from the project root (e.g. `python scripts/train_unified.py ...`) or from inside `scripts/` (e.g. `python train_unified.py ...`).

```
scripts/
├── train_unified.py                      # 🚀 Unified training script (all methods)
├── train_all_unified.py                  # 🔥 Batch training script
├── evaluate_batch_optimized.py           # 📊 Optimized batch evaluation script
├── plot_unified.py                       # 📉 Unified plotting script
├── plot_all_unified.py                   # 🎨 Batch plotting script
├── plot_activation_similarity_vs_performance.py  # 📈 Activation similarity vs performance
├── plot_base_methods_comparison.py       # 📊 Base methods comparison
├── plot_subspace_overlap_overall.py      # 🔍 Subspace overlap overall
├── data.py                               # 📦 Dataset utilities
├── utils.py                              # 🛠️ Common utilities
├── remap_utils.py                        # 🗺️ Dataset remap utilities
├── prepare_data.py                       # 📥 Raw data preparation
├── create_training_datasets.py           # 📦 Single-dataset training set creation
├── create_all_training_datasets.py       # 📦 Batch training set creation
├── discover_all_results.py               # 🔎 Discover and aggregate results
├── reparse_evaluations.py                # 🔄 Re-parse evaluation outputs
├── run_table_generator.py                # 📋 Table generation runner
├── run.py                                # ▶️ Generic run entrypoint
├── generate_latex_tables.py               # 📄 LaTeX table generation
├── subspace_overlap_analysis.py          # 🔍 Subspace overlap analysis
└── activation_similarity_analysis.py     # 📊 Activation similarity analysis
```

## 🚀 Quick Start

### 0. Environment Setup

**Install the conda environment:**
```bash
# Create conda environment from ia2.yaml
conda env create -p <path_to_envs>/ia2 -f ia2.yaml

# Activate the environment
conda activate <path_to_envs>/ia2
```

**Alternative installation:**
```bash
# Install in default conda environments directory
conda env create -f ia2.yaml

# Activate the environment
conda activate ia2
```

### 1. Dataset Preparation

Before training models, you need to prepare the datasets:

**Step 1: Prepare raw datasets**
```bash
# Prepare datasets for different tasks
python prepare_data.py --dataset gsm8k --num_train_samples 2000 --num_val_samples 500
python prepare_data.py --dataset sst2 --num_train_samples 2000 --num_val_samples 500
python prepare_data.py --dataset sciqa --num_train_samples 2000 --num_val_samples 500
```

**Step 2: Create training datasets**
```bash
# Create training datasets for all configurations
python create_all_training_datasets.py --datasets gsm8k sst2 sciqa --num_train_examples 100 200 --num_runs 3 --max_icl_demos 5 --num_dev_examples 50
```

### Supported Datasets

The framework supports the following datasets:
- **Math**: `gsm8k`, `gsm8ks`, `hmath_algebra`
- **Science**: `sciqa`, `sciq_remap`, `qasc_remap`
- **Language**: `sst2`, `poems`, `finsen`, `agnews`, `bbcnews`, `strategytf`

**Example dataset preparation for different domains:**
```bash
# Math datasets
python prepare_data.py --dataset gsm8k --num_train_samples 2000 --num_val_samples 500
python prepare_data.py --dataset cmath --num_train_samples 2000 --num_val_samples 500 --subset algebra

# Science datasets  
python prepare_data.py --dataset sciqa --num_train_samples 2000 --num_val_samples 500
python prepare_data.py --dataset qasc --num_train_samples 2000 --num_val_samples 500

# Language datasets
python prepare_data.py --dataset sst2 --num_train_samples 2000 --num_val_samples 500
python prepare_data.py --dataset agnews --num_train_samples 2000 --num_val_samples 500
```

### 2. Training Models

All methods use `train_unified.py` with `--training_method <method>`. **LoRA** is the default adapter; for IA3, Prompt Tuning, or Prefix Tuning use the same variant with the prefix (e.g. `ia3-tok`, `prompt-act`) and add adapter-specific args: `--ia3_type` (IA3), `--num_virtual_tokens` (Prompt/Prefix).

**LoRA examples (core variants):**
```bash
# SFT (hard labels: ground_truth or icl_outputs)
python train_unified.py --training_method tok --dataset gsm8k --label_type icl_outputs --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0

# Soft-label SFT (tokl): match ICL output distributions; requires label_type=icl_outputs, optional --tokl_top_k
python train_unified.py --training_method tokl --dataset gsm8k --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0 --tokl_top_k all

# IA2 (activation alignment)
python train_unified.py --training_method act --dataset gsm8k --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0

# IA2 + SFT combined
python train_unified.py --training_method tna --dataset gsm8k --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0 --ce_loss_weight 0.002

# Sequential IA2 → SFT: train act first, then a2t (loads the act checkpoint automatically)
python train_unified.py --training_method act --dataset gsm8k --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0
python train_unified.py --training_method a2t --dataset gsm8k --label_type icl_outputs --lora_type qkv --lora_r 8 --lora_alpha 8 --num_generated_tokens 1 --num_train_examples 100 --lr 1e-4 --run_idx 0
```

**Other adapters:** Use the same variant with the adapter prefix and the appropriate flag:
- **IA3:** `--training_method ia3-tok` (or `ia3-tokl`, `ia3-act`, etc.) and `--ia3_type qkv`
- **Prompt Tuning:** `--training_method prompt-tok` (or `prompt-tokl`, `prompt-act`, etc.) and `--num_virtual_tokens 20`
- **Prefix Tuning:** `--training_method prefix-tok` (or `prefix-tokl`, `prefix-act`, etc.) and `--num_virtual_tokens 20`

**Batch training:**
```bash
# LoRA: multiple methods and optional sequential
python train_all_unified.py --training_methods tok tokl act tna --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0
python train_all_unified.py --training_methods tok act --include_sequential --sequential_first --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0

# Other adapters: same --training_methods pattern with adapter-specific args
python train_all_unified.py --training_methods ia3-tok ia3-act ia3-tna --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0 --ia3_types qkv
python train_all_unified.py --training_methods prompt-tok prompt-act prompt-tna --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0 --num_virtual_tokens 20
python train_all_unified.py --training_methods prefix-tok prefix-act prefix-tna --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0 --num_virtual_tokens 20

# Hyperparameter sweep
python train_all_unified.py --training_methods tok tokl act tna --datasets gsm8k --num_train_examples 100 200 --lrs 1e-4 5e-4 1e-3 --run_indices 0 1 2 --max_parallel 4
```

### 3. Evaluating Models

**Batch evaluation (recommended):**
```bash
# Evaluate all model types (include tokl for soft-label SFT)
python evaluate_batch_optimized.py --model_types tok tokl act tna base --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5

# Filter specific configurations (e.g. tokl with --tokl_top_k for specific runs)
python evaluate_batch_optimized.py --model_types tok tokl --lora_types qkv --num_examples 100 --lrs 1e-4 --run_indices 0 --uncertainty_analysis

# Uncertainty analysis
python evaluate_batch_optimized.py --model_types tok tokl act tna base --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5 --uncertainty_analysis
```

### 4. Generating Plots

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

Methods follow the pattern `{adapter}-{variant}` for IA3/Prompt/Prefix; LoRA uses the variant only. **Variants:** `tok` (SFT), `tokl` (soft-label SFT), `act` (IA2), `tna` (IA2+SFT), `a2t` (IA2→SFT), `t2a` (SFT→IA2).

| Method | Description | Required / notable arguments | Output Directory |
|--------|-------------|-----------------------------|------------------|
| **LoRA** | | | |
| `tok` | SFT: CE on hard labels | `--label_type` | `../outputs/tok/{dataset}/` |
| `tokl` | Soft-label SFT: match ICL output distributions (KL/CE) | `label_type=icl_outputs` only; optional `--tokl_top_k` | `../outputs/tokl/{dataset}/` |
| `act` | IA2: MSE on activations | — | `../outputs/act/{dataset}/` |
| `tna` | IA2 + SFT combined | `--ce_loss_weight` | `../outputs/tna/{dataset}/` |
| `a2t` | Sequential IA2 → SFT | `--label_type` for SFT phase | `../outputs/a2t/{dataset}/` |
| `t2a` | Sequential SFT → IA2 | — | `../outputs/t2a/{dataset}/` |
| **IA3** | Same variants with `ia3-` prefix | `--ia3_type` for all | `../outputs/ia3-{variant}/{dataset}/` |
| **Prompt Tuning** | Same variants with `prompt-` prefix | `--num_virtual_tokens` for all | `../outputs/prompt-{variant}/{dataset}/` |
| **Prefix Tuning** | Same variants with `prefix-` prefix | `--num_virtual_tokens` for all | `../outputs/prefix-{variant}/{dataset}/` |

### Model Naming Convention

**SFT models (`tok`):** `{model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}_{label_type}`  
Example: `Qwen3-4B-Base_qkv_8_8_1_100_0.0001_0_icl_outputs`

**Soft-label SFT (`tokl`):** Same as tok plus optional `_topk{K}` when `--tokl_top_k` is set (e.g. `..._icl_outputs_topk5`).

**IA2 models (`act`):** `{model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}`  
Example: `Qwen3-4B-Base_qkv_8_8_1_100_0.0001_0`

**IA2 + SFT models (`tna`):** `{model}_{lora_type}_{r}_{alpha}_{tokens}_{examples}_{lr}_{run}_{ce_weight}`  
Example: `Qwen3-4B-Base_qkv_8_8_1_100_0.0001_0_0.002`

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
├── tokl/{dataset}/                   # Soft-label SFT training models
├── act/{dataset}/                    # IA2 training models  
├── tna/{dataset}/                    # IA2 + SFT training models
├── a2t/{dataset}/                    # Sequential IA2 → SFT models
├── t2a/{dataset}/                    # Sequential SFT → IA2 models
└── evaluations/
    ├── base/                         # Base model evaluations
    ├── tok/{dataset}/                # SFT model evaluations
    ├── tokl/{dataset}/               # Soft-label SFT model evaluations
    ├── act/{dataset}/                # IA2 model evaluations
    ├── tna/{dataset}/                # IA2 + SFT model evaluations
    ├── base_uncertainty/             # Base model uncertainty evaluations
    ├── tok_uncertainty/{dataset}/    # SFT model uncertainty evaluations
    ├── tokl_uncertainty/{dataset}/   # Soft-label SFT uncertainty evaluations
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
| `--training_method` | str | Required | Training method: `tok`, `tokl`, `act`, `tna`, etc. (see Overview) |
| `--dataset` | str | `gsm8k` | Training dataset |
| `--model_id` | str | `Qwen/Qwen3-4B-Base` | Base model |
| `--lora_type` | str | `qkv` | LoRA target modules |
| `--lora_r` | int | `8` | LoRA rank |
| `--lora_alpha` | int | `8` | LoRA scaling parameter |
| `--num_generated_tokens` | int | `1` | Number of tokens to generate/train on |
| `--num_train_examples` | int | `100` | Number of training examples |
| `--lr` | float | `1e-4` | Learning rate |
| `--run_idx` | int | `0` | Run index for multiple runs |

### Method-Specific Arguments

**By variant (all adapter types):**
- **SFT (`tok`)**: `--label_type`: `ground_truth` or `icl_outputs`
- **Soft-label SFT (`tokl`)**: Uses ICL output distributions only (`label_type` fixed to `icl_outputs`). `--tokl_top_k`: top-K logits to store (`all` or integer, default `all`)
- **IA2 + SFT (`tna`)**: `--ce_loss_weight`: weight for CE loss (0–1)
- **Sequential (`a2t` / `t2a`)**: Handled automatically; `a2t` needs `--label_type` for the SFT phase

**By adapter type:**
- **IA3** (`ia3-*`): `--ia3_type` (e.g. `qkv`, `qko`, `qkvo`)
- **Prompt / Prefix** (`prompt-*`, `prefix-*`): `--num_virtual_tokens` (default: 20)

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_type` | str | Required | Model type: `tok`, `tokl`, `act`, `tna`, `base`, or adapter-prefixed (e.g. `ia3-tok`, `prompt-tokl`) |
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
# 1. Train models individually (tok, tokl, act, tna)
python train_unified.py --training_method tok --dataset gsm8k --label_type icl_outputs --num_train_examples 100 --lr 1e-4 --run_idx 0
python train_unified.py --training_method tokl --dataset gsm8k --num_train_examples 100 --lr 1e-4 --run_idx 0
python train_unified.py --training_method act --dataset gsm8k --num_train_examples 100 --lr 1e-4 --run_idx 0
python train_unified.py --training_method tna --dataset gsm8k --num_train_examples 100 --lr 1e-4 --run_idx 0 --ce_loss_weight 0.002

# 2. Evaluate models
python evaluate_batch_optimized.py --model_types base tok tokl act tna --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5

# 3. Generate plots  
python plot_unified.py --trained_dataset gsm8k --eval_dataset gsm8k --icl_source_dataset gsm8k --icl_max_demos 5 --model_types base tok tokl act tna
```

**Option 2: Batch Training (Recommended)**
```bash
# 1. Batch train all models with parallel processing
python train_all_unified.py --training_methods tok tokl act tna --datasets gsm8k --num_train_examples 100 --lrs 1e-4 --run_indices 0 --max_parallel 3

# 2. Batch evaluate all models
python evaluate_batch_optimized.py --model_types base tok tokl act tna --trained_datasets gsm8k --eval_datasets gsm8k --icl_source_datasets gsm8k --icl_max_demos 5

# 3. Generate plots
python plot_unified.py --trained_dataset gsm8k --eval_dataset gsm8k --icl_source_dataset gsm8k --icl_max_demos 5 --model_types base tok tokl act tna
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
python evaluate_batch_optimized.py --model_types base tok tokl act tna --uncertainty_analysis

# 2. Generate uncertainty plots
python plot_unified.py --trained_dataset gsm8k --eval_dataset gsm8k --icl_source_dataset gsm8k --icl_max_demos 5 --uncertainty_mode --plot_types all
```

### Hyperparameter Sweep Workflow

```bash
# 1. Comprehensive hyperparameter sweep across methods
python train_all_unified.py \
    --training_methods tok tokl act tna \
    --datasets gsm8k \
    --num_train_examples 50 100 200 \
    --lrs 5e-5 1e-4 5e-4 1e-3 \
    --run_indices 0 1 2 \
    --ce_loss_weights 0.001 0.002 0.005 \
    --max_parallel 4 \
    --wandb_log

# 2. Evaluate all trained models
python evaluate_batch_optimized.py \
    --model_types tok tokl act tna \
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
    --model_types tok tokl act tna
```

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



## 📚 Citation

If you use this code or find IA2 useful in your research, please cite:

```bibtex
@article{mishra2025ia2,
  title={{IA2: Alignment with ICL activations improves Supervised Fine-Tuning}},
  author={Mishra, Aayush and Khashabi, Daniel and Liu, Anqi},
  journal={arXiv preprint arXiv:2509.22621},
  year={2025}
}
```
