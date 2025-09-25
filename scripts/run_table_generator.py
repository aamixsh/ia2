import os

for uncertainty_mode in ['on', 'off']:
    for label_type in ['ground_truth', 'icl_outputs']:
        for metric_type in ['top1', 'label_set']:
            for model_type in ['llama', 'qwen']:
                for base_method in ['lora', 'ia3']:
                    if uncertainty_mode == 'off' and metric_type == 'label_set':
                        continue
                    training_methods = ['icl', 'sft', 'ia2', 'ia2_sft', 'tna']
                    if label_type == 'ground_truth':
                        training_methods = ['icl', 'sft', 'ia2', 'ia2_sft']
                    training_methods_str = ' '.join(training_methods)
                    os.system(f"python generate_latex_tables.py --uncertainty_mode {uncertainty_mode} --label_type {label_type} --metric_type {metric_type} --model_type {model_type} --base_method {base_method} --training_methods {training_methods_str}")