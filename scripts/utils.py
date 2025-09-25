import re


def parse_answer_sciqa(text):
    try:
        return text.split('####')[-1].strip()
    except:
        return ""

# Function to parse the answer from the generated output
def parse_answer_boxed(generated_output):
    # Assuming the answer is in the format \boxed{answer}
    match = re.search(r'\\boxed{(.*?)}', generated_output)
    if match:
        # Use a regex to capture everything inside the outer brackets, including nested ones
        inner_match = re.search(r'\\boxed{(.*)}', generated_output)
        if inner_match:
            answer = inner_match.group(1)
            
            # Remove all spaces inside the \boxed{} element for uniformity
            answer = answer.replace(' ', '')
            
            # Normalize \dfrac{} to \frac{} for uniformity
            answer = re.sub(r'\\dfrac{', r'\\frac{', answer)
            
            return answer
    return None

def get_model_name(training_method, model_name, lora_type, lora_r, lora_alpha, 
                   num_generated_tokens, num_train_examples, lr, run_idx, 
                   label_type=None, ce_loss_weight=None, ia3_type=None, num_virtual_tokens=None,
                   ldr_mode=False, num_labelled_samples=None, num_unlabelled_samples=None, max_permutations=None):
    """Generate standardized model name based on training method and parameters"""
    # Parse training method to extract base method and variant
    if '-' in training_method:
        base_method, training_variant = training_method.split('-', 1)
    else:
        base_method = 'lora'
        training_variant = training_method

    # Add LDR parameters to base name if in LDR mode
    ldr_suffix = ""
    if ldr_mode and num_labelled_samples is not None and num_unlabelled_samples is not None and max_permutations is not None:
        ldr_suffix = f"_ldr{num_labelled_samples}_{num_unlabelled_samples}_{max_permutations}"

    # Handle existing LoRA methods (keep unchanged)
    if training_method in ["tok", "a2t", "t2a"]:
        base_name = f"{model_name}_{lora_type}_{lora_r}_{lora_alpha}_{num_generated_tokens}_{num_train_examples}_{lr}_{run_idx}{ldr_suffix}"
        return f"{base_name}_{label_type}"
    elif training_method == "act":
        base_name = f"{model_name}_{lora_type}_{lora_r}_{lora_alpha}_{num_generated_tokens}_{num_train_examples}_{lr}_{run_idx}{ldr_suffix}"
        return base_name
    elif training_method == "tna":
        base_name = f"{model_name}_{lora_type}_{lora_r}_{lora_alpha}_{num_generated_tokens}_{num_train_examples}_{lr}_{run_idx}{ldr_suffix}"
        return f"{base_name}_{ce_loss_weight}"
    
    # Handle new method variants
    elif base_method == "ia3":
        base_name = f"{model_name}_ia3_{ia3_type}_{num_generated_tokens}_{num_train_examples}_{lr}_{run_idx}{ldr_suffix}"
        if training_variant in ["tok", "a2t", "t2a"]:
            return f"{base_name}_{label_type}"
        elif training_variant == "act":
            return base_name
        elif training_variant == "tna":
            return f"{base_name}_{ce_loss_weight}"
    elif base_method in ["prompt", "prefix"]:
        base_name = f"{model_name}_{base_method}_{num_virtual_tokens}_{num_generated_tokens}_{num_train_examples}_{lr}_{run_idx}{ldr_suffix}"
        if training_variant in ["tok", "a2t", "t2a"]:
            return f"{base_name}_{label_type}"
        elif training_variant == "act":
            return base_name
        elif training_variant == "tna":
            return f"{base_name}_{ce_loss_weight}"
    else:
        raise ValueError(f"Unknown training method: {training_method}")

def construct_results_path(args):
    """Construct the path for evaluation results."""
    model_name_base = args.model_id.split('/')[-1]
    
    # Parse model_type to extract base method and training variant
    if '-' in args.model_type:
        base_method, training_variant = args.model_type.split('-', 1)
    else:
        base_method = 'lora'
        training_variant = args.model_type
    
    # Base results directory structure  
    suffix = "_uncertainty" if args.uncertainty_analysis else ""
    
    # Add LDR suffix if in LDR mode
    ldr_suffix = ""
    if hasattr(args, 'ldr_mode') and args.ldr_mode and hasattr(args, 'num_labelled_samples') and args.num_labelled_samples is not None:
        ldr_suffix = f"_ldr{args.num_labelled_samples}_{args.num_unlabelled_samples}_{args.max_permutations}"
    
    if args.model_type == 'base':
        if not args.eval_with_icl:
            return construct_base_without_icl_path(args)
        results_dir = f"{args.base_output_dir}/evaluations/base{suffix}/{args.eval_dataset_name}"
        results_filename = f"{model_name_base}_base_on_{args.icl_source_dataset}_demos{args.icl_max_demos}_T{args.num_generated_tokens_eval}_N{args.num_train_examples}_{args.run_idx}"
    else:
        results_dir = f"{args.base_output_dir}/evaluations/{args.model_type}{suffix}/{args.trained_dataset}"
        
        # Construct filename with standardized format
        base_filename = f"{model_name_base}_{args.eval_dataset_name}_on_{args.icl_source_dataset}_demos{args.icl_max_demos}"
        
        # Handle existing LoRA methods (keep unchanged)
        if args.model_type in ['tok', 'a2t']:
            results_filename = f"{base_filename}_{args.lora_type}_{args.lora_r}_{args.lora_alpha}_{args.label_type}_{args.num_generated_tokens_eval}_{args.num_train_examples}_{args.lr}_{args.run_idx}{ldr_suffix}"
            
        elif args.model_type in ['act', 't2a']:
            results_filename = f"{base_filename}_{args.lora_type}_r{args.lora_r}_a{args.lora_alpha}_{args.num_generated_tokens_eval}_{args.num_train_examples}_{args.lr}_{args.run_idx}{ldr_suffix}"
            
        elif args.model_type == 'tna':
            results_filename = f"{base_filename}_{args.lora_type}_r{args.lora_r}_a{args.lora_alpha}_{args.num_generated_tokens_eval}tok_{args.num_train_examples}ex_{args.lr}lr_cew{args.ce_loss_weight}_{args.run_idx}{ldr_suffix}"
        
        # Handle new method variants
        elif base_method == 'ia3':
            if training_variant in ['tok', 'a2t']:
                results_filename = f"{base_filename}_ia3_{args.ia3_type}_{args.label_type}_{args.num_generated_tokens_eval}_{args.num_train_examples}_{args.lr}_{args.run_idx}{ldr_suffix}"
            elif training_variant == 'act':
                results_filename = f"{base_filename}_ia3_{args.ia3_type}_{args.num_generated_tokens_eval}_{args.num_train_examples}_{args.lr}_{args.run_idx}{ldr_suffix}"
            elif training_variant == 'tna':
                results_filename = f"{base_filename}_ia3_{args.ia3_type}_{args.num_generated_tokens_eval}tok_{args.num_train_examples}ex_{args.lr}lr_cew{args.ce_loss_weight}_{args.run_idx}{ldr_suffix}"
        
        elif base_method in ['prompt', 'prefix']:
            if training_variant in ['tok', 'a2t']:
                results_filename = f"{base_filename}_{base_method}_{args.num_virtual_tokens}_{args.label_type}_{args.num_generated_tokens_eval}_{args.num_train_examples}_{args.lr}_{args.run_idx}{ldr_suffix}"
            elif training_variant == 'act':
                results_filename = f"{base_filename}_{base_method}_{args.num_virtual_tokens}_{args.num_generated_tokens_eval}_{args.num_train_examples}_{args.lr}_{args.run_idx}{ldr_suffix}"
            elif training_variant == 'tna':
                results_filename = f"{base_filename}_{base_method}_{args.num_virtual_tokens}_{args.num_generated_tokens_eval}tok_{args.num_train_examples}ex_{args.lr}lr_cew{args.ce_loss_weight}_{args.run_idx}{ldr_suffix}"
        
        else:
            raise ValueError(f"Unknown model_type: {args.model_type}")
    
    return results_dir, results_filename


def construct_base_without_icl_path(args):
    """Construct the path for base model without_icl results."""
    model_name_base = args.model_id.split('/')[-1]
    
    # Base results directory structure  
    suffix = "_uncertainty" if args.uncertainty_analysis else ""
    results_dir = f"../outputs/evaluations/base{suffix}/{args.eval_dataset_name}"
    results_filename = f"{model_name_base}_base_without_icl_T{args.num_generated_tokens_eval}"
    
    return results_dir, results_filename


def parse_answer_gsm8k(text, answer_patterns=None):
    """
    Extract answer from text using various patterns.
    
    Args:
        text: The text to parse
        answer_patterns: List of regex patterns to try (optional)
    
    Returns:
        Extracted answer string
    """
    if not text:
        return ""

    try:
    
        # Clean the text - remove special tokens and normalize
        text = re.sub(r'<\|eot_id\|>', '', text)
        text = text.strip()
        
        # Default patterns to try (in order of preference)
        if answer_patterns is None:
            answer_patterns = [
                # LaTeX boxed format
                r'\\boxed\{([^}]+)\}',
                # #### format (most reliable)
                r'####\s*([^\n]+)',
                # Answer: format (prefer the last occurrence)
                r'Answer:\s*([^\n]+)',
                # The answer is format
                r'The answer is\s*([^\n]+)',
                # answer is format
                r'answer is\s*([^\n]+)',
                # Answer format (without colon)
                r'Answer\s+([^\n]+)',
                # answer format (without colon)
                r'answer\s+([^\n]+)',
                # Best answer format
                r'The best answer is:\s*([^\n]+)',
                # Dollar amount patterns
                r'\$\s*([0-9,]+(?:\.[0-9]+)?)',
                # Number with units at the end
                r'([0-9,]+(?:\.[0-9]+)?)\s*(?:miles?|meters?|hours?|minutes?|bolts?|glasses?|vacuum cleaners?|sheep|eggs?|cups?|downloads?|years?|weeks?|dollars?|GB|GB/minute)',
            ]
        
        # Try each pattern, but prefer later matches for certain patterns
        matches = []
        for pattern in answer_patterns:
            all_matches = re.findall(pattern, text, re.IGNORECASE)
            if all_matches:
                # For Answer: patterns, prefer the last occurrence
                matches.append(all_matches[-1])
        
        if matches:
            # Clean up the extracted answer
            answer = matches[0].strip()
            # Remove common artifacts and clean up the number
            answer = re.sub(r'^\$', '', answer)  # Remove leading $
            answer = re.sub(r'^\s*:\s*', '', answer)  # Remove leading colon
            answer = re.sub(r'^\s*\\boxed\{', '', answer)  # Remove LaTeX boxed start
            answer = re.sub(r'\}\s*$', '', answer)  # Remove LaTeX boxed end
            answer = re.sub(r'[.,]\s*$', '', answer)  # Remove trailing period or comma
            
            # Extract just the number from the cleaned answer
            number_match = re.search(r'([0-9,]+(?:\.[0-9]+)?)', answer)
            if number_match:
                clean_number = number_match.group(1).replace(',', '')
                try:
                    num_val = float(clean_number)
                    return str(int(num_val)) if num_val.is_integer() else str(num_val)
                except ValueError:
                    pass
            
            return answer
        
        # If no pattern matches, try to extract numbers with better handling
        # Look for numbers that might be answers (avoid years, percentages, etc.)
        number_patterns = [
            # Numbers with common units
            r'([0-9,]+(?:\.[0-9]+)?)\s*(?:miles?|meters?|hours?|minutes?|bolts?|glasses?|vacuum cleaners?|sheep|eggs?|cups?|downloads?|years?|weeks?|dollars?|GB|GB/minute)\b',
            # Dollar amounts
            r'\$\s*([0-9,]+(?:\.[0-9]+)?)',
            # Standalone numbers (with commas)
            r'\b([0-9,]+(?:\.[0-9]+)?)\b'
        ]
        
        for pattern in number_patterns:
            numbers = re.findall(pattern, text, re.IGNORECASE)
            if numbers:
                # Prefer the last number that's not clearly a year (1900-2100) or percentage
                for num in reversed(numbers):
                    try:
                        clean_num = num.replace(',', '')
                        num_val = float(clean_num)
                        # Skip years, percentages, and very small decimals that look like rates
                        if not (1900 <= num_val <= 2100 and num_val.is_integer()) and not (0 < num_val < 1):
                            return str(int(num_val)) if num_val.is_integer() else str(num_val)
                    except ValueError:
                        continue
        
        # If still no match, try to find any number at the end of meaningful lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in reversed(lines):
            # Skip lines that are clearly not answers (too long, have question marks, etc.)
            if len(line) > 200 or '?' in line:
                continue
                
            numbers = re.findall(r'([0-9,]+(?:\.[0-9]+)?)', line)
            if numbers:
                try:
                    clean_num = numbers[-1].replace(',', '')
                    num_val = float(clean_num)
                    # Skip years and very small decimals
                    if not (1900 <= num_val <= 2100 and num_val.is_integer()) and not (0 < num_val < 1):
                        return str(int(num_val)) if num_val.is_integer() else str(num_val)
                except ValueError:
                    continue
        
        # If no number found, return empty string
        return text
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return text