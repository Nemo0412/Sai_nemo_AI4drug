"""
Evaluate all checkpoints and compute accuracy metrics
This script evaluates each checkpoint saved during training and saves accuracy results
"""

import os
import json
import torch
import yaml
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def load_checkpoint_model(base_model_path, checkpoint_path):
    """Load model from checkpoint"""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA weights from checkpoint
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    return model, tokenizer


def extract_toxicity(response):
    """Extract toxicity label (0 or 1) from response"""
    toxicity = 0
    
    tox_match = re.search(r'Toxicity value:\s*(\d+)', response)
    if tox_match:
        toxicity = int(tox_match.group(1))
        toxicity = 1 if toxicity >= 1 else 0
    
    if 'non-toxic' in response.lower():
        toxicity = 0
    elif 'toxic' in response.lower() and 'non-toxic' not in response.lower():
        toxicity = 1
    
    return 1 if toxicity >= 1 else 0


def extract_efficiency(response):
    """Extract efficiency score (1-10) from response"""
    eff_patterns = [
        r'[Ee]fficiency [Ss]core[:\s]+(\d+)',
        r'[Ee]fficiency[:\s]+(\d+)',
        r'[Ss]core[:\s]+(\d+)',
        r'predicted score[:\s]+(\d+)',
        r'is (\d+)',
    ]
    
    for pattern in eff_patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            score = max(1, min(10, score))
            return score
    
    return None


def predict_toxicity_and_efficiency(model, tokenizer, smiles, max_length=512):
    """Predict both toxicity and efficiency for a SMILES string"""
    prompt = f"Analyze this lipid molecule for LNP delivery systems: {smiles}"
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert in lipid nanoparticle (LNP) delivery systems and drug discovery. You analyze lipid molecules for their toxicity and delivery efficiency, providing detailed reasoning based on molecular structure, functional groups, and known properties."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    toxicity = extract_toxicity(response)
    efficiency = extract_efficiency(response)
    
    return toxicity, efficiency, response


def load_test_data(toxic_test_path, efficiency_test_path):
    """Load test data from JSONL files"""
    test_data = []
    
    # Load toxic test data
    if os.path.exists(toxic_test_path):
        with open(toxic_test_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                messages = data.get('messages', [])
                if len(messages) >= 3:
                    user_msg = messages[1].get('content', '')
                    assistant_msg = messages[2].get('content', '')
                    
                    # Extract SMILES
                    smiles_match = re.search(r':\s*([A-Za-z0-9@\[\]()=#\\\/\-\+\.,;:]+)', user_msg)
                    if smiles_match:
                        smiles = smiles_match.group(1).strip()
                        
                        # Extract true labels
                        tox_match = re.search(r'Toxicity value:\s*(\d+)', assistant_msg)
                        true_toxicity = int(tox_match.group(1)) if tox_match else 0
                        true_toxicity = 1 if true_toxicity >= 1 else 0
                        
                        test_data.append({
                            'smiles': smiles,
                            'true_toxicity': true_toxicity,
                            'true_efficiency': None,  # Toxic data doesn't have efficiency
                            'type': 'toxic'
                        })
    
    # Load efficiency test data
    if os.path.exists(efficiency_test_path):
        with open(efficiency_test_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                messages = data.get('messages', [])
                if len(messages) >= 3:
                    user_msg = messages[1].get('content', '')
                    assistant_msg = messages[2].get('content', '')
                    
                    # Extract SMILES
                    smiles_match = re.search(r':\s*([A-Za-z0-9@\[\]()=#\\\/\-\+\.,;:]+)', user_msg)
                    if smiles_match:
                        smiles = smiles_match.group(1).strip()
                        
                        # Extract true labels
                        tox_match = re.search(r'Toxicity value:\s*(\d+)', assistant_msg)
                        eff_match = re.search(r'[Ee]fficiency [Ss]core[:\s]+(\d+)|[Ee]fficiency[:\s]+(\d+)|[Ss]core[:\s]+(\d+)', assistant_msg)
                        
                        true_toxicity = int(tox_match.group(1)) if tox_match else 0
                        true_toxicity = 1 if true_toxicity >= 1 else 0
                        
                        if eff_match:
                            true_efficiency = int(eff_match.group(1) or eff_match.group(2) or eff_match.group(3))
                            true_efficiency = max(1, min(10, true_efficiency))
                        else:
                            true_efficiency = None
                        
                        test_data.append({
                            'smiles': smiles,
                            'true_toxicity': true_toxicity,
                            'true_efficiency': true_efficiency,
                            'type': 'efficiency'
                        })
    
    return test_data


def evaluate_checkpoint(base_model_path, checkpoint_path, test_data, output_file):
    """Evaluate a single checkpoint and save results"""
    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    
    # Load model
    model, tokenizer = load_checkpoint_model(base_model_path, checkpoint_path)
    
    # Predictions
    pred_toxicity = []
    pred_efficiency = []
    true_toxicity = []
    true_efficiency = []
    
    # Evaluate on test data
    for item in tqdm(test_data, desc="Evaluating"):
        smiles = item['smiles']
        true_tox = item['true_toxicity']
        true_eff = item['true_efficiency']
        
        try:
            pred_tox, pred_eff, _ = predict_toxicity_and_efficiency(model, tokenizer, smiles)
            
            pred_toxicity.append(pred_tox)
            true_toxicity.append(true_tox)
            
            if true_eff is not None and pred_eff is not None:
                pred_efficiency.append(pred_eff)
                true_efficiency.append(true_eff)
        except Exception as e:
            print(f"Error predicting {smiles}: {e}")
            continue
    
    # Calculate metrics
    results = {
        'checkpoint': str(checkpoint_path),
        'step': extract_step_from_checkpoint(checkpoint_path),
    }
    
    # Toxicity accuracy
    if len(pred_toxicity) > 0:
        tox_accuracy = accuracy_score(true_toxicity, pred_toxicity)
        results['toxicity_accuracy'] = float(tox_accuracy)
        results['toxicity_samples'] = len(pred_toxicity)
    
    # Efficiency metrics
    if len(pred_efficiency) > 0:
        eff_accuracy = accuracy_score(true_efficiency, pred_efficiency)
        mae = mean_absolute_error(true_efficiency, pred_efficiency)
        rmse = np.sqrt(mean_squared_error(true_efficiency, pred_efficiency))
        
        # Within ±1 and ±2 accuracy
        within_1 = np.mean(np.abs(np.array(true_efficiency) - np.array(pred_efficiency)) <= 1)
        within_2 = np.mean(np.abs(np.array(true_efficiency) - np.array(pred_efficiency)) <= 2)
        
        results['efficiency_accuracy'] = float(eff_accuracy)
        results['efficiency_mae'] = float(mae)
        results['efficiency_rmse'] = float(rmse)
        results['efficiency_within_1'] = float(within_1)
        results['efficiency_within_2'] = float(within_2)
        results['efficiency_samples'] = len(pred_efficiency)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"  Toxicity Accuracy: {results.get('toxicity_accuracy', 'N/A'):.4f}")
    print(f"  Efficiency Accuracy: {results.get('efficiency_accuracy', 'N/A'):.4f}")
    print(f"  Efficiency MAE: {results.get('efficiency_mae', 'N/A'):.4f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def extract_step_from_checkpoint(checkpoint_path):
    """Extract step number from checkpoint path"""
    # Checkpoint path format: checkpoint-{step}
    match = re.search(r'checkpoint-(\d+)', str(checkpoint_path))
    if match:
        return int(match.group(1))
    return None


def find_all_checkpoints(output_dir):
    """Find all checkpoint directories"""
    checkpoints = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return checkpoints
    
    for item in output_path.iterdir():
        if item.is_dir() and 'checkpoint-' in item.name:
            checkpoints.append(item)
    
    # Sort by step number
    checkpoints.sort(key=lambda x: extract_step_from_checkpoint(x) or 0)
    
    return checkpoints


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints')
    parser.add_argument('--config', type=str, default='training_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory containing checkpoints (overrides config)')
    parser.add_argument('--results_dir', type=str, default='checkpoint_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    base_model_path = config['model']['base_model_path']
    output_dir = args.output_dir or config['model']['output_dir']
    toxic_test_path = config['data']['toxic_test_data_path']
    efficiency_test_path = config['data']['efficiency_test_data_path']
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(toxic_test_path, efficiency_test_path)
    print(f"Loaded {len(test_data)} test samples")
    
    # Find all checkpoints
    checkpoints = find_all_checkpoints(output_dir)
    print(f"\nFound {len(checkpoints)} checkpoints")
    
    # Evaluate each checkpoint
    all_results = []
    for checkpoint in checkpoints:
        step = extract_step_from_checkpoint(checkpoint)
        if step is None:
            continue
        
        result_file = results_dir / f"checkpoint-{step}_results.json"
        
        if result_file.exists():
            print(f"\nSkipping checkpoint-{step} (already evaluated)")
            with open(result_file, 'r') as f:
                results = json.load(f)
        else:
            results = evaluate_checkpoint(
                base_model_path,
                checkpoint,
                test_data,
                result_file
            )
        
        all_results.append(results)
    
    # Save summary
    summary_file = results_dir / "all_checkpoints_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal checkpoints evaluated: {len(all_results)}")
    print(f"\nResults saved to: {summary_file}")
    print(f"\nIndividual checkpoint results in: {results_dir}")
    
    # Print summary table
    if all_results:
        print("\nCheckpoint | Step | Tox Acc | Eff Acc | Eff MAE")
        print("-" * 60)
        for r in all_results:
            step = r.get('step', 'N/A')
            tox_acc = r.get('toxicity_accuracy', 0)
            eff_acc = r.get('efficiency_accuracy', 0)
            eff_mae = r.get('efficiency_mae', 0)
            print(f"{r['checkpoint'].split('/')[-1]:<15} | {step:<5} | {tox_acc:.4f} | {eff_acc:.4f} | {eff_mae:.4f}")


if __name__ == "__main__":
    main()

