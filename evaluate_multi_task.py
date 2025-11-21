"""
Evaluate Multi-Task Model on both Toxicity and Efficiency prediction
"""

import json
import torch
import yaml
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_model(base_model_path, lora_path):
    """Load finetuned model"""
    print(f"Loading model from {base_model_path} with LoRA from {lora_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    return model, tokenizer


def extract_toxicity(response):
    """Extract toxicity label (0 or 1) from response - MUST be 0 or 1"""
    toxicity = 0
    
    tox_match = re.search(r'Toxicity value:\s*(\d+)', response)
    if tox_match:
        toxicity = int(tox_match.group(1))
        # Ensure it's 0 or 1
        toxicity = 1 if toxicity >= 1 else 0
    
    if 'non-toxic' in response.lower():
        toxicity = 0
    elif 'toxic' in response.lower() and 'non-toxic' not in response.lower():
        toxicity = 1
    
    # Final check: ensure output is strictly 0 or 1
    return 1 if toxicity >= 1 else 0


def extract_efficiency(response):
    """Extract efficiency score (1-10) from response - MUST be discrete integer 1-10"""
    eff_patterns = [
        r'[Ss]core[:\s]+(\d+)',
        r'[Ee]fficiency[:\s]+(\d+)',
        r'predicted score[:\s]+(\d+)',
        r'is (\d+)',
    ]
    
    for pattern in eff_patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))  # Convert to integer (discrete value)
            # Clamp to valid range [1, 10]
            score = max(1, min(10, score))
            return score
    
    return None


def predict_toxicity(model, tokenizer, smiles, max_length=512):
    """
    Predict toxicity for a SMILES string
    Output: discrete binary value (0 or 1 only)
    """
    prompt = f"Is this molecular structure toxic? {smiles}"
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular toxicity based on their SMILES structures."
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
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0.7,
            top_p=0.8,
        )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return extract_toxicity(response), response


def predict_efficiency(model, tokenizer, smiles, max_length=512):
    """
    Predict efficiency for a SMILES string
    Output: discrete integer value (1-10 only)
    """
    prompt = f"What is the predicted score for this molecular structure: {smiles}?"
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular scores based on their SMILES structures."
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
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0.7,
            top_p=0.8,
        )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return extract_efficiency(response), response


def evaluate_toxicity(model, tokenizer, test_data_path):
    """Evaluate toxicity prediction"""
    print("\n" + "="*80)
    print("EVALUATING TOXICITY PREDICTION")
    print("="*80)
    
    # Load test data
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Test samples: {len(test_data)}")
    
    predictions = []
    ground_truths = []
    
    for i, item in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(test_data)}...")
        
        smiles = item['messages'][1]['content'].split(': ')[-1].strip()
        assistant_msg = item['messages'][-1]['content']
        true_toxicity = extract_toxicity(assistant_msg)
        
        pred_toxicity, response = predict_toxicity(model, tokenizer, smiles)
        
        predictions.append(pred_toxicity)
        ground_truths.append(true_toxicity)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truths, predictions)
    
    print(f"\nToxicity Prediction Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(ground_truths, predictions, target_names=['Non-toxic', 'Toxic']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(ground_truths, predictions))
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'ground_truths': ground_truths
    }


def evaluate_efficiency(model, tokenizer, test_data_path):
    """Evaluate efficiency prediction"""
    print("\n" + "="*80)
    print("EVALUATING EFFICIENCY PREDICTION")
    print("="*80)
    
    # Load test data
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Test samples: {len(test_data)}")
    
    predictions = []
    ground_truths = []
    
    for i, item in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(test_data)}...")
        
        smiles = item['messages'][1]['content'].split(': ')[-1].strip()
        assistant_msg = item['messages'][-1]['content']
        true_efficiency = extract_efficiency(assistant_msg)
        
        if true_efficiency is None:
            continue
        
        pred_efficiency, response = predict_efficiency(model, tokenizer, smiles)
        
        if pred_efficiency is not None:
            predictions.append(pred_efficiency)
            ground_truths.append(true_efficiency)
    
    if len(predictions) == 0:
        print("No valid predictions made!")
        return None
    
    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    exact_match = (predictions == ground_truths).sum() / len(predictions)
    within_1 = (np.abs(predictions - ground_truths) <= 1).sum() / len(predictions)
    within_2 = (np.abs(predictions - ground_truths) <= 2).sum() / len(predictions)
    mae = np.abs(predictions - ground_truths).mean()
    rmse = np.sqrt(((predictions - ground_truths) ** 2).mean())
    
    print(f"\nEfficiency Prediction Results:")
    print(f"  Exact Match: {exact_match:.4f} ({exact_match*100:.2f}%)")
    print(f"  Within ±1: {within_1:.4f} ({within_1*100:.2f}%)")
    print(f"  Within ±2: {within_2:.4f} ({within_2*100:.2f}%)")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return {
        'exact_match': exact_match,
        'within_1': within_1,
        'within_2': within_2,
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions.tolist(),
        'ground_truths': ground_truths.tolist()
    }


def main():
    """Main evaluation function"""
    # Load config
    with open('training_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model, tokenizer = load_model(
        config['model']['base_model_path'],
        config['model']['output_dir']
    )
    
    # Evaluate toxicity
    toxic_results = evaluate_toxicity(
        model, 
        tokenizer, 
        config['data']['toxic_test_data_path']
    )
    
    # Evaluate efficiency
    efficiency_results = evaluate_efficiency(
        model,
        tokenizer,
        config['data']['efficiency_test_data_path']
    )
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nToxicity Prediction:")
    print(f"  Accuracy: {toxic_results['accuracy']:.4f}")
    print(f"\nEfficiency Prediction:")
    if efficiency_results:
        print(f"  Exact Match: {efficiency_results['exact_match']:.4f}")
        print(f"  Within ±1: {efficiency_results['within_1']:.4f}")
        print(f"  MAE: {efficiency_results['mae']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()

