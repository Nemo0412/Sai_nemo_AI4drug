"""
Predictor Agent for LipoAgent Multi-Agent Framework
Simultaneously predicts lipid toxicity and delivery efficiency with reasoning
"""

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from confidence_calculator import ConfidenceCalculator


class PredictorAgent:
    """
    Predictor Agent that simultaneously predicts:
    - Lipid toxicity (1-10 scale, lower is better)
    - Delivery efficiency (1-10 scale, higher is better)
    - Generates textual reasoning
    - Estimates uncertainty
    """
    
    def __init__(self, base_model_path, lora_path, score_min=1, score_max=10, device="auto"):
        """
        Initialize Predictor Agent
        
        Args:
            base_model_path: Path to base Qwen 7B model
            lora_path: Path to finetuned LoRA weights
            score_min: Minimum score value
            score_max: Maximum score value
            device: Device to run model on
        """
        self.score_min = score_min
        self.score_max = score_max
        self.device = device
        
        print(f"[Predictor Agent] Loading model from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side='right'
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.float16,
        )
        
        print(f"[Predictor Agent] Loading LoRA weights from {lora_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.model.eval()
        
        self.confidence_calculator = ConfidenceCalculator(score_min, score_max)
    
    def predict(self, smiles_structure, num_samples=5, temperature=0.7, top_p=0.8, max_length=512):
        """
        Predict toxicity, efficiency, reasoning, and uncertainty for a molecule
        
        Args:
            smiles_structure: SMILES string
            num_samples: Number of samples for uncertainty estimation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_length: Maximum generation length
            
        Returns:
            dict: Prediction results with toxicity, efficiency, reasoning, and confidence
        """
        # Create comprehensive prompt for simultaneous prediction
        prompt = f"""Analyze this lipid molecule for LNP delivery systems: {smiles_structure}

Please provide:
1. Toxicity score (1-10, where 1 is least toxic and 10 is most toxic)
2. Delivery efficiency score (1-10, where 1 is least efficient and 10 is most efficient)
3. Detailed reasoning explaining both scores based on molecular structure, functional groups, and known properties.

Format your response as:
Toxicity: [score]
Efficiency: [score]
Reasoning: [detailed explanation]"""
        
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
        
        # Format conversation
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate multiple samples for uncertainty estimation
        toxicity_samples = []
        efficiency_samples = []
        reasoning_samples = []
        
        for _ in range(num_samples):
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse response
            parsed = self._parse_response(response)
            if parsed['toxicity'] is not None:
                toxicity_samples.append(parsed['toxicity'])
            if parsed['efficiency'] is not None:
                efficiency_samples.append(parsed['efficiency'])
            if parsed['reasoning']:
                reasoning_samples.append(parsed['reasoning'])
        
        # Calculate confidence and statistics
        toxicity_result = None
        efficiency_result = None
        
        if len(toxicity_samples) >= num_samples // 2:
            toxicity_result = self.confidence_calculator.calculate_confidence_from_samples(toxicity_samples)
        
        if len(efficiency_samples) >= num_samples // 2:
            efficiency_result = self.confidence_calculator.calculate_confidence_from_samples(efficiency_samples)
        
        # Use most common reasoning or combine them
        final_reasoning = self._combine_reasoning(reasoning_samples) if reasoning_samples else "No reasoning generated."
        
        return {
            'smiles': smiles_structure,
            'toxicity': {
                'score': toxicity_result['mean_score'] if toxicity_result else None,
                'confidence': toxicity_result['confidence'] if toxicity_result else 0.0,
                'std': toxicity_result['std_score'] if toxicity_result else None,
                'samples': toxicity_samples
            },
            'efficiency': {
                'score': efficiency_result['mean_score'] if efficiency_result else None,
                'confidence': efficiency_result['confidence'] if efficiency_result else 0.0,
                'std': efficiency_result['std_score'] if efficiency_result else None,
                'samples': efficiency_samples
            },
            'reasoning': final_reasoning,
            'overall_confidence': min(
                toxicity_result['confidence'] if toxicity_result else 0.0,
                efficiency_result['confidence'] if efficiency_result else 0.0
            ),
            'num_samples': len(toxicity_samples)
        }
    
    def _parse_response(self, response):
        """
        Parse model response to extract toxicity, efficiency, and reasoning
        
        Args:
            response: Model response text
            
        Returns:
            dict: Parsed values
        """
        result = {
            'toxicity': None,
            'efficiency': None,
            'reasoning': ''
        }
        
        # Extract toxicity score
        toxicity_patterns = [
            r'[Tt]oxicity[:\s]+(\d+(?:\.\d+)?)',
            r'[Tt]oxic[:\s]+(\d+(?:\.\d+)?)',
        ]
        for pattern in toxicity_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))
                    if self.score_min <= score <= self.score_max:
                        result['toxicity'] = score
                        break
                except (ValueError, IndexError):
                    continue
        
        # Extract efficiency score
        efficiency_patterns = [
            r'[Ee]fficiency[:\s]+(\d+(?:\.\d+)?)',
            r'[Ee]fficient[:\s]+(\d+(?:\.\d+)?)',
            r'[Dd]elivery[:\s]+(\d+(?:\.\d+)?)',
        ]
        for pattern in efficiency_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))
                    if self.score_min <= score <= self.score_max:
                        result['efficiency'] = score
                        break
                except (ValueError, IndexError):
                    continue
        
        # Extract reasoning
        reasoning_patterns = [
            r'[Rr]easoning[:\s]+(.*?)(?:\n\n|\Z)',
            r'[Ee]xplanation[:\s]+(.*?)(?:\n\n|\Z)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                result['reasoning'] = match.group(1).strip()
                break
        
        # If no reasoning found, use the whole response after scores
        if not result['reasoning']:
            # Try to extract text after the scores
            lines = response.split('\n')
            reasoning_lines = []
            found_scores = False
            for line in lines:
                if any(keyword in line.lower() for keyword in ['reasoning', 'explanation', 'because', 'due to']):
                    found_scores = True
                if found_scores or (result['toxicity'] is None and result['efficiency'] is None):
                    reasoning_lines.append(line)
            result['reasoning'] = '\n'.join(reasoning_lines).strip()
        
        return result
    
    def _combine_reasoning(self, reasoning_samples):
        """
        Combine multiple reasoning samples into a final reasoning
        
        Args:
            reasoning_samples: List of reasoning strings
            
        Returns:
            str: Combined reasoning
        """
        if not reasoning_samples:
            return "No reasoning available."
        
        # Use the longest/most detailed reasoning
        return max(reasoning_samples, key=len)
    
    def is_low_confidence(self, prediction, confidence_threshold=6.0):
        """
        Check if prediction has low confidence
        
        Args:
            prediction: Prediction result dictionary
            confidence_threshold: Confidence threshold
            
        Returns:
            bool: True if low confidence
        """
        return prediction['overall_confidence'] < confidence_threshold

