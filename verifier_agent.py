"""
Verifier Agent for LipoAgent Multi-Agent Framework
Validates logical consistency between predicted scores and explanations
"""

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


class VerifierAgent:
    """
    Verifier Agent that inspects low-confidence predictions and validates
    logical consistency between predicted scores and explanations
    """
    
    def __init__(self, base_model_path, device="auto"):
        """
        Initialize Verifier Agent
        
        Args:
            base_model_path: Path to base Qwen 7B model (no finetuning needed)
            device: Device to run model on
        """
        self.device = device
        
        print(f"[Verifier Agent] Loading model from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side='right'
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.float16,
        )
        self.model.eval()
    
    def verify(self, smiles_structure, predicted_toxicity, predicted_efficiency, reasoning):
        """
        Verify logical consistency between predicted scores and reasoning
        
        Args:
            smiles_structure: SMILES string
            predicted_toxicity: Predicted toxicity score
            predicted_efficiency: Predicted efficiency score
            reasoning: Reasoning explanation
            
        Returns:
            dict: Verification result with consistency check and alternative scores if needed
        """
        prompt = f"""You are a verification expert for lipid nanoparticle (LNP) delivery systems.

Given the following prediction for molecule: {smiles_structure}

Predicted Toxicity: {predicted_toxicity}/10
Predicted Efficiency: {predicted_efficiency}/10
Reasoning: {reasoning}

Please verify:
1. Is the reasoning logically consistent with the predicted scores?
2. Are the scores reasonable given the molecular structure?
3. If there are inconsistencies, what would be more appropriate scores?

Respond in the following format:
Consistency: [YES/NO/PARTIAL]
Verification: [detailed analysis]
Suggested Toxicity: [score or N/A]
Suggested Efficiency: [score or N/A]
Confidence: [HIGH/MEDIUM/LOW]"""
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert verifier for drug discovery predictions. You check logical consistency between predicted scores and their explanations, identifying inconsistencies and suggesting corrections when needed."
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
        
        # Generate verification
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.3,
                top_p=0.9,
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse verification response
        verification_result = self._parse_verification(response)
        
        return {
            'smiles': smiles_structure,
            'is_consistent': verification_result['is_consistent'],
            'consistency_level': verification_result['consistency_level'],
            'verification_text': verification_result['verification_text'],
            'suggested_toxicity': verification_result['suggested_toxicity'],
            'suggested_efficiency': verification_result['suggested_efficiency'],
            'verification_confidence': verification_result['confidence'],
            'raw_response': response
        }
    
    def _parse_verification(self, response):
        """
        Parse verification response
        
        Args:
            response: Model response text
            
        Returns:
            dict: Parsed verification result
        """
        result = {
            'is_consistent': False,
            'consistency_level': 'UNKNOWN',
            'verification_text': '',
            'suggested_toxicity': None,
            'suggested_efficiency': None,
            'confidence': 'MEDIUM'
        }
        
        # Extract consistency
        consistency_patterns = [
            r'[Cc]onsistency[:\s]+(YES|NO|PARTIAL)',
            r'[Cc]onsistent[:\s]+(YES|NO|PARTIAL)',
        ]
        for pattern in consistency_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                level = match.group(1).upper()
                result['consistency_level'] = level
                result['is_consistent'] = (level == 'YES')
                break
        
        # Extract verification text
        verification_patterns = [
            r'[Vv]erification[:\s]+(.*?)(?:\n\n|Suggested|Confidence|\Z)',
            r'[Aa]nalysis[:\s]+(.*?)(?:\n\n|Suggested|Confidence|\Z)',
        ]
        for pattern in verification_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                result['verification_text'] = match.group(1).strip()
                break
        
        # Extract suggested scores
        suggested_toxicity_patterns = [
            r'[Ss]uggested\s+[Tt]oxicity[:\s]+(\d+(?:\.\d+)?|N/A)',
        ]
        for pattern in suggested_toxicity_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                val = match.group(1)
                if val.upper() != 'N/A':
                    try:
                        result['suggested_toxicity'] = float(val)
                    except ValueError:
                        pass
                break
        
        suggested_efficiency_patterns = [
            r'[Ss]uggested\s+[Ee]fficiency[:\s]+(\d+(?:\.\d+)?|N/A)',
        ]
        for pattern in suggested_efficiency_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                val = match.group(1)
                if val.upper() != 'N/A':
                    try:
                        result['suggested_efficiency'] = float(val)
                    except ValueError:
                        pass
                break
        
        # Extract confidence
        confidence_patterns = [
            r'[Cc]onfidence[:\s]+(HIGH|MEDIUM|LOW)',
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result['confidence'] = match.group(1).upper()
                break
        
        # If no verification text found, use the whole response
        if not result['verification_text']:
            result['verification_text'] = response.strip()
        
        return result
    
    def agrees_with_prediction(self, verification_result, tolerance=1.0):
        """
        Check if verifier agrees with the original prediction
        
        Args:
            verification_result: Verification result dictionary
            tolerance: Tolerance for score differences
            
        Returns:
            bool: True if verifier agrees
        """
        if verification_result['is_consistent']:
            return True
        
        # Check if suggested scores are close to original
        if verification_result['suggested_toxicity'] is not None:
            # If suggestions are very different, verifier disagrees
            return False
        
        # If consistency is PARTIAL, it's a partial agreement
        if verification_result['consistency_level'] == 'PARTIAL':
            return False
        
        return False

