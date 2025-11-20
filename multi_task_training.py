"""
Multi-Task Training with Conditional Loss
Trains model to predict both toxicity and efficiency simultaneously
"""

import json
import torch
import torch.nn.functional as F
import yaml
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import os
import re
import numpy as np


class MultiTaskSMILESDataset:
    """Dataset class for multi-task SMILES data (toxicity + efficiency)"""
    
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(jsonl_file)
    
    def load_data(self, jsonl_file):
        """Load data from JSONL file"""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def extract_labels(self, assistant_message):
        """
        Extract toxicity and efficiency labels from assistant message
        
        Expected format: "This molecular structure {smiles} is {toxic/non-toxic}. Toxicity value: {0/1}."
        For efficiency, we need to parse from the message or use a default
        """
        toxicity = 0
        efficiency = None  # None if not found (will skip efficiency loss)
        
        # Extract toxicity (0 or 1)
        tox_match = re.search(r'Toxicity value:\s*(\d+)', assistant_message)
        if tox_match:
            toxicity = int(tox_match.group(1))
        
        # Check for toxic/non-toxic keywords
        if 'non-toxic' in assistant_message.lower():
            toxicity = 0
        elif 'toxic' in assistant_message.lower() and 'non-toxic' not in assistant_message.lower():
            toxicity = 1
        
        # Extract efficiency if available
        eff_patterns = [
            r'Efficiency[:\s]+(\d+)',
            r'efficiency[:\s]+(\d+)',
            r'Score[:\s]+(\d+)',  # Alternative format
            r'score[:\s]+(\d+)',
        ]
        for pattern in eff_patterns:
            eff_match = re.search(pattern, assistant_message)
            if eff_match:
                efficiency = int(eff_match.group(1))
                # Validate range (1-10)
                if 1 <= efficiency <= 10:
                    break
                else:
                    efficiency = None
        
        return toxicity, efficiency
    
    def format_conversation(self, messages):
        """Format messages into a single string for training"""
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return formatted
    
    def preprocess_data(self):
        """Preprocess data for training"""
        processed = []
        for item in self.data:
            # Format conversation
            text = self.format_conversation(item['messages'])
            
            # Extract labels
            assistant_msg = item['messages'][-1]['content']  # Last message is assistant
            toxicity, efficiency = self.extract_labels(assistant_msg)
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            
            # Set labels for language modeling (standard causal LM)
            encoded['labels'] = encoded['input_ids'].copy()
            
            # Add multi-task labels
            encoded['toxicity_label'] = toxicity
            encoded['efficiency_label'] = efficiency
            
            processed.append(encoded)
        
        return Dataset.from_list(processed)


class MultiTaskTrainer(Trainer):
    """Custom Trainer with Conditional Multi-Task Loss"""
    
    def __init__(self, alpha=1.0, efficiency_num_classes=10, efficiency_eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.efficiency_num_classes = efficiency_num_classes
        self.efficiency_eps = efficiency_eps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute Conditional Multi-Task Loss:
        1. Toxicity loss: binary_cross_entropy_with_logits
        2. Efficiency loss: cross_entropy (only for non-toxic samples)
        3. Total loss: loss_tox + alpha * loss_eff
        """
        # Extract labels
        labels = inputs.pop("labels")
        toxicity_labels = inputs.pop("toxicity_label")
        efficiency_labels = inputs.pop("efficiency_label")
        
        # Forward pass with output_hidden_states to get hidden states
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Standard causal LM loss (for language modeling)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Get hidden states for the last non-padding token
        # Find the last non-padding token for each sequence
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            # Get indices of last non-padding tokens
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because of 0-indexing
            batch_size = hidden_states.size(0)
            last_token_hidden = hidden_states[torch.arange(batch_size), seq_lengths]
        else:
            # Fallback: use last token
            last_token_hidden = hidden_states[:, -1, :]
        
        # Project hidden states to classification logits
        # For toxicity: binary classification (single output)
        hidden_dim = last_token_hidden.size(-1)
        
        # Create simple linear projections (these could be added as model heads)
        # For now, we'll use a simple mean pooling approach
        # In production, you'd add proper classification heads to the model
        
        # Toxicity logits: project to 1 dimension
        toxicity_logits = last_token_hidden.mean(dim=-1, keepdim=True)
        
        # Efficiency logits: project to num_classes dimensions
        # Use a simple projection: take first num_classes dimensions and apply linear transform
        efficiency_projection = last_token_hidden[:, :self.efficiency_num_classes]
        # Add a learnable bias-like term by using mean of remaining dimensions
        if hidden_dim > self.efficiency_num_classes:
            efficiency_bias = last_token_hidden[:, self.efficiency_num_classes:].mean(dim=-1, keepdim=True)
            efficiency_logits = efficiency_projection + efficiency_bias
        else:
            efficiency_logits = efficiency_projection
        
        # Convert labels to tensors
        toxicity_labels_tensor = torch.tensor(toxicity_labels, device=logits.device, dtype=torch.float32)
        
        # 1. Toxicity loss: binary_cross_entropy_with_logits
        loss_tox = F.binary_cross_entropy_with_logits(
            toxicity_logits.squeeze(-1),
            toxicity_labels_tensor
        )
        
        # 2. Efficiency loss: cross_entropy (only for non-toxic samples)
        # Check if we have efficiency labels
        has_efficiency = any(eff is not None for eff in efficiency_labels)
        
        if has_efficiency:
            # Filter out None values and create valid efficiency labels
            efficiency_labels_tensor = torch.tensor(
                [eff if eff is not None else 5 for eff in efficiency_labels],  # Default to 5 if None
                device=logits.device,
                dtype=torch.long
            )
            
            # Create mask for non-toxic samples (toxicity == 0) AND valid efficiency labels
            valid_eff_mask = torch.tensor(
                [eff is not None for eff in efficiency_labels],
                device=logits.device,
                dtype=torch.float32
            )
            mask = ((toxicity_labels_tensor == 0) * valid_eff_mask).float()
            
            # Calculate cross entropy for all samples
            # Convert efficiency labels from 1-10 to 0-9 for cross_entropy
            efficiency_labels_0_indexed = (efficiency_labels_tensor - 1).clamp(0, self.efficiency_num_classes - 1)
            ce_all = F.cross_entropy(
                efficiency_logits,
                efficiency_labels_0_indexed,
                reduction='none'
            )
            
            # Apply mask and normalize (only for non-toxic samples with valid efficiency)
            loss_eff = (ce_all * mask).sum() / (mask.sum() + self.efficiency_eps)
        else:
            # No efficiency labels available, skip efficiency loss
            loss_eff = torch.tensor(0.0, device=logits.device)
            mask = torch.zeros_like(toxicity_labels_tensor)
        
        # 3. Total loss: loss_tox + alpha * loss_eff
        # Combine with language modeling loss
        total_loss = lm_loss + loss_tox + self.alpha * loss_eff
        
        # Log individual losses
        if hasattr(self.state, 'global_step') and self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'loss': total_loss.item(),
                'lm_loss': lm_loss.item(),
                'toxicity_loss': loss_tox.item(),
                'efficiency_loss': loss_eff.item(),
            })
        
        return (total_loss, outputs) if return_outputs else total_loss


def setup_lora_config(config):
    """Configure LoRA parameters from config"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
    )
    return lora_config


def train_multi_task_model(config_path="training_config.yaml"):
    """
    Train model with Conditional Multi-Task Loss
    
    Args:
        config_path: Path to training configuration file
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("MULTI-TASK TRAINING WITH CONDITIONAL LOSS")
    print("="*80)
    print(f"Configuration loaded from: {config_path}")
    print(f"Alpha (efficiency loss weight): {config['loss']['alpha']}")
    print()
    
    # Load tokenizer
    print(f"Loading tokenizer: {config['model']['base_model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model_path'],
        trust_remote_code=True,
        padding_side='right',
        revision="main"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {config['model']['base_model_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model_path'],
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=config['optimization']['load_in_8bit'],
        torch_dtype=torch.bfloat16 if config['optimization']['bf16'] else torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Prepare for kbit training
    if config['optimization']['load_in_8bit']:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print(f"\nLoading training data: {config['data']['train_data_path']}")
    dataset_loader = MultiTaskSMILESDataset(
        config['data']['train_data_path'],
        tokenizer,
        config['data']['max_length']
    )
    train_dataset = dataset_loader.preprocess_data()
    print(f"Training samples: {len(train_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        fp16=config['optimization']['fp16'],
        bf16=config['optimization']['bf16'],
        save_strategy=config['training']['save_strategy'],
        logging_steps=config['training']['logging_steps'],
        warmup_steps=config['training']['warmup_steps'],
        optim=config['optimization']['optim'],
        save_total_limit=config['training']['save_total_limit'],
        report_to="none",
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        max_grad_norm=config['training']['max_grad_norm'],
    )
    
    # Initialize custom trainer
    trainer = MultiTaskTrainer(
        alpha=config['loss']['alpha'],
        efficiency_num_classes=config['loss']['efficiency_num_classes'],
        efficiency_eps=config['loss']['efficiency_eps'],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\nStarting training...")
    print(f"Loss configuration:")
    print(f"  - Toxicity loss: {config['loss']['toxicity_loss_type']}")
    print(f"  - Efficiency loss: {config['loss']['efficiency_loss_type']} (alpha={config['loss']['alpha']})")
    print(f"  - Efficiency classes: {config['loss']['efficiency_num_classes']}")
    print()
    
    trainer.train()
    
    # Save model
    print(f"\nSaving model to: {config['model']['output_dir']}")
    model.save_pretrained(config['model']['output_dir'])
    tokenizer.save_pretrained(config['model']['output_dir'])
    
    print("\nTraining completed!")
    print("="*80)


if __name__ == "__main__":
    train_multi_task_model("training_config.yaml")

