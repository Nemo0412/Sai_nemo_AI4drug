"""
Multi-Task Training with Conditional Loss
Trains model to predict both toxicity and efficiency simultaneously
- Toxic data: for toxicity prediction (0 or 1)
- Efficiency data: for efficiency prediction (1-10 discrete values)
"""

import json
import torch
import torch.nn.functional as F
import yaml
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import os
import re
import numpy as np


class MultiTaskSMILESDataset:
    """Dataset class for multi-task SMILES data (toxicity + efficiency)"""
    
    def __init__(self, toxic_data_path, efficiency_data_path, tokenizer, max_length=512):
        """
        Initialize dataset with both toxic and efficiency data
        
        Args:
            toxic_data_path: Path to toxic data JSONL file (contains toxicity labels 0/1)
            efficiency_data_path: Path to efficiency data JSONL file (contains efficiency scores 1-10)
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.toxic_data = self.load_data(toxic_data_path) if toxic_data_path else []
        self.efficiency_data = self.load_data(efficiency_data_path) if efficiency_data_path else []
    
    def load_data(self, jsonl_file):
        """Load data from JSONL file"""
        if not os.path.exists(jsonl_file):
            print(f"Warning: {jsonl_file} not found, skipping...")
            return []
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def extract_toxicity_label(self, assistant_message):
        """Extract toxicity label (0 or 1) from assistant message - MUST be 0 or 1"""
        toxicity = 0
        
        # Extract toxicity (0 or 1)
        tox_match = re.search(r'Toxicity value:\s*(\d+)', assistant_message)
        if tox_match:
            toxicity = int(tox_match.group(1))
            # Ensure it's 0 or 1 (discrete binary value)
            toxicity = 1 if toxicity >= 1 else 0
        
        # Check for toxic/non-toxic keywords
        if 'non-toxic' in assistant_message.lower():
            toxicity = 0
        elif 'toxic' in assistant_message.lower() and 'non-toxic' not in assistant_message.lower():
            toxicity = 1
        
        # Final check: ensure output is strictly 0 or 1
        return 1 if toxicity >= 1 else 0
    
    def extract_efficiency_label(self, assistant_message):
        """Extract efficiency label (1-10) from assistant message - MUST be discrete integer 1-10"""
        efficiency = None
        
        # Extract efficiency/score
        eff_patterns = [
            r'[Ss]core[:\s]+(\d+)',
            r'[Ee]fficiency[:\s]+(\d+)',
            r'predicted score[:\s]+(\d+)',
            r'is (\d+)',  # "is 5" format
        ]
        for pattern in eff_patterns:
            eff_match = re.search(pattern, assistant_message)
            if eff_match:
                efficiency = int(eff_match.group(1))  # Convert to integer (discrete value)
                # Clamp to valid range [1, 10] and ensure it's an integer
                efficiency = max(1, min(10, efficiency))
                return efficiency
        
        return efficiency
    
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
        """Preprocess data for training - combine toxic and efficiency data"""
        processed = []
        
        # Process toxic data (for toxicity prediction)
        print(f"Processing {len(self.toxic_data)} toxic data samples...")
        for item in self.toxic_data:
            text = self.format_conversation(item['messages'])
            assistant_msg = item['messages'][-1]['content']
            toxicity = self.extract_toxicity_label(assistant_msg)
            
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            encoded['labels'] = encoded['input_ids'].copy()
            encoded['toxicity_label'] = toxicity
            encoded['efficiency_label'] = None  # No efficiency label for toxic-only data
            encoded['data_type'] = 'toxic'
            processed.append(encoded)
        
        # Process efficiency data (for efficiency prediction)
        print(f"Processing {len(self.efficiency_data)} efficiency data samples...")
        for item in self.efficiency_data:
            text = self.format_conversation(item['messages'])
            assistant_msg = item['messages'][-1]['content']
            efficiency = self.extract_efficiency_label(assistant_msg)
            
            if efficiency is None:
                continue  # Skip if efficiency label not found
            
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            encoded['labels'] = encoded['input_ids'].copy()
            encoded['toxicity_label'] = 0  # Assume non-toxic for efficiency data (will be masked in loss)
            encoded['efficiency_label'] = efficiency
            encoded['data_type'] = 'efficiency'
            processed.append(encoded)
        
        print(f"Total processed samples: {len(processed)}")
        print(f"  - Toxic samples: {len(self.toxic_data)}")
        print(f"  - Efficiency samples: {len([p for p in processed if p['data_type'] == 'efficiency'])}")
        
        return Dataset.from_list(processed)


class MultiTaskDataCollator:
    """Custom data collator that preserves toxicity_label and efficiency_label"""
    
    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None
        )
    
    def __call__(self, features):
        # Separate standard fields from custom fields
        # features is a list of dictionaries
        standard_features = []
        custom_fields = {
            'toxicity_label': [],
            'efficiency_label': [],
            'data_type': []
        }
        
        for f in features:
            # Create a copy of the feature without custom fields
            standard_feature = {}
            for key, value in f.items():
                if key in ['toxicity_label', 'efficiency_label', 'data_type']:
                    custom_fields[key].append(value)
                else:
                    standard_feature[key] = value
            standard_features.append(standard_feature)
        
        # Use default collator for standard fields (expects a list)
        batch = self.collator(standard_features)
        
        # Add custom fields back as lists (will be converted to tensors in compute_loss)
        for key, values in custom_fields.items():
            if values:  # Only add if not empty
                batch[key] = values
        
        return batch


class CheckpointEvaluationCallback(TrainerCallback):
    """Callback to evaluate model after each checkpoint save"""
    
    def __init__(self, eval_dataset, tokenizer, output_dir):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.results = []
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Called after each checkpoint save"""
        if state.global_step % args.save_steps == 0:
            print(f"\n{'='*80}")
            print(f"Evaluating checkpoint at step {state.global_step}")
            print(f"{'='*80}")
            
            # Evaluate checkpoint using evaluate_checkpoints.py
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            print(f"Checkpoint saved at: {checkpoint_path}")
            print(f"Run 'python evaluate_checkpoints.py' after training to evaluate all checkpoints")


class MultiTaskTrainer(Trainer):
    """Custom Trainer with Conditional Multi-Task Loss"""
    
    def __init__(self, alpha=1.0, efficiency_num_classes=10, efficiency_eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.efficiency_num_classes = efficiency_num_classes
        self.efficiency_eps = efficiency_eps
        self.accuracy_history = []  # Store accuracy for each checkpoint
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation based on classification heads
        Returns accuracy for both toxicity and efficiency predictions
        """
        # Note: This is called during evaluation, but we need to compute metrics
        # based on the actual model outputs. For now, return empty dict.
        # Full evaluation with text generation will be done by evaluate_checkpoints.py
        return {}
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation_loop to compute accuracy metrics during evaluation
        """
        # Call parent evaluation_loop
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Compute accuracy metrics using classification heads
        if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
            try:
                # Get predictions and labels from the last batch
                # Note: This is a simplified version. Full evaluation is done by evaluate_checkpoints.py
                metrics = output.metrics
                
                # Add placeholder metrics (actual metrics computed by evaluate_checkpoints.py)
                metrics[f"{metric_key_prefix}_toxicity_accuracy"] = 0.0
                metrics[f"{metric_key_prefix}_efficiency_accuracy"] = 0.0
                
                output.metrics = metrics
            except Exception as e:
                print(f"Warning: Could not compute accuracy metrics during evaluation: {e}")
        
        return output
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute Conditional Multi-Task Loss:
        1. Toxicity loss: binary_cross_entropy_with_logits (all samples)
           - Output: discrete binary value (0 or 1 only)
        2. Efficiency loss: cross_entropy (only for non-toxic samples)
           - Output: discrete integer value (1-10 only)
        3. Total loss: loss_tox + alpha * loss_eff
        """
        # Extract labels - use get() to handle missing keys gracefully
        labels = inputs.pop("labels", None)
        toxicity_labels = inputs.get("toxicity_label", None)
        efficiency_labels = inputs.get("efficiency_label", None)
        
        # Remove labels from inputs if they exist (to avoid passing to model)
        if "toxicity_label" in inputs:
            inputs.pop("toxicity_label")
        if "efficiency_label" in inputs:
            inputs.pop("efficiency_label")
        
        # Convert to tensors if they're not already
        if toxicity_labels is not None and not isinstance(toxicity_labels, torch.Tensor):
            toxicity_labels = torch.tensor(toxicity_labels, dtype=torch.float32)
        if efficiency_labels is not None and not isinstance(efficiency_labels, torch.Tensor):
            efficiency_labels = torch.tensor(efficiency_labels, dtype=torch.long)
        
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
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            last_token_hidden = hidden_states[torch.arange(batch_size), seq_lengths]
        else:
            last_token_hidden = hidden_states[:, -1, :]
        
        # Project hidden states to classification logits
        hidden_dim = last_token_hidden.size(-1)
        batch_size = last_token_hidden.size(0)
        
        # Toxicity logits: project to 1 dimension (binary classification: 0 or 1)
        # Output MUST be discrete: 0 or 1 only (binary classification)
        # Use mean pooling with learnable weight (simplified approach)
        # For binary classification, output shape: [batch_size, 1]
        toxicity_logits = last_token_hidden.mean(dim=-1, keepdim=True)  # Shape: [batch_size, 1]
        
        # Efficiency logits: project to num_classes dimensions (multi-class classification: 1-10)
        # Output MUST be discrete: integer values 1-10 only (10-class classification)
        # For 10-class classification, output shape: [batch_size, 10]
        # Use first 10 dimensions of hidden state + mean of remaining as bias
        if hidden_dim >= self.efficiency_num_classes:
            efficiency_logits = last_token_hidden[:, :self.efficiency_num_classes]  # [batch_size, 10]
            # Add bias from remaining dimensions
            if hidden_dim > self.efficiency_num_classes:
                bias = last_token_hidden[:, self.efficiency_num_classes:].mean(dim=-1, keepdim=True)
                efficiency_logits = efficiency_logits + bias
        else:
            # If hidden_dim < 10, pad with zeros
            efficiency_logits = last_token_hidden
            padding = torch.zeros(batch_size, self.efficiency_num_classes - hidden_dim, 
                                device=last_token_hidden.device, dtype=last_token_hidden.dtype)
            efficiency_logits = torch.cat([efficiency_logits, padding], dim=-1)
        
        # Convert labels to tensors - handle None values
        if toxicity_labels is None:
            # If no toxicity labels, create zeros (assume all non-toxic)
            toxicity_labels_tensor = torch.zeros(batch_size, device=logits.device, dtype=torch.float32)
        elif isinstance(toxicity_labels, (list, tuple)):
            toxicity_labels_tensor = torch.tensor(toxicity_labels, device=logits.device, dtype=torch.float32)
        else:
            toxicity_labels_tensor = toxicity_labels.to(device=logits.device, dtype=torch.float32)
        
        # Ensure batch size matches
        if toxicity_labels_tensor.dim() == 0:
            toxicity_labels_tensor = toxicity_labels_tensor.unsqueeze(0)
        if toxicity_labels_tensor.size(0) != batch_size:
            # If single value, expand to batch size
            if toxicity_labels_tensor.size(0) == 1:
                toxicity_labels_tensor = toxicity_labels_tensor.expand(batch_size)
            else:
                raise ValueError(f"Toxicity labels batch size mismatch: {toxicity_labels_tensor.size(0)} vs {batch_size}")
        
        # 1. Toxicity loss: binary_cross_entropy_with_logits (all samples)
        # Ensure labels are strictly 0 or 1
        toxicity_labels_tensor = toxicity_labels_tensor.clamp(0, 1)
        loss_tox = F.binary_cross_entropy_with_logits(
            toxicity_logits.squeeze(-1),
            toxicity_labels_tensor
        )
        
        # 2. Efficiency loss: cross_entropy (only for non-toxic samples)
        if efficiency_labels is None:
            has_efficiency = False
        elif isinstance(efficiency_labels, (list, tuple)):
            has_efficiency = any(eff is not None for eff in efficiency_labels)
        else:
            has_efficiency = True
        
        if has_efficiency:
            # Filter out None values and create valid efficiency labels
            # Ensure efficiency labels are discrete integers in range [1, 10]
            valid_efficiency_labels = []
            for eff in efficiency_labels:
                if eff is not None:
                    # Clamp to valid range [1, 10] and ensure integer
                    eff_int = int(max(1, min(10, eff)))
                    valid_efficiency_labels.append(eff_int)
                else:
                    valid_efficiency_labels.append(5)  # Default to 5 if None
            
            efficiency_labels_tensor = torch.tensor(
                valid_efficiency_labels,
                device=logits.device,
                dtype=torch.long
            )
            
            # Create mask for non-toxic samples (toxicity == 0) AND valid efficiency labels
            valid_eff_mask = torch.tensor(
                [eff is not None for eff in efficiency_labels],
                device=logits.device,
                dtype=torch.float32
            )
            # Only compute efficiency loss for non-toxic samples (toxicity == 0)
            mask = ((toxicity_labels_tensor == 0) * valid_eff_mask).float()
            
            # Calculate cross entropy for all samples
            # Convert efficiency labels from 1-10 to 0-9 for cross_entropy (discrete classes)
            # Ensure labels are in valid range [0, 9] for 10-class classification
            efficiency_labels_0_indexed = (efficiency_labels_tensor - 1).clamp(0, self.efficiency_num_classes - 1).long()
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
                'efficiency_loss': loss_eff.item() if isinstance(loss_eff, torch.Tensor) else loss_eff,
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
    Train model with Conditional Multi-Task Loss using both toxic and efficiency data
    
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
    
    # Prepare dataset - load both toxic and efficiency data
    toxic_train_path = config['data'].get('toxic_train_data_path')
    efficiency_train_path = config['data'].get('efficiency_train_data_path')
    toxic_test_path = config['data'].get('toxic_test_data_path')
    efficiency_test_path = config['data'].get('efficiency_test_data_path')
    
    print(f"\nLoading training data:")
    print(f"  Toxic data: {toxic_train_path}")
    print(f"  Efficiency data: {efficiency_train_path}")
    
    train_dataset_loader = MultiTaskSMILESDataset(
        toxic_data_path=toxic_train_path,
        efficiency_data_path=efficiency_train_path,
        tokenizer=tokenizer,
        max_length=config['data']['max_length']
    )
    train_dataset = train_dataset_loader.preprocess_data()
    print(f"Total training samples: {len(train_dataset)}")
    
    # Prepare evaluation dataset
    print(f"\nLoading evaluation data:")
    print(f"  Toxic test data: {toxic_test_path}")
    print(f"  Efficiency test data: {efficiency_test_path}")
    
    eval_dataset_loader = MultiTaskSMILESDataset(
        toxic_data_path=toxic_test_path,
        efficiency_data_path=efficiency_test_path,
        tokenizer=tokenizer,
        max_length=config['data']['max_length']
    )
    eval_dataset = eval_dataset_loader.preprocess_data()
    print(f"Total evaluation samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        fp16=config['optimization']['fp16'],
        bf16=config['optimization']['bf16'],
        save_strategy=config['training'].get('save_strategy', 'epoch'),
        save_steps=config['training'].get('save_steps', None),
        logging_steps=config['training']['logging_steps'],
        warmup_steps=config['training']['warmup_steps'],
        optim=config['optimization']['optim'],
        save_total_limit=config['training'].get('save_total_limit', 2),
        eval_strategy=config['training'].get('eval_strategy', 'no'),
        eval_steps=config['training'].get('eval_steps', None),
        per_device_eval_batch_size=config['training'].get('per_device_eval_batch_size', 1),
        report_to="none",
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        max_grad_norm=config['training']['max_grad_norm'],
        load_best_model_at_end=False,  # Don't load best model, we'll evaluate all checkpoints
    )
    
    # Initialize callback for checkpoint evaluation
    eval_callback = CheckpointEvaluationCallback(
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        tokenizer=tokenizer,
        output_dir=config['model']['output_dir']
    )
    
    # Initialize custom data collator
    data_collator = MultiTaskDataCollator(tokenizer=tokenizer)
    
    # Initialize custom trainer
    trainer = MultiTaskTrainer(
        alpha=config['loss']['alpha'],
        efficiency_num_classes=config['loss']['efficiency_num_classes'],
        efficiency_eps=config['loss']['efficiency_eps'],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[eval_callback],
    )
    
    # Start training
    print("\nStarting training...")
    print(f"Loss configuration:")
    print(f"  - Toxicity loss: {config['loss']['toxicity_loss_type']} (all samples)")
    print(f"  - Efficiency loss: {config['loss']['efficiency_loss_type']} (only non-toxic samples, alpha={config['loss']['alpha']})")
    print(f"  - Efficiency classes: {config['loss']['efficiency_num_classes']}")
    print(f"\nTraining settings:")
    print(f"  - Save every {config['training'].get('save_steps', 'N/A')} steps")
    print(f"  - Evaluate every {config['training'].get('eval_steps', 'N/A')} steps")
    print()
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to: {config['model']['output_dir']}")
    model.save_pretrained(config['model']['output_dir'])
    tokenizer.save_pretrained(config['model']['output_dir'])
    
    print("\nTraining completed!")
    print("="*80)
    print("\nTo evaluate all checkpoints, run:")
    print(f"  python evaluate_checkpoints.py --config {config_path}")
    print("="*80)


if __name__ == "__main__":
    train_multi_task_model("training_config.yaml")
