# mRNA Transfection Efficiency Prediction with RDKit-Enhanced Multi-Task Learning# LipoAgent - Multi-Agent Drug Discovery Framework



基于Qwen2-7B-Instruct和RDKit分子描述符的mRNA转染效率和毒性预测系统。Multi-agent framework for lipid nanoparticle (LNP) delivery systems with conditional multi-task loss training.



## 项目概述## Quick Start



本项目使用LoRA微调Qwen2-7B-Instruct模型，结合RDKit分子描述符预测mRNA转染效率（1-10分）和毒性（0/1）。### 1. Finetune Training (Multi-Task Loss)

```bash

**关键特性：**python multi_task_training.py

- RDKit分子特征增强（分子量、LogP、氢键供体/受体、TPSA等）```

- 多任务学习（效率预测 + 毒性分类）- **Config**: `training_config.yaml`

- 自适应容差评估（极值±1，中间值±2）- **Output**: `./qwen_lora_finetuned_multi_task/`

- Multi-agent预测流程（预测器 + 验证器）- **Data**: 

  - Toxic data: `data/toxic_data/toxicity_train_data.jsonl` (toxicity labels: 0 or 1)

## 目录结构  - Efficiency data: `data/train_data.jsonl` (efficiency scores: 1-10)



```### 2. Multi-Agent Prediction Loop

Sai_nemo_AI4drug/```bash

├── train_multitask_rdkit.py          # RDKit增强的训练脚本python multi_agent_pipeline.py

├── enhance_data_with_rdkit.py        # 数据RDKit特征增强```

├── split_data.py                     # 数据集划分- **Config**: `config.yaml`

├── evaluate_model.py                 # 模型评估（最终版本）- **Output**: `output/` directory

├── show_results.py                   # 快速查看评估结果- **Requires**: Trained model from step 1

├── multi_agent_pipeline.py           # Multi-agent预测流程

├── predictor_agent.py                # 预测器Agent## Complete Workflow

├── verifier_agent.py                 # 验证器Agent

├── predict_virtual_library.py        # 虚拟文库预测```bash

├── config.yaml                       # 训练配置# Step 1: Prepare data (if needed)

├── requirements.txt                  # Python依赖python process_toxicity_data.py  # Process toxic data

├── data/                             # 训练数据目录

│   ├── efficiency_train.jsonl# Step 2: Train model (uses both toxic and efficiency data)

│   ├── efficiency_test.jsonlpython multi_task_training.py

│   ├── toxicity_train.jsonl

│   └── toxicity_test.jsonl# Step 3: Evaluate model (tests both toxicity and efficiency)

├── data_ann/                         # 标注数据和预测结果python evaluate_multi_task.py

└── qwen_multitask_rdkit_checkpoints/ # 训练的checkpoint

```# Step 4: Run multi-agent prediction

python multi_agent_pipeline.py

## 环境配置```



### 安装依赖## Architecture



```bash**Multi-Agent Framework:**

pip install -r requirements.txt- **Predictor Agent**: Predicts toxicity + efficiency with reasoning (uses finetuned model)

```- **Verifier Agent**: Validates logical consistency (uses base model)

- **Negotiation**: Up to 2 loops, then human feedback if no agreement

**主要依赖：**

- torch >= 2.0.0**Training:**

- transformers >= 4.30.0- **Conditional Multi-Task Loss**: `loss = loss_tox + alpha * loss_eff`

- peft >= 0.4.0  - **Toxicity loss**: Binary cross-entropy (all samples from toxic data, labels: 0 or 1)

- rdkit >= 2023.0.0  - **Efficiency loss**: Cross-entropy (only non-toxic samples from efficiency data, labels: 1-10 discrete values)

- openpyxl (用于Excel文件读取)  - Toxic data is used for toxicity prediction training

  - Efficiency data is used for efficiency prediction training

## 1. 模型训练  - Efficiency loss is only computed for non-toxic samples (conditional)



### 步骤1：数据准备和增强## Configuration



```bash### Training Config (`training_config.yaml`)

# 1. 划分训练/测试集```yaml

python3 split_data.pydata:

  toxic_train_data_path: "data/toxic_data/toxicity_train_data.jsonl"  # Toxicity labels (0/1)

# 2. 使用RDKit增强数据（添加分子描述符）  efficiency_train_data_path: "data/train_data.jsonl"  # Efficiency scores (1-10)

python3 enhance_data_with_rdkit.py

```loss:

  alpha: 1.0  # Efficiency loss weight

**RDKit特征包括：**  efficiency_num_classes: 10

- 分子量 (MW)

- LogP（脂水分配系数）training:

- 氢键供体/受体数量  num_epochs: 3

- 拓扑极性表面积 (TPSA)  learning_rate: 2e-4

- 可旋转键数量  batch_size: 4

- 芳香环数量```

- 总原子数

### Multi-Agent Config (`config.yaml`)

### 步骤2：配置训练参数```yaml

model:

编辑 `config.yaml`：  base_model_path: "Qwen/Qwen-7B-Chat"

  lora_path: "./qwen_lora_finetuned"  # Path to trained model

```yaml

model:multi_agent:

  base_model: "Qwen/Qwen2-7B-Instruct"  max_negotiation_loops: 2

    low_confidence_threshold: 6.0

lora:```

  r: 64

  lora_alpha: 16## Output Files

  lora_dropout: 0.05

  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]**Training:**

- Model saved to: `./qwen_lora_finetuned_multi_task/`

training:

  num_epochs: 3**Multi-Agent:**

  batch_size: 2- `output/agreed_molecules.json` - Agent agreements

  gradient_accumulation_steps: 8- `output/disagreed_molecules.json` - Agent disagreements

  learning_rate: 1e-4- `output/human_feedback_molecules.json` - Human feedback cases

  max_length: 512- `output/predictions.json` - All predictions

  warmup_steps: 100- `output/feedback_history.json` - Feedback history

  

data:## Verify Setup

  efficiency_train: "data/efficiency_train.jsonl"

  efficiency_test: "data/efficiency_test.jsonl"```bash

  toxicity_train: "data/toxicity_train.jsonl"python verify_code.py

  toxicity_test: "data/toxicity_test.jsonl"```

  

output:## Requirements

  checkpoint_dir: "qwen_multitask_rdkit_checkpoints"

  save_steps: 20- Python 3.8+

```- PyTorch, Transformers, PEFT

- See `requirements.txt`

### 步骤3：启动训练

## Key Files

```bash

# 后台训练| File | Purpose |

nohup python3 train_multitask_rdkit.py > training.log 2>&1 &|------|---------|

| `multi_task_training.py` | Finetune with conditional loss (toxic + efficiency data) |

# 保存进程ID| `evaluate_multi_task.py` | Evaluate both toxicity and efficiency predictions |

echo $! > training_pid.txt| `multi_agent_pipeline.py` | Multi-agent prediction loop |

| `predictor_agent.py` | Predictor agent implementation |

# 监控训练进度| `verifier_agent.py` | Verifier agent implementation |

tail -f training.log| `training_config.yaml` | Training hyperparameters |

```| `config.yaml` | Multi-agent configuration |


**训练输出：**
- Checkpoint保存在 `qwen_multitask_rdkit_checkpoints/checkpoint-{step}/`
- 每20步保存一个checkpoint
- 训练日志包含loss和评估指标

## 2. 模型评估

### 评估单个Checkpoint

```bash
python3 evaluate_model.py \
  --checkpoint qwen_multitask_rdkit_checkpoints/checkpoint-140 \
  --efficiency_test data/efficiency_test.jsonl \
  --toxicity_test data/toxicity_test.jsonl \
  --output results/checkpoint_140_results.json
```

**评估指标：**
- **Efficiency MAE**: 平均绝对误差
- **Efficiency Accuracy**: 自适应容差准确率
  - 极值（1,2,9,10）：±1容差
  - 中间值（3-8）：±2容差
- **Toxicity Accuracy**: 分类准确率（调整版，包含600个效率样本作为真阴性）

### 快速查看结果

```bash
# 查看特定结果文件
python3 show_results.py results/checkpoint_140_results.json

# 查看最新结果（默认checkpoint-140）
python3 show_results.py
```

**输出示例：**
```
=== Checkpoint-140 评估结果 ===
Efficiency MAE: 1.398
Efficiency Accuracy: 80.7%
Toxicity Accuracy: 88.5%
```

### 最佳Checkpoint

根据评估结果，**checkpoint-140** 表现最佳：
- Efficiency MAE: 1.398
- Efficiency Accuracy: 80.7%
- Toxicity Accuracy: 88.5%

## 3. Multi-Agent预测流程

Multi-agent系统包含两个Agent：
1. **Predictor Agent**: 生成初步预测和推理
2. **Verifier Agent**: 验证预测的合理性，提供反馈

### 单分子预测（带验证）

```bash
python3 multi_agent_pipeline.py \
  --smiles "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O" \
  --checkpoint qwen_multitask_rdkit_checkpoints/checkpoint-140
```

**输出示例：**
```json
{
  "smiles": "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O",
  "molecular_weight": 267.36,
  "logp": 1.23,
  "efficiency_score": 8,
  "toxicity": 0,
  "predictor_reasoning": "该分子具有适中的分子量和LogP值...",
  "verifier_feedback": "预测合理，分子结构支持高效率转染...",
  "confidence": "high"
}
```

### 虚拟文库批量预测

```bash
# 预测Excel文件中的所有分子
nohup python3 predict_virtual_library.py > prediction.log 2>&1 &
echo $! > prediction_pid.txt

# 监控进度
tail -f prediction.log
```

**输入：** `data_ann/virtual library-2500.xlsx`
**输出：** `data_ann/result_new.json`

**输出格式：**
```json
[
  {
    "number": 1,
    "smiles": "CC(C)NCC...",
    "efficiency_score": 9,
    "molecular_weight": 267.36,
    "logp": 1.23,
    "reasoning": "高效率分子，具有理想的分子量和脂溶性..."
  },
  {
    "number": 2,
    "smiles": "COC1=CC=C...",
    "efficiency_score": 8,
    "molecular_weight": 245.28,
    "logp": 2.15,
    "reasoning": "良好的膜渗透性和适中的极性..."
  }
]
```

**特性：**
- 按效率分数降序排列
- 效率≥7的分子包含详细reasoning
- 自动提取RDKit分子特征
- 使用训练时相同的prompt格式

### Multi-agent测试流程

```bash
# 测试完整的multi-agent流程
python3 test_pipeline.py
```

## 常见问题

### 1. CUDA内存不足

降低batch size或使用梯度累积：
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

### 2. RDKit解析失败

某些SMILES可能无法被RDKit解析，这些分子会被自动跳过并记录在日志中。

### 3. 网络连接HuggingFace失败

模型会自动使用本地缓存（~/.cache/huggingface/），首次运行需要确保网络连接。

### 4. 预测格式与训练不一致

确保使用相同的prompt模板：
```python
prompt = f"What is the predicted score for this molecular structure: {smiles}? Molecular properties: MW={mw:.1f}Da, LogP={logp:.2f}, H-bond donors={hbd}, H-bond acceptors={hba}, TPSA={tpsa:.1f}Ų, Rotatable bonds={rb}, Aromatic rings={ar}, Total atoms={atoms}"
```

## 性能指标

### Checkpoint-140（最佳模型）

| 指标 | 数值 |
|------|------|
| Efficiency MAE | 1.398 |
| Efficiency Accuracy | 80.7% |
| Toxicity Accuracy | 88.5% |
| 训练步数 | 140 steps |
| 训练轮次 | ~3 epochs |

### 数据集统计

- 效率训练集: 600 samples
- 效率测试集: 150 samples
- 毒性训练集: 200 samples
- 毒性测试集: 50 samples

## 引用

如果您使用本项目，请引用：

```bibtex
@software{mrna_efficiency_prediction,
  title={mRNA Transfection Efficiency Prediction with RDKit-Enhanced Multi-Task Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Sai_nemo_AI4drug}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
