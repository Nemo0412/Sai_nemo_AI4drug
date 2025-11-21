# 数据集使用总结 (Data Summary)

## Finetuning 训练数据集

### 1. Toxic Data (毒性数据)
- **文件路径**: `data/toxic_data/toxicity_train_data.jsonl`
- **数据量**: **200 个样本**
- **标签类型**: Toxicity (毒性)
- **标签值**: **0 或 1** (离散二值)
  - 0 = Non-toxic (无毒)
  - 1 = Toxic (有毒)
- **用途**: 训练模型预测分子是否有毒

### 2. Efficiency Data (效率数据)
- **文件路径**: `data/efficiency_train_data.jsonl`
- **数据量**: **600 个样本**
- **标签类型**: Efficiency Score (效率分数)
- **标签值**: **1-10 的离散整数**
- **用途**: 训练模型预测分子的delivery efficiency分数

### 训练数据总计
- **总样本数**: 200 (toxic) + 600 (efficiency) = **800 个样本**
- **训练方式**: 同时使用两种数据，使用Conditional Multi-Task Loss
  - Toxic data → 训练毒性预测 (所有样本)
  - Efficiency data → 训练效率预测 (仅非毒性样本)

---

## 测试数据集

### 1. Toxic Test Data (毒性测试数据)
- **文件路径**: `data/toxic_data/toxicity_test_data.jsonl`
- **数据量**: **200 个样本**
- **标签类型**: Toxicity (毒性)
- **标签值**: **0 或 1** (离散二值)
- **评估指标**: 准确率 (Accuracy)

### 2. Efficiency Test Data (效率测试数据)
- **文件路径**: `data/efficiency_test_data.jsonl`
- **数据量**: **600 个样本**
- **标签类型**: Efficiency Score (效率分数)
- **标签值**: **1-10 的离散整数**
- **评估指标**: 
  - 精确匹配 (Exact Match)
  - ±1误差 (Within ±1)
  - ±2误差 (Within ±2)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

### 测试数据总计
- **总样本数**: 200 (toxic) + 600 (efficiency) = **800 个样本**

---

## 数据来源

### Toxic Data
- **原始文件**: `data/toxic_data/Carcinogenicity_Carcinogenicity.csv`
- **处理脚本**: `process_toxicity_data.py`
- **处理方式**: 提取前400个有效样本，分成200训练 + 200测试

### Efficiency Data
- **原始文件**: 通过 `prepare_training_data.py` 从Excel文件转换
- **处理方式**: 转换为JSONL格式，包含SMILES和效率分数

---

## 配置文件

所有数据路径在 `training_config.yaml` 中配置：

```yaml
data:
  # Toxic data: for toxicity prediction (0 or 1)
  toxic_train_data_path: "data/toxic_data/toxicity_train_data.jsonl"
  toxic_test_data_path: "data/toxic_data/toxicity_test_data.jsonl"
  # Efficiency data: for efficiency prediction (1-10 discrete values)
  efficiency_train_data_path: "data/efficiency_train_data.jsonl"
  efficiency_test_data_path: "data/efficiency_test_data.jsonl"
```

---

## 总结表格

| 数据类型 | 训练集 | 测试集 | 标签类型 | 标签范围 |
|---------|--------|--------|---------|---------|
| **Toxic** | 200 | 200 | 毒性 | 0 或 1 |
| **Efficiency** | 600 | 600 | 效率分数 | 1-10 (整数) |
| **总计** | **800** | **800** | - | - |

---

## 注意事项

1. **输出约束**:
   - Toxicity输出: 严格为 0 或 1
   - Efficiency输出: 严格为 1-10 的离散整数

2. **Loss计算**:
   - Toxicity loss: 对所有样本计算
   - Efficiency loss: 仅对非毒性样本计算 (conditional)

3. **数据平衡**:
   - 训练集: Toxic 200个，Efficiency 600个样本
   - 测试集: Toxic 200个，Efficiency 600个样本
   - 数据已shuffle (seed=42) 确保随机性


