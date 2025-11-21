# 提交Finetuning任务

## 快速提交

```bash
sbatch submit_finetuning.sh
```

## 任务配置

- **每5步保存checkpoint**: `save_steps: 5`
- **每5步进行评估**: `eval_steps: 5`
- **GPU**: H100 x1
- **内存**: 128G
- **时间限制**: 24小时
- **分区**: msigpu

## 任务流程

1. **训练阶段**:
   - 加载模型和数据
   - 每5步保存checkpoint到 `./qwen_lora_finetuned_multi_task/checkpoint-{step}/`
   - 每5步进行评估（基于分类头的快速评估）

2. **评估阶段**（训练完成后自动执行）:
   - 评估所有保存的checkpoints
   - 计算toxicity和efficiency的accuracy
   - 保存结果到 `checkpoint_results/`

## 查看任务状态

### 提交任务
```bash
sbatch submit_finetuning.sh
```

### 查看任务队列
```bash
squeue -u $USER
```

### 查看训练日志（实时）
```bash
# 获取任务ID
JOB_ID=$(squeue -u $USER -h -o %i | head -1)

# 查看输出日志
tail -f logs/finetune_${JOB_ID}.out

# 查看错误日志
tail -f logs/finetune_${JOB_ID}.err
```

### 查看最新日志
```bash
# 查看最新的输出日志
tail -f logs/finetune_*.out | tail -50

# 查看最新的错误日志
tail -f logs/finetune_*.err | tail -50
```

## 输出文件

### 训练输出
```
./qwen_lora_finetuned_multi_task/
├── checkpoint-5/
├── checkpoint-10/
├── checkpoint-15/
├── ...
└── checkpoint-{final}/
```

### 评估结果
```
checkpoint_results/
├── checkpoint-5_results.json
├── checkpoint-10_results.json
├── checkpoint-15_results.json
├── ...
└── all_checkpoints_summary.json
```

## 手动评估checkpoints

如果训练已完成但评估未完成，可以手动运行：

```bash
python3 evaluate_checkpoints.py --config training_config.yaml
```

## 取消任务

```bash
# 查看任务ID
squeue -u $USER

# 取消任务
scancel <JOB_ID>
```

## 注意事项

1. **评估时间**: 训练过程中的评估是快速评估（基于分类头），完整评估（基于文本生成）在训练完成后进行
2. **存储空间**: 每个checkpoint约占用一定空间，确保有足够存储
3. **时间限制**: 如果训练时间超过24小时，需要修改脚本中的 `--time` 参数

