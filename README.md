# Efficient Fine-Tuning of BERT-Based Language Models via Parameter Sharing and LoRA

## 项目简介
本项目实现了一种高效的 BERT 模型微调方法，通过**参数共享**和 **低秩适配（LoRA）** 技术显著减少可训练参数，同时保持在问答任务上的竞争性能。此方法旨在解决大模型在计算和内存上的高成本问题，为资源受限环境下的 NLP 应用提供轻量级解决方案。

- **方法亮点**：
  - 使用 LoRA 在注意力层注入低秩适配器，仅更新模型 0.27% 的参数。
  - 将 BERT 的 12 个 Transformer 层分为四个块，块内共享参数，提高参数效率。
  - 冻结预训练模型的主干参数，仅训练适配器和任务特定输出层。
  
- **实验结果**：
  - 数据集：SQuAD v1.1
  - EM（Exact Match）：63.60%
  - F1 分数：78.16%
  - 相比全微调，训练成本和内存使用降低 >99%。

---

## 环境依赖

请确保 Python >=3.8，并安装以下库：

```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.57.3
pip install datasets
pip install numpy tqdm
```

可选 GPU 支持：NVIDIA GPU + CUDA 13.0

1. 数据准备

下载 SQuAD v1.1 数据集并放入 data/ 目录。

确保训练和验证数据的路径正确。

2. 模型训练

python scripts/train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/train-v1.1.json \
    --validation_file data/dev-v1.1.json \
    --output_dir models/finetuned \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --use_lora True \
    --parameter_sharing True

python scripts/evaluate.py \
    --model_dir models/finetuned \
    --validation_file data/dev-v1.1.json


相关 LoRA 和 BERT 参考文献：
Frees, D., et al., Exploring Efficient Learning of Small BERT Networks with LoRA, ArXiv, 2025.
Devlin, J., et al., BERT: Pre-training of Deep Bidirectional Transformers, NAACL-HLT 2019.
