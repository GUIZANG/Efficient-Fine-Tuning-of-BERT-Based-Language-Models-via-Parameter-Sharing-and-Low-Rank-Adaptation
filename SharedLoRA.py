import json
import re
import string
import collections
import numpy as np
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)

# =========================
# 1. 数据加载（SQuAD 1.1）
# =========================
def load_squad(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    examples = []
    for article in data:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                ans = qa["answers"][0]
                examples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answer_text": ans["text"],
                    "answer_start": ans["answer_start"],
                })
    return examples


train_path = r"D:\SemA\Papers\Experiment\models\datasets\train-v1.1.json"
dev_path   = r"D:\SemA\Papers\Experiment\models\datasets\dev-v1.1.json"

train_ds = Dataset.from_list(load_squad(train_path))
dev_ds   = Dataset.from_list(load_squad(dev_path))

# =========================
# 2. Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    r"D:\SemA\Papers\Experiment\models\bert-base-uncased",
    use_fast=True
)

MAX_LEN = 384
STRIDE = 128

# =========================
# 3. 预处理（Sliding Window）
# =========================
def preprocess(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer_start = examples["answer_start"][sample_idx]
        answer_text = examples["answer_text"][sample_idx]
        answer_end = answer_start + len(answer_text)

        cls_index = tokenized["input_ids"][i].index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        start, end = cls_index, cls_index

        for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
            if seq_id != 1:
                continue
            if offset[0] <= answer_start < offset[1]:
                start = idx
            if offset[0] < answer_end <= offset[1]:
                end = idx

        start_positions.append(start)
        end_positions.append(end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
dev_tok   = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names)

# =========================
# 4. Shared LoRA Linear
# =========================
class SharedLoRALinear(nn.Module):
    def __init__(self, base_layer, A, B, alpha):
        super().__init__()
        self.base = base_layer
        self.A = A
        self.B = B
        self.scaling = alpha / A.size(0)

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling


# =========================
# 5. 模型 + 注入 Shared LoRA
# =========================
model = AutoModelForQuestionAnswering.from_pretrained(
    r"D:\SemA\Papers\Experiment\models\bert-base-uncased"
)

hidden = model.config.hidden_size
r = 16
alpha = 32

shared_A = nn.Parameter(torch.randn(r, hidden) * 0.02)
shared_B = nn.Parameter(torch.zeros(hidden, r))

model.shared_lora_A = shared_A
model.shared_lora_B = shared_B

for layer in model.bert.encoder.layer:
    attn = layer.attention.self
    attn.query = SharedLoRALinear(attn.query, shared_A, shared_B, alpha)
    attn.value = SharedLoRALinear(attn.value, shared_A, shared_B, alpha)

# 冻结 base，仅训练 Shared LoRA + QA head
for name, param in model.named_parameters():
    if "shared_lora" in name or "qa_outputs" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# =========================
# 6. EM / F1
# =========================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def f1_score(gold, pred):
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    return 2 * num_same / (len(gold_toks) + len(pred_toks))

def exact_match(gold, pred):
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_metrics(p):
    start_logits, end_logits = p.predictions
    start_preds = np.argmax(start_logits, axis=1)
    end_preds   = np.argmax(end_logits, axis=1)

    ems, f1s = [], []

    for i in range(len(start_preds)):
        sample = dev_ds[i]

        encoding = tokenizer(
            sample["question"],
            sample["context"],
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
        )

        offsets = encoding["offset_mapping"]

        s = min(start_preds[i], len(offsets) - 1)
        e = min(end_preds[i], len(offsets) - 1)

        pred = sample["context"][offsets[s][0]: offsets[e][1]]
        gold = sample["answer_text"]

        ems.append(exact_match(gold, pred))
        f1s.append(f1_score(gold, pred))

    return {
        "exact_match": np.mean(ems) * 100,
        "f1": np.mean(f1s) * 100,
    }

# =========================
# 7. TrainingArguments
# =========================
args = TrainingArguments(
    output_dir="./shared_lora_outputs",
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=1,
    save_safetensors=False,

    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=24,

    fp16=True,
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=dev_tok,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("🚀 Training Shared LoRA QA...")
trainer.train()

print("📊 Final Evaluation:")
print(trainer.evaluate())
