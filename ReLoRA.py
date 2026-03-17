import json
import numpy as np
import collections
import re
import string

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model


# ======================
# 1. Load SQuAD v1.1
# ======================
def load_and_prepare_squad(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        squad_data = json.load(f)

    examples = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append({
                    "context": context,
                    "question": qa["question"],
                    "answers": qa["answers"][0]
                })
    return examples


train_path = r"D:\SemA\Papers\Experiment\models\datasets\train-v1.1.json"
dev_path   = r"D:\SemA\Papers\Experiment\models\datasets\dev-v1.1.json"

train_dataset = Dataset.from_list(load_and_prepare_squad(train_path))
dev_dataset   = Dataset.from_list(load_and_prepare_squad(dev_path))


# ======================
# 2. Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    r"D:\SemA\Papers\Experiment\models\bert-base-uncased",
    use_fast=True
)


# ======================
# 3. Preprocess (CORRECT span alignment)
# ======================
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="longest",
        max_length=384,
        return_offsets_mapping=True,
        return_token_type_ids=True,
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        sequence_ids = tokenized.sequence_ids(i)

        start_token = None
        end_token = None

        for idx, (s, e) in enumerate(offsets):
            if sequence_ids[idx] != 1:  # only context
                continue
            if s <= start_char < e:
                start_token = idx
            if s < end_char <= e:
                end_token = idx

        if start_token is None or end_token is None:
            cls_idx = tokenized["input_ids"][i].index(tokenizer.cls_token_id)
            start_token = cls_idx
            end_token = cls_idx

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")

    return tokenized


train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

dev_tokenized = dev_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dev_dataset.column_names,
)


# ======================
# 4. Model + LoRA
# ======================
base_model = AutoModelForQuestionAnswering.from_pretrained(
    r"D:\SemA\Papers\Experiment\models\bert-base-uncased"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.2,
    target_modules=["query", "value"],  # 🔑 QA 最稳配置
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


# ======================
# 5. Metrics (Official-style)
# ======================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold = normalize_answer(a_gold).split()
    pred = normalize_answer(a_pred).split()
    common = collections.Counter(gold) & collections.Counter(pred)
    num_same = sum(common.values())

    if len(gold) == 0 or len(pred) == 0:
        return int(gold == pred)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


# ======================
# 6. n-best inference (IMPORTANT)
# ======================
def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    start_logits, end_logits = predictions

    n_best = 20
    max_len = 30

    ems, f1s = [], []

    for i in range(len(start_logits)):
        encoding = tokenizer(
            dev_dataset[i]["question"],
            dev_dataset[i]["context"],
            truncation=True,
            max_length=384,
            return_offsets_mapping=True,
        )

        offsets = encoding["offset_mapping"]
        context = dev_dataset[i]["context"]
        gold = dev_dataset[i]["answers"]["text"]

        start_indexes = np.argsort(start_logits[i])[-n_best:]
        end_indexes = np.argsort(end_logits[i])[-n_best:]

        best_score = -1e9
        best_span = (0, 0)

        for s in start_indexes:
            for e in end_indexes:
                if e < s or e - s + 1 > max_len:
                    continue
                score = start_logits[i][s] + end_logits[i][e]
                if score > best_score:
                    best_score = score
                    best_span = (s, e)

        s, e = best_span
        pred = context[offsets[s][0]: offsets[e][1]]

        ems.append(compute_exact(gold, pred))
        f1s.append(compute_f1(gold, pred))

    return {
        "exact_match": np.mean(ems) * 100,
        "f1": np.mean(f1s) * 100,
    }


# ======================
# 7. Training Arguments
# ======================
training_args = TrainingArguments(
    output_dir="./results_lora",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=500,

    learning_rate=1e-4,        # 🔑 LoRA 高 LR
    num_train_epochs=6,
    warmup_ratio=0.1,
    weight_decay=0.001,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # effective 32
    per_device_eval_batch_size=32,

    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)


# ======================
# 8. Trainer
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,1
)

print("🚀 Start LoRA fine-tuning...")
trainer.train()

print("✅ Final Evaluation:")
metrics = trainer.evaluate()
print(metrics)
