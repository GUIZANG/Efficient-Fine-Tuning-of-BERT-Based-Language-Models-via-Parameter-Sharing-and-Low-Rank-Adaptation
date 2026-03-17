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


# ======================
# 1. 读取并展开 SQuAD
# ======================
def load_and_prepare_squad(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        squad_data = json.load(f)

    examples = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                # SQuAD答案可能有多个，这里只用第一个（可扩展）
                answer = qa["answers"][0]
                examples.append({
                    "context": context,
                    "question": question,
                    "answers": answer
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
    r"D:\SemA\Papers\Experiment\models\bert-base-uncased"
)


# ======================
# 3. 预处理（对齐答案）
# ======================
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",  # 可改成 "longest" 视显存调整
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

        # 初始化为-1，方便检测未匹配
        start_token = -1
        end_token = -1

        # 找start_token（第一个包含answer起始字符的token）
        for idx, (s, e) in enumerate(offsets):
            if s <= start_char < e:
                start_token = idx
                break

        # 找end_token（第一个包含answer结束字符的token）
        for idx, (s, e) in enumerate(offsets):
            if s < end_char <= e:
                end_token = idx
                break

        # 若找不到，设置成 CLS token 位置（0）
        if start_token == -1:
            start_token = 0
        if end_token == -1:
            end_token = 0

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")  # 去掉offset，节省内存

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
# 4. 模型
# ======================
model = AutoModelForQuestionAnswering.from_pretrained(
    r"D:\SemA\Papers\Experiment\models\bert-base-uncased"
)


# ======================
# 5. EM / F1 计算辅助函数
# ======================
def normalize_answer(s):
    def lower(text): return text.lower()
    def remove_punc(text): return "".join(ch for ch in text if ch not in string.punctuation)
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ======================
# 6. Trainer 用的 metrics
# ======================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    start_logits, end_logits = predictions

    start_preds = np.argmax(start_logits, axis=-1)
    end_preds = np.argmax(end_logits, axis=-1)

    f1_scores = []
    em_scores = []

    for i in range(len(start_preds)):
        # 重新tokenize验证集样本，获取offset_mapping
        encoding = tokenizer(
            dev_dataset[i]["question"],
            dev_dataset[i]["context"],
            truncation=True,
            max_length=384,
            return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"]
        context = dev_dataset[i]["context"]
        gold_answer = dev_dataset[i]["answers"]["text"]

        start_idx = min(start_preds[i], len(offsets) - 1)
        end_idx = min(end_preds[i], len(offsets) - 1)

        pred_start = offsets[start_idx][0]
        pred_end = offsets[end_idx][1]

        pred_answer = context[pred_start:pred_end]

        em_scores.append(compute_exact(gold_answer, pred_answer))
        f1_scores.append(compute_f1(gold_answer, pred_answer))

    return {
        "exact_match": np.mean(em_scores) * 100,
        "f1": np.mean(f1_scores) * 100,
    }


# ======================
# 7. 训练参数
# ======================
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=500,
    save_steps=1000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    # gradient_accumulation_steps=2, # 如果显存不足可以开启累积梯度
    # gradient_checkpointing=True,   # 减少显存，训练速度稍慢
)


print("Eval dataset size:", len(dev_tokenized))


# ======================
# 8. Trainer
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# ======================
# 9. 开始训练
# ======================
print("Starting training...")
trainer.train()
print("Training finished.")

metrics = trainer.evaluate()
print(metrics)
