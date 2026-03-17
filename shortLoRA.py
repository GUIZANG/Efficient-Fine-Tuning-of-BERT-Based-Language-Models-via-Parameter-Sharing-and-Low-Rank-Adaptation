import json
from datasets import Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
import torch

# 1. 加载本地SQuAD数据，简化，只取前1000条
def load_squad_local(file_path, max_samples=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        squad_dict = json.load(f)

    samples = []
    count = 0
    for article in squad_dict['data']:
        for para in article['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                if count >= max_samples:
                    break
                question = qa['question']
                answers = qa['answers'][0] if qa['answers'] else {"text": "", "answer_start": 0}
                samples.append({
                    'context': context,
                    'question': question,
                    'answers': answers
                })
                count += 1
            if count >= max_samples:
                break
        if count >= max_samples:
            break
    return Dataset.from_list(samples)

# 2. 加载本地模型和tokenizer
model_path = "D:/SemA/Papers/Experiment/models/bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

# 3. 预处理函数：简化版本，不精确对齐答案，只编码文本
def preprocess_function(examples):
    questions = examples['question']
    contexts = examples['context']

    inputs = tokenizer(
        questions,
        contexts,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors=None
    )

    # dummy start/end positions 都设为0，跑通训练
    batch_size = len(questions)
    inputs["start_positions"] = [0] * batch_size
    inputs["end_positions"] = [0] * batch_size

    return inputs

# 4. 加载数据集，只取1000条
train_dataset = load_squad_local("D:/SemA/Papers/Experiment/models/datasets/train-v1.1.json", max_samples=1000)
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['context', 'question', 'answers'])

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="no",
    logging_steps=10,
    logging_dir="./logs",
    report_to=None,
    disable_tqdm=False
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# 7. 开始训练
trainer.train()
