# STEP 1: Import libraries
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import f1_score, precision_score, recall_score

# STEP 2: Load your dataset (JSONL)
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/raw/train50k.jsonl",
        "validation": "data/raw/validation20k.jsonl"
    }
)

# (dataset)
# STEP 3: Load tokenizer (NO re-tokenization)
MODEL_NAME = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# STEP 4: Create label mappings
all_labels = set()

for sample in dataset["train"]:
    all_labels.update(sample["label_ids"])

num_labels = len(all_labels)

id2label = {i: f"LABEL_{i}" for i in sorted(all_labels)}
label2id = {v: k for k, v in id2label.items()}

# STEP 5: Convert tokens → input_ids

def encode_batch(batch):
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        add_special_tokens=True
    )

    aligned_labels = []

    for i, labels in enumerate(batch["label_ids"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                # Special tokens like [CLS], [SEP]
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                # Same wordpiece → repeat label
                label_ids.append(labels[word_id])

            prev_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized

dataset = dataset.map(
    encode_batch,
    batched=True,
    remove_columns=["tokens", "labels", "label_ids"]
)

# STEP 6: Data collator (batch padding)
data_collator = DataCollatorForTokenClassification(tokenizer)

# STEP 7: Load model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# STEP 8: Define evaluation metrics (F1)
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_preds = []
    true_labels = []

    for pred, lab in zip(predictions, labels):
        curr_preds = []
        curr_labels = []

        for p, l in zip(pred, lab):
            if l != -100:
                curr_preds.append(id2label[p])
                curr_labels.append(id2label[l])

        true_preds.append(curr_preds)
        true_labels.append(curr_labels)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }


# STEP 9: Training configuration (batches + epochs)
training_args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"
)

# STEP 10: Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer   # instead of tokenizer=
)


# STEP 11: Train the model
trainer.train()

# STEP 12: Evaluate final model
metrics = trainer.evaluate()
print(metrics)


# STEP 13: Save the final model ✅
SAVE_DIR = "./ner_model_final"

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
