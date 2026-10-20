import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
DATA_DIR = Path("preverifier_data_json_v2")

class JsonlDataset(Dataset):
    def __init__(self, split: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict] = []

        path = DATA_DIR / f"{split}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.examples.append({
                    "text": obj["text"],
                    "label": int(obj["label"]),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.long)
        return item


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    train_dataset = JsonlDataset("train", tokenizer)
    val_dataset   = JsonlDataset("val", tokenizer)

    training_args = TrainingArguments(
        output_dir="preverifier_ckpts_json_v2",

        do_train=True,
        do_eval=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,

        learning_rate=2e-5,
        weight_decay=0.01,

        dataloader_num_workers=2,
        dataloader_pin_memory=False,

        logging_steps=10,
        save_total_limit=2,
    )

    from evaluate import load as load_metric
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("preverifier_model_from_variants_json_v2")
    tokenizer.save_pretrained("preverifier_model_from_variants_json_v2")

if __name__ == "__main__":
    main()
