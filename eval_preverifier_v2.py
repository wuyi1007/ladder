import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "preverifier_model_from_variants_json_v2"
TEST_PATH = "preverifier_data_json_v2/test.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def score(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        p1 = torch.softmax(logits, dim=-1)[0, 1].item()
    return p1

# load test set
pos_examples = []
neg_examples = []

with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        if ex["label"] == 1:
            pos_examples.append(ex)
        else:
            neg_examples.append(ex)

print(f"test positives: {len(pos_examples)}, negatives: {len(neg_examples)}")

# sample a few
random.seed(42)
pos_sample = random.sample(pos_examples, min(5, len(pos_examples)))
neg_sample = random.sample(neg_examples, min(5, len(neg_examples)))

def eval_group(name, group):
    probs = []
    print(f"\n{name}:")
    for ex in group:
        p = score(ex["text"])
        probs.append(p)
        print(f"  p={p:.3f} :: {ex['text'][:120]}...")
    if probs:
        print(f"  avg {name} prob = {sum(probs)/len(probs):.3f}")

eval_group("POS", pos_sample)
eval_group("NEG", neg_sample)
