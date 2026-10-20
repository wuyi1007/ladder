import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "preverifier_model_from_variants_json_v2"
TEST_PATH = "preverifier_eval_20251210_073553.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

labels = []
probs  = []

# Load all test examples
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        text = ex["text"]
        lab  = int(ex["label"])

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            p = torch.softmax(logits, dim=-1)[0, 1].item()

        labels.append(lab)
        probs.append(p)

N = len(labels)
pos_total = sum(labels)
neg_total = N - pos_total

print(f"Total test examples: {N}, positives={pos_total}, negatives={neg_total}")
print()
print("thr   recall_pos  prec_pos   kept_frac   "
      "filtered_T  filtered_F  kept_F   kept_T")
print("----  ----------  --------   ---------   "
      "----------  ----------  -------  -------")

# Thresholds from 0.0 to 1.0 in steps of 0.05
for thr in [i / 20 for i in range(0, 21)]:
    tp = fp = tn = fn = 0

    for y, p in zip(labels, probs):
        # pred=1 = "kept by pre-verifier"
        pred = 1 if p >= thr else 0

        if y == 1 and pred == 1:
            tp += 1  # kept & true
        elif y == 1 and pred == 0:
            fn += 1  # filtered & true
        elif y == 0 and pred == 1:
            fp += 1  # kept & false
        elif y == 0 and pred == 0:
            tn += 1  # filtered & false

    # derived metrics
    recall = tp / pos_total if pos_total else 0.0
    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    kept_frac = (tp + fp) / N  # fraction of all variants that pass pre-verifier

    # your requested summaries (counts; you can think of percentages easily)
    filtered_true  = fn  # filtered by pre-verifier but actually true
    filtered_false = tn  # filtered by pre-verifier and actually false
    kept_false     = fp  # kept by pre-verifier but actually false
    kept_true      = tp  # kept by pre-verifier and actually true

    print(
        f"{thr:4.2f}  "
        f"{recall:10.3f}  {prec:8.3f}   {kept_frac:9.3f}   "
        f"{filtered_true:10d}  {filtered_false:10d}  {kept_false:7d}  {kept_true:7d}"
    )
