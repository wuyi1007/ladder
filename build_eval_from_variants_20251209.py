import json
from pathlib import Path

# 1) Input: your new LADDER variants file
JSON_PATH = "variants_all_20251209_234614.json"

# 2) Output: one JSONL file with (text, label) for evaluation
OUT_PATH = Path("preverifier_eval_20251209_234614.jsonl")

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = 0
    n_pos = 0
    n_neg = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for rec in data:
            original = rec.get("original")
            variant = rec.get("variant")
            if not original or not variant:
                continue

            vp = rec.get("verification_passed", None)
            label = 1 if vp is True else 0

            text = (
                f"Original: {original} "
                f"[SEP] Variant: {variant} "
                f"[DIFF: {rec.get('requested_difficulty', '')}]"
            )

            obj = {
                "text": text,
                "original": original,
                "variant": variant,
                "label": label,
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            n_total += 1
            if label == 1:
                n_pos += 1
            else:
                n_neg += 1

    print(f"Wrote {OUT_PATH}")
    print(f"Total examples: {n_total}, positives: {n_pos}, negatives: {n_neg}")

if __name__ == "__main__":
    main()
