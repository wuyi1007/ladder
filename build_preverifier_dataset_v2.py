import json
import random
from pathlib import Path

random.seed(42)

JSON_PATH = "variants_all_20251129_095709.json"
OUT_DIR = Path("preverifier_data_json_v2")
OUT_DIR.mkdir(exist_ok=True)

NEG_PER_POS = 3  # how many negatives per positive to sample

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) Collect all examples with natural labels from verification_passed
    examples = []
    for idx, rec in enumerate(data):
        original = rec.get("original")
        variant = rec.get("variant")
        if not original or not variant:
            continue

        vp = rec.get("verification_passed", None)
        label = 1 if vp is True else 0

        examples.append({
            "id": idx,
            "original": original,
            "variant": variant,
            "requested_difficulty": rec.get("requested_difficulty", ""),
            "label": label,
        })

    print(f"Total raw examples: {len(examples)}")

    # 2) Group by original, and within each, split positives/negatives
    by_original = {}
    for ex in examples:
        by_original.setdefault(ex["original"], []).append(ex)

    # filter out originals with no positives (they don't help supervised learning)
    by_original = {
        orig: exs for orig, exs in by_original.items()
        if any(e["label"] == 1 for e in exs)
    }

    originals = list(by_original.keys())
    random.shuffle(originals)

    print(f"Original problems with at least 1 positive: {len(originals)}")

    n = len(originals)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    splits = {
        "train": originals[:n_train],
        "val":   originals[n_train:n_train+n_val],
        "test":  originals[n_train+n_val:],
    }

    def make_split(split_name, orig_list):
        rows = []
        for orig in orig_list:
            exs = by_original[orig]
            pos = [e for e in exs if e["label"] == 1]
            neg = [e for e in exs if e["label"] == 0]

            if not pos:
                continue

            for p in pos:
                # add the positive example
                text_pos = (
                    f"Original: {p['original']} "
                    f"[SEP] Variant: {p['variant']} "
                    f"[DIFF: {p['requested_difficulty']}]"
                )
                rows.append({
                    "text": text_pos,
                    "original": p["original"],
                    "variant": p["variant"],
                    "label": 1,
                })

                # sample a few negatives from same original if any
                if neg:
                    for _ in range(NEG_PER_POS):
                        neg_ex = random.choice(neg)
                        text_neg = (
                            f"Original: {p['original']} "
                            f"[SEP] Variant: {neg_ex['variant']} "
                            f"[DIFF: {neg_ex['requested_difficulty']}]"
                        )
                        rows.append({
                            "text": text_neg,
                            "original": p["original"],
                            "variant": neg_ex["variant"],
                            "label": 0,
                        })

        random.shuffle(rows)
        print(
            f"{split_name}: {len(rows)} examples "
            f"({sum(r['label']==1 for r in rows)} pos, "
            f"{sum(r['label']==0 for r in rows)} neg)"
        )
        out_path = OUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {out_path}")

    for split_name, orig_list in splits.items():
        make_split(split_name, orig_list)

if __name__ == "__main__":
    main()
