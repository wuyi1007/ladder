#!/usr/bin/env python3
"""
Analyze 2-shot pre-verifier behavior on variants_all_*.json.

It expects that each record looks like what generate_variants_preverifier.py
now produces, i.e.:

{
  "original": ...,
  "requested_difficulty": ...,
  "variant": <FINAL chosen variant>,
  "preverifier_prob": ...,
  "preverifier_threshold": 0.25,
  "num_regenerations": 0 or 1,
  "candidate_history": [
      {
        "variant": ...,
        "preverifier_prob": ...,
        "preverifier_decision": "keep"/"drop",
        "is_initial": true/false,
        "verification_passed": true/false/None
      },
      ...
  ],
  ...
}

This script computes:

  - Basic counts by num_regenerations.
  - Pass rates of FINAL candidates by num_regenerations.
  - Loss_True_fraction:
        (# initial candidates that are actually True but have p<threshold)
        / T_initial
    where T_initial = # initial candidates with verification_passed == True.

  - Gain_True_from_regen:
        #slots where the FINAL candidate is a regenerated one
        AND verification_passed == True.

    It also reports:
        * baseline_true_no_regen = #slots with num_regenerations == 0
                                   and initial verification_passed == True
        * ratios based on that baseline.

  - Approximate time saved:
        assume each *extra* candidate we did NOT numerically verify
        would cost 2 seconds. For a slot with 2 candidates (initial + regen),
        we only send ONE to the true numerical verifier, so we "save"
        one call of ~2 seconds.
"""

import json
import sys

DEFAULT_THRESHOLD = 0.25
ASSUMED_VERIFY_SECONDS = 2.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_preverifier_2shot.py <variants_all.json>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    n_slots = len(records)
    print("=== analyze_preverifier_2shot ===")
    print(f"File: {path}")
    print(f"Total final variants (slots): {n_slots}")

    # Basic stats by num_regenerations
    regen_buckets = {0: 0, 1: 0, ">1": 0}
    final_true_by_regen = {0: 0, 1: 0, ">1": 0}
    total_by_regen = {0: 0, 1: 0, ">1": 0}
    p_sum_by_regen = {0: 0.0, 1: 0.0, ">1": 0.0}

    # Loss/gain stats
    T_initial = 0  # # of initial candidates that are actually True
    T_final = 0    # # of final candidates that are actually True

    loss_true_count = 0  # initial True but p<threshold
    threshold_used = None

    # baseline: "no regen" accuracy = #slots with regen=0 and initial True
    baseline_true_no_regen = 0

    # gain: #slots where FINAL candidate is regenerated AND True
    regen_true_final = 0

    # time saving: each candidate in candidate_history that is NOT final
    # is a numerical-verifier call we didn't make in the main pipeline.
    dropped_candidates_total = 0

    for rec in records:
        num_regens = rec.get("num_regenerations", 0)
        if num_regens == 0:
            bucket = 0
        elif num_regens == 1:
            bucket = 1
        else:
            bucket = ">1"

        regen_buckets[bucket] += 1
        total_by_regen[bucket] += 1

        p_final = rec.get("preverifier_prob", None)
        if isinstance(p_final, (int, float)):
            p_sum_by_regen[bucket] += float(p_final)

        thr = rec.get("preverifier_threshold", DEFAULT_THRESHOLD)
        if threshold_used is None:
            threshold_used = thr

        cand_hist = rec.get("candidate_history", [])
        if not cand_hist:
            # If candidate_history is missing or empty, skip loss/gain stats
            continue

        # Find initial candidate
        initial_cand = None
        for c in cand_hist:
            if c.get("is_initial"):
                initial_cand = c
                break
        if initial_cand is None:
            initial_cand = cand_hist[0]

        # Find the candidate that matches the FINAL variant stored at top level
        final_expr = rec.get("variant")
        final_cand = None
        for c in cand_hist:
            if c.get("variant") == final_expr:
                final_cand = c
                break
        if final_cand is None:
            # Fallback: if something is off, treat initial as final
            final_cand = initial_cand

        init_v = initial_cand.get("verification_passed", None)
        final_v = final_cand.get("verification_passed", None)

        # Count final True
        if final_v is True:
            T_final += 1
            final_true_by_regen[bucket] += 1

        # Count initial True and loss of True
        if init_v is True:
            T_initial += 1

            # baseline accuracy "no regen" = use initial candidate only
            if num_regens == 0:
                baseline_true_no_regen += 1

            init_p = initial_cand.get("preverifier_prob", 0.0)
            if init_p < thr:
                # This is a "True" candidate that lies below threshold
                loss_true_count += 1

        # Gain: final candidate is regenerated AND True
        if num_regens >= 1:
            if final_cand.get("is_initial") is False and final_v is True:
                regen_true_final += 1

        # Time saving: candidate_history length - 1 = how many extra
        # candidates we *could* have numerically verified but didn't
        # in the main pipeline. (For 2-shot, this is 0 or 1.)
        dropped_candidates_total += max(0, len(cand_hist) - 1)

    # ----------------- Print basic stats -----------------
    print()
    print("Counts by num_regenerations:")
    print(f"  regen = 0 : {regen_buckets[0]}")
    print(f"  regen = 1 : {regen_buckets[1]}")
    print(f"  regen > 1 : {regen_buckets['>1']}")

    print("\nPass rates of FINAL candidate by num_regenerations (using verification_passed):")
    for bucket in [0, 1, ">1"]:
        tot = total_by_regen[bucket]
        true_ct = final_true_by_regen[bucket]
        rate = true_ct / tot if tot > 0 else 0.0
        avg_p = p_sum_by_regen[bucket] / tot if tot > 0 else 0.0
        print(
            f"  regen = {bucket}: "
            f"pass_rate = {rate:.3f}  "
            f"(true={true_ct}, total={tot}),  "
            f"avg_preverifier_prob = {avg_p:.3f}"
        )

    # ----------------- Loss / Gain of True -----------------
    print("\n=== Loss / Gain of True ===")
    print(f"Threshold used (τ): {threshold_used}")

    print(f"T_initial (initial candidates with verification_passed == True): {T_initial}")
    print(f"T_final   (FINAL candidates with verification_passed == True): {T_final}")

    # Loss: fraction of initial True that fall below threshold
    if T_initial > 0:
        loss_true_fraction = loss_true_count / T_initial
    else:
        loss_true_fraction = 0.0

    print(f"Loss_True_count (initial True with p < τ): {loss_true_count}")
    print(f"Loss_True_fraction = Loss_True_count / T_initial = {loss_true_fraction:.3f}")

    # Gain: how many *final* True answers come from regenerated variants
    print(f"\nBaseline_true_no_regen (regen=0 & initial True): {baseline_true_no_regen}")
    print(f"Regen_true_final (FINAL candidate is regenerated & True): {regen_true_final}")

    # Ratio #1: relative to baseline_no_regen (what you proposed earlier)
    if baseline_true_no_regen > 0:
        gain_ratio_baseline = regen_true_final / baseline_true_no_regen
    else:
        gain_ratio_baseline = 0.0

    # Ratio #2: relative to T_initial (alternative normalization)
    if T_initial > 0:
        gain_ratio_Tinitial = regen_true_final / T_initial
    else:
        gain_ratio_Tinitial = 0.0

    print(f"Gain_ratio_baseline = Regen_true_final / Baseline_true_no_regen = {gain_ratio_baseline:.3f}")
    print(f"Gain_ratio_Tinitial = Regen_true_final / T_initial = {gain_ratio_Tinitial:.3f}")

    # ----------------- Time saving estimate -----------------
    print("\n=== Approximate verification time saved ===")
    print(
        f"Dropped candidates total (non-final per slot): {dropped_candidates_total} "
        "(these would have been extra numerical-verifier calls)."
    )
    time_saved_sec = dropped_candidates_total * ASSUMED_VERIFY_SECONDS
    time_saved_per_slot = time_saved_sec / n_slots if n_slots > 0 else 0.0

    print(f"Assumed cost per extra call: {ASSUMED_VERIFY_SECONDS:.1f} seconds")
    print(f"Total time saved (approx): {time_saved_sec:.1f} seconds")
    print(f"Average time saved per slot: {time_saved_per_slot:.3f} seconds")

    print("\nDone.")


if __name__ == "__main__":
    main()
