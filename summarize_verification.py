#!/usr/bin/env python
"""
summarize_verification_final.py

Run the same numerical verification procedure as in generate_variants
on the FINAL variants stored in a variants_all_*.json file.

- Uses SymPy-based integration with a 1s timeout (ProcessPoolExecutor)
- For definite integrals: treat as verified (True), like original code
- For indefinite integrals: integrate, differentiate, compare back
- Timeouts / errors => counted as not verified (False)

Usage:
    python summarize_verification_final.py path/to/variants_all_*.json
"""

import argparse
import concurrent.futures
import json
import re
import time
from typing import Any, Dict, List

import sympy as sp

# ---- COPIED NUMERICAL VERIFIER LOGIC (from generate_variants) ----

TIMEOUT_SECONDS = 1  # same as in generate_variants
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)


def integrate_wrapper(integrand: sp.Basic, x: sp.Symbol) -> sp.Basic:
    """Compute sp.integrate(integrand, x) for indefinite integrals."""
    return sp.integrate(integrand, x)


def run_integration(
    integrand: sp.Basic, x: sp.Symbol, timeout: int = TIMEOUT_SECONDS
) -> sp.Basic:
    """Run indefinite integration with a timeout."""
    future = executor.submit(integrate_wrapper, integrand, x)
    return future.result(timeout=timeout)


def is_definite_integral(integral_str: str) -> bool:
    """Return True if the integral string looks like integrate(..., (x, a, b))."""
    pattern = r"integrate\([^,]+,\s*\(x,\s*[^,]+,\s*[^)]+\)\)"
    return re.search(pattern, integral_str) is not None


def verify_integral(integral_str: str) -> bool:
    """
    Verify the antiderivative by checking that differentiating it gives back the integrand.
    (For definite integrals this check is skipped and we return True.)

    This is copied to match the behavior in generate_variants.
    """
    x = sp.symbols("x")
    if is_definite_integral(integral_str):
        # Original behavior: we don't attempt symbolic verification for definite integrals
        return True

    try:
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return False

        integrand_str = match.group(1)
        integrand = sp.sympify(integrand_str)
        try:
            antideriv = run_integration(integrand, x, timeout=TIMEOUT_SECONDS)
        except Exception:
            # Same log message as original code
            print("Integration timed out in verify_integral; returning non-verified result.")
            return False

        diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
        return diff_expr == 0
    except Exception as e:
        # Same behavior: log and treat as not verified
        print("Error verifying integral:", e)
        return False


# ---- ANALYSIS CODE FOR FINAL VARIANTS ----


def summarize_verification(path: str) -> None:
    # Load variants_all_*.json
    with open(path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total = 0
    n_true = 0
    n_false = 0
    times: List[float] = []

    for rec in data:
        variant = rec.get("variant")
        if not variant:
            continue

        total += 1

        start = time.perf_counter()
        ok = verify_integral(variant)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if ok:
            n_true += 1
        else:
            n_false += 1

    print(f"File: {path}")
    print(f"Total final variants checked: {total}")
    print(f"Verified TRUE : {n_true}")
    print(f"Verified FALSE: {n_false}")
    if total > 0:
        print(f"True rate      : {n_true / total:.3f}")

    if times:
        avg_t = sum(times) / len(times)
        max_t = max(times)
        min_t = min(times)
        # simple median
        sorted_t = sorted(times)
        mid = len(sorted_t) // 2
        if len(sorted_t) % 2 == 1:
            median_t = sorted_t[mid]
        else:
            median_t = 0.5 * (sorted_t[mid - 1] + sorted_t[mid])

        print()
        print("Verification time (seconds):")
        print(f"  avg   = {avg_t:.4f}")
        print(f"  median= {median_t:.4f}")
        print(f"  min   = {min_t:.4f}")
        print(f"  max   = {max_t:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_path",
        help="Path to variants_all_*.json to analyze (final variants only)",
    )
    args = parser.parse_args()
    summarize_verification(args.json_path)


if __name__ == "__main__":
    main()
