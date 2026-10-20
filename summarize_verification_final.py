#!/usr/bin/env python3
"""
Run the same style of numerical verification used in generate_variants
on the FINAL variants stored in a variants_all_*.json file.

- For each record, we look at rec["variant"] (the final chosen variant).
- We run a SymPy-based check similar in spirit to the one in generate_variants.py:
    * For definite integrals: we treat them as "verified" (True).
    * For indefinite integrals: we integrate symbolically with a timeout and
      differentiate to see if we recover the integrand.
- We print:
    * total number of variants
    * how many verified True vs False
    * total verification time and average per variant
"""

import json
import math
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp
import concurrent.futures


# === COPY of the numerical integration helpers (same style as generate_variants) ===

TIMEOUT_SECONDS = 1  # same as in your snippet

# Global process pool executor (matches the pattern from generate_variants)
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
    """Return True if the integral string contains definite limits of the form (x, a, b)."""
    pattern = r"integrate\([^,]+,\s*\(x,\s*[^,]+,\s*[^)]+\)\)"
    return re.search(pattern, integral_str) is not None


def verify_integral(integral_str: str) -> bool:
    """
    Verify an integral in the same spirit as generate_variants:

    - If it looks like a definite integral integrate(f(x), (x, a, b)):
        we **treat it as verified (True)**.
      (The LADDER code does not re-check definite results numerically.)

    - If it looks like an indefinite integral integrate(f(x), x):
        we try to integrate symbolically with a timeout and then differentiate
        the antiderivative to see if we get back f(x).

    If anything goes wrong (timeouts, parsing errors, etc.), we print a
    diagnostic and return False (non-verified).
    """
    x = sp.symbols("x")

    # Definite integral: LADDER's pipeline treats these as verified
    if is_definite_integral(integral_str):
        return True

    try:
        # Match "integrate(<integrand>, x)"
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            # Can't even parse as an integral in the expected format
            return False

        integrand_str = match.group(1)
        integrand = sp.sympify(integrand_str)

        # Try to integrate with a timeout via a separate process
        try:
            antideriv = run_integration(integrand, x, timeout=TIMEOUT_SECONDS)
        except Exception:
            print("Integration timed out in verify_integral; returning non-verified result.")
            return False

        # Differentiate and simplify the difference
        diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)

        # If the simplified difference is exactly zero, we treat it as verified
        return diff_expr == 0

    except Exception as e:
        print("Error verifying integral:", e)
        return False


# === MAIN ANALYSIS LOGIC ===


def summarize_final_variants(json_path: str) -> None:
    """
    Load variants_all_*.json, run verify_integral on each final variant,
    and print a summary plus timing info.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    verified_true = 0
    verified_false = 0

    start_time = time.perf_counter()

    for rec in data:
        variant = rec.get("variant")
        if not variant:
            continue

        total += 1
        t0 = time.perf_counter()
        result = verify_integral(variant)
        t1 = time.perf_counter()

        # Optionally, you could store this back onto the record here
        # rec["verification_passed_final_check"] = result
        # rec["verification_time_seconds"] = t1 - t0

        if result:
            verified_true += 1
        else:
            verified_false += 1

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / total if total > 0 else 0.0

    print(f"File: {json_path}")
    print(f"Total final variants checked: {total}")
    print(f"  Verified TRUE : {verified_true}")
    print(f"  Verified FALSE: {verified_false}")
    print()
    print(f"Total verification wall-clock time: {total_time:.3f} seconds")
    print(f"Average time per variant          : {avg_time:.4f} seconds")


def main():
    if len(sys.argv) != 2:
        print("Usage: python summarize_verification_final.py <variants_all_json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    summarize_final_variants(json_path)


if __name__ == "__main__":
    main()
