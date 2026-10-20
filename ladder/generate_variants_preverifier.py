"""
Module for generating mathematical integration problem variants with a neural
pre-verifier.

This module:
  - Generates creative variants of integration problems with an LLM.
  - Uses a lightweight MiniLM-style pre-verifier to score variants.
  - Applies a 2-shot drop/keep scheme per "slot":
        * First candidate from the batch LLM call is scored.
        * If p >= 0.25 -> keep (no regeneration).
        * If p < 0.25  -> generate ONE new variant, score it,
                          then pick the better of the two by p.
  - Only the final chosen variant per slot goes through numerical / symbolic
    verification (so #numerical-verifier calls is unchanged vs baseline).
  - Logs pre-verifier probabilities, decisions, and candidate history for
    analysis.

Save this file as: generate_variants_preverifier.py
"""

import asyncio
import concurrent.futures
import json
import math
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import sympy as sp
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ladder.utils.inference import generate_text

# ---------------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 1

# If False, we skip symbolic integration and just return solution=None, evaluations={}
CALCULATE_SYMBOLIC = False

# Process pool for SymPy integration with timeouts
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

# ===== PRE-VERIFIER CONFIG (UPDATED) =====
PREVERIFIER_MODEL_DIR = "preverifier_model_from_variants_json_v2"

# Single boundary:
#   p < 0.25  -> "drop"
#   p >= 0.25 -> "keep"
PREVERIFIER_THRESHOLD = 0.25

_pre_tokenizer: Optional[AutoTokenizer] = None
_pre_model: Optional[AutoModelForSequenceClassification] = None
_pre_device: Optional[torch.device] = None


def _load_preverifier() -> None:
    """
    Lazy-load the MiniLM pre-verifier model + tokenizer.

    Assumes PREVERIFIER_MODEL_DIR contains the fine-tuned MiniLM checkpoint:
      - config.json
      - tokenizer files
      - model.safetensors / pytorch_model.bin
    """
    global _pre_tokenizer, _pre_model, _pre_device
    if _pre_model is not None and _pre_tokenizer is not None and _pre_device is not None:
        return

    _pre_tokenizer = AutoTokenizer.from_pretrained(PREVERIFIER_MODEL_DIR)
    _pre_model = AutoModelForSequenceClassification.from_pretrained(PREVERIFIER_MODEL_DIR)
    _pre_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _pre_model.to(_pre_device)
    _pre_model.eval()


@torch.no_grad()
def preverify_prob(
    original_integral: str,
    variant_integral: str,
    difficulty: str = "",
) -> float:
    """
    Compute p in [0, 1] = probability that this variant will pass the numerical verifier.

    Uses the same input text format as the pre-verifier training:
        "Original: <orig> [SEP] Variant: <var> [DIFF: difficulty]"

    Returns:
        float probability for label "1" (assumed to be "good / passes").
        On error, returns 0.5 as a neutral fallback.
    """
    try:
        _load_preverifier()
        assert _pre_model is not None and _pre_tokenizer is not None and _pre_device is not None

        text = (
            f"Original: {original_integral} [SEP] "
            f"Variant: {variant_integral} [DIFF: {difficulty}]"
        )
        inputs = _pre_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(_pre_device)

        outputs = _pre_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        # Assume binary classification: index 1 = "good"
        p_good = probs[0, 1].item()
        # Clamp just in case
        p_good = max(0.0, min(1.0, float(p_good)))
        return p_good
    except Exception as e:
        print(f"[preverify_prob] Warning: error during pre-verifier call: {e}")
        return 0.5


# ---------------------------------------------------------------------------
# SYMPY UTILITIES
# ---------------------------------------------------------------------------


def integrate_wrapper(integrand: sp.Basic, x: sp.Symbol) -> sp.Basic:
    """Compute sp.integrate(integrand, x) for indefinite integrals."""
    return sp.integrate(integrand, x)


def run_integration(
    integrand: sp.Basic, x: sp.Symbol, timeout: int = TIMEOUT_SECONDS
) -> sp.Basic:
    """Run indefinite integration with a timeout."""
    future = executor.submit(integrate_wrapper, integrand, x)
    return future.result(timeout=timeout)


def definite_integrate_wrapper(
    integrand: sp.Basic, x: sp.Symbol, lower: sp.Basic, upper: sp.Basic
) -> sp.Basic:
    """Compute sp.integrate(integrand, (x, lower, upper)) for definite integrals."""
    return sp.integrate(integrand, (x, lower, upper))


def run_definite_integration(
    integrand: sp.Basic,
    x: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    timeout: int = TIMEOUT_SECONDS,
) -> sp.Basic:
    """Run definite integration with a timeout."""
    future = executor.submit(definite_integrate_wrapper, integrand, x, lower, upper)
    return future.result(timeout=timeout)


def is_definite_integral(integral_str: str) -> bool:
    """Return True if the integral string contains definite limits of the form (x, a, b)."""
    pattern = r"integrate\([^,]+,\s*\(x,\s*[^,]+,\s*[^)]+\)\)"
    return re.search(pattern, integral_str) is not None


def verify_integral(integral_str: str) -> bool:
    """
    Verify the antiderivative by checking that differentiating it gives back the integrand.
    (For definite integrals this check is skipped.)
    """
    x = sp.symbols("x")
    if is_definite_integral(integral_str):
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
            print("Integration timed out in verify_integral; returning non-verified result.")
            return False

        diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
        return diff_expr == 0
    except Exception as e:
        print("Error verifying integral:", e)
        return False


def compute_solution_and_evals(
    integral_str: str,
    num_points: int = 3,
    lower: float = -10,
    upper: float = 10,
    tol: float = 1e-6,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Compute the antiderivative for an indefinite integral or the definite integral result
    if limits are provided.

    If CALCULATE_SYMBOLIC is False, we skip integration and return (None, {}).
    """
    if not CALCULATE_SYMBOLIC:
        return None, {}

    x = sp.symbols("x")
    if is_definite_integral(integral_str):
        # Process as a definite integral
        pattern = r"integrate\((.+),\s*\(x,\s*([^,]+),\s*([^)]+)\)\)"
        match = re.search(pattern, integral_str)
        if not match:
            return None, {}

        integrand_str = match.group(1)
        lower_limit_str = match.group(2)
        upper_limit_str = match.group(3)
        try:
            integrand = sp.sympify(integrand_str)
            lower_limit = sp.sympify(lower_limit_str)
            upper_limit = sp.sympify(upper_limit_str)
        except Exception as e:
            print("Error sympifying definite integral parts:", e)
            return None, {}

        try:
            result = run_definite_integration(
                integrand, x, lower_limit, upper_limit, timeout=TIMEOUT_SECONDS
            )
        except Exception:
            print(
                "Integration timed out in compute_solution_and_evals (definite); marking as too hard."
            )
            return None, {}

        solution_str = str(result)
        try:
            result_eval = float(result.evalf())
        except Exception:
            result_eval = None
        evaluations = {"definite_result": result_eval}
        return solution_str, evaluations
    else:
        # Process as an indefinite integral
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return None, {}

        integrand_str = match.group(1)
        try:
            integrand = sp.sympify(integrand_str)
        except Exception as e:
            print("Error sympifying integrand:", e)
            return None, {}

        try:
            antideriv = run_integration(integrand, x, timeout=TIMEOUT_SECONDS)
        except Exception:
            print(
                "Integration timed out in compute_solution_and_evals; marking as too hard."
            )
            return None, {}

        solution_str = str(antideriv)
        evaluations: Dict[str, Any] = {}
        attempts = 0
        max_attempts = num_points * 10
        while len(evaluations) < num_points and attempts < max_attempts:
            attempts += 1
            test_val = random.uniform(lower, upper)
            eval_val = antideriv.evalf(subs={x: test_val})
            if hasattr(eval_val, "as_real_imag"):
                re_val, im_val = eval_val.as_real_imag()
                if abs(im_val) < tol:
                    evaluations[round(test_val, 3)] = float(re_val)
            else:
                evaluations[round(test_val, 3)] = float(eval_val)
        return solution_str, evaluations


# ---------------------------------------------------------------------------
# LLM VARIANT PARSING & PROMPTING
# ---------------------------------------------------------------------------


def parse_variants(text: str) -> List[Dict[str, str]]:
    """
    Parse the LLM response text and extract a list of variant dictionaries.
    Each variant is expected to be delimited by "====" and include both
    a Reasoning and a Variant line.
    """
    variants: List[Dict[str, str]] = []
    blocks = re.split(r"====\s*", text)
    for block in blocks:
        if "Variant:" in block and "Reasoning:" in block:
            # Extract reasoning using regex
            reasoning_match = re.search(
                r"Reasoning:\s*(.*?)\s*Variant:", block, re.DOTALL
            )
            reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""

            # Split block into lines and find the line that starts with "Variant:"
            variant_expr = None
            for line in block.splitlines():
                if line.strip().startswith("Variant:"):
                    # Remove the 'Variant:' prefix and any extra whitespace
                    variant_expr = line.strip()[len("Variant:") :].strip()
                    break

            # Optional: if the variant expression doesn't end with a ')', skip it
            if variant_expr and variant_expr.endswith(")"):
                variants.append({"reasoning": reasoning_text, "variant": variant_expr})
    return variants


def get_random_prompt_template(
    integral_str: str, difficulty: str, count: int, transforms_text: str, personas_str: str
) -> str:
    """
    Return a prompt template that instructs the LLM to produce variants.
    If the base integral is definite then require variants in the form
      integrate(<integrand>, (x, lower_limit, upper_limit))
    otherwise use the indefinite form.
    """
    if is_definite_integral(integral_str):
        expected_format = "integrate(<integrand>, (x, lower_limit, upper_limit))"
    else:
        expected_format = "integrate(<integrand>, x)"

    templates: List[str] = []

    templates.append(
        f"Assume you can adopt various mathematical personas such as {personas_str}.\n\n"
        f"Given the integral: {integral_str}\n"
        f"Your task is to generate {count} creative and unexpected variant(s) that are {difficulty} than the original.\n\n"
        "Important constraints:\n"
        "- Always use 'x' as the variable of integration\n"
        "- Do not introduce arbitrary constants - all constants must have specific numerical values\n\n"
        "Follow these steps:\n"
        "1. Analyze the original integral deeply, looking for hidden patterns and non-obvious properties.\n"
        "2. Think outside conventional approaches – consider unusual substitutions, creative identities, or surprising transformations.\n"
        f"3. Draw inspiration from various mathematical fields. Some ideas: {transforms_text}\n"
        "4. Provide a detailed explanation of your creative reasoning process.\n"
        f"5. Present each variant in valid Python sympy syntax in the form: {expected_format}.\n\n"
        "Push yourself to find truly novel variants that might surprise even experienced mathematicians!\n\n"
        "Return your answer in the following exact format for each variant:\n"
        "====\n"
        "Variant <number>:\n"
        "Reasoning: <your creative chain-of-thought explanation>\n"
        f"Variant: {expected_format}\n"
        "===="
    )

    templates.append(
        f"Channel the creative spirit of great mathematicians like {personas_str}.\n\n"
        f"For this integral: {integral_str}\n"
        f"Create {count} mathematically interesting variant(s) that are {difficulty} than the original.\n\n"
        "Critical requirements:\n"
        "- Use 'x' as the integration variable - do not change this\n"
        "- All constants must be specific numbers (e.g., 2, 3.14, etc.) - no arbitrary constants\n\n"
        "Steps:\n"
        "1. Look for hidden mathematical beauty and unexpected connections in the original integral.\n"
        "2. Consider how different areas of mathematics might offer surprising approaches.\n"
        f"3. Experiment with transformations such as {transforms_text}\n"
        "4. Explain your mathematical insights and creative process.\n"
        f"5. Express each variant using valid Python sympy syntax: {expected_format}\n\n"
        "Aim to create variants that showcase the rich interconnections in mathematics!\n\n"
        "Use this format:\n"
        "====\n"
        "Variant <number>:\n"
        "Reasoning: <your creative chain-of-thought explanation>\n"
        f"Variant: {expected_format}\n"
        "===="
    )

    return random.choice(templates)


# ---------------------------------------------------------------------------
# TRANSFORMATION CATALOG
# ---------------------------------------------------------------------------

TRANSFORMATIONS_BY_DIFFICULTY: Dict[str, List[str]] = {
    "easier": [
        "simplify the denominator",
        "reduce exponents",
        "remove complex terms",
        "convert to partial fractions",
        "simplify trigonometric terms",
        "combine like terms",
        "factor common elements",
        "remove nested functions",
        "convert to standard forms",
        "linearize the integrand",
        "eliminate radicals",
        "reduce polynomial degree",
        "split compound fractions",
        "remove logarithmic terms",
        "substitute simpler functions",
    ],
    "equivalent": [
        "change a function to a different but equivalent one",
        "change coefficient values slightly",
        "alter constant terms",
        "modify an exponent slightly",
        "rewrite the integrand in a different form without changing overall complexity",
        "exchange similar functions (e.g., sin to cos)",
        "adjust parameters while keeping the integral equivalent",
        "rearrange the order of terms",
        "use trigonometric identities to rewrite the expression",
        "substitute equivalent exponential forms",
        "distribute terms differently",
        "factor common terms in a new way",
        "rewrite using alternate algebraic forms",
        "swap numerator and denominator with reciprocal",
        "use alternate but equivalent radical forms",
        "rewrite using different logarithmic properties",
        "apply integration by substitution with a trivial substitution",
        "apply partial fractions in an equivalent manner",
        "rationalize the integrand slightly",
    ],
    "harder": [
        "increase complexity of terms",
        "add additional factors",
        "introduce higher degrees",
        "combine multiple functions",
        "add nested functions",
        "use function composition",
        "incorporate transcendental functions",
        "mix different function types",
        "increase algebraic complexity",
        "add non-elementary functions",
        "use more complex substitutions",
        "increase number of terms",
        "add rational complexity",
        "combine different mathematical concepts",
        "increase structural complexity",
    ],
}


# ---------------------------------------------------------------------------
# VARIANT GENERATION (LLM CALLS)
# ---------------------------------------------------------------------------


async def generate_single_raw_variant(
    integral_str: str,
    difficulty: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate ONE new raw variant (without numerical verification).
    Used when the first candidate is "drop" (p < threshold).

    Returns:
      {
        "variant": "<integrate(...)>",
        "reasoning": "...",
        "transformations_used": [...]
      }
    or None if parsing failed.
    """
    transformations = TRANSFORMATIONS_BY_DIFFICULTY.get(difficulty.lower(), [])
    if not transformations:
        transformations = ["make a small change"]

    num_choices = random.choice(range(3, 7))
    chosen_transforms = random.sample(
        transformations, min(num_choices, len(transformations))
    )
    transforms_text = ", ".join(chosen_transforms)

    personas = [
        "Richard Feynman who loves finding intuitive physical interpretations",
        "Leonhard Euler who excels at infinite series and creative substitutions",
        "Carl Friedrich Gauss who sees deep mathematical patterns",
        "Emmy Noether who focuses on symmetry and invariance",
        "Paul Dirac who prefers elegant mathematical beauty",
        "Isaac Newton who thinks in terms of physical motion and rates of change",
        "Gottfried Leibniz who seeks systematic notation and patterns",
        "Bernhard Riemann who explores complex geometric relationships",
        "Pierre-Simon Laplace who excels at transform methods",
        "Joseph-Louis Lagrange who loves analytical mechanics approaches",
        "Henri Poincaré who sees topological connections",
        "Srinivasa Ramanujan who has incredible intuition for identities",
        "David Hilbert who approaches problems with rigorous formalism",
        "John von Neumann who combines computational and theoretical insights",
        "Sophie Germain who finds innovative prime number relationships",
        "George Pólya who uses creative problem-solving strategies",
        "Augustin-Louis Cauchy who emphasizes rigorous analysis",
        "Évariste Galois who sees algebraic structure in everything",
        "Ada Lovelace who thinks algorithmically",
        "Alan Turing who approaches problems computationally",
        "Kurt Gödel who seeks logical foundations",
        "Edward Witten who applies physics insights to mathematics",
        "Terence Tao who combines multiple mathematical disciplines",
        "Katherine Johnson who excels at practical numerical computations",
        "Maryam Mirzakhani who thinks in terms of geometric dynamics",
    ]
    personas_str = ", ".join(personas)

    prompt_variant = get_random_prompt_template(
        integral_str, difficulty, 1, transforms_text, personas_str
    )

    temperature_choice = random.choice([0.8, 1.0, 1.2, 1.4])
    response_text = await generate_text(
        MODEL, prompt_variant, temperature=temperature_choice
    )

    parsed = parse_variants(response_text)
    if not parsed:
        return None

    v = parsed[0]
    v["transformations_used"] = chosen_transforms
    return v


# async def process_single_variant(
#     original_integral: str, difficulty: str, variant_data: Dict[str, Any]
# ) -> Optional[Dict[str, Any]]:
#     """
#     Process one variant *slot* with a 2-shot drop/keep scheme.

#     Behavior:
#       1) Take initial candidate (from LLM batch).
#          - Compute p via pre-verifier.
#          - If p >= 0.25: decision = "keep", we accept this candidate,
#            no second generation.

#          - If p < 0.25: decision = "drop", we generate ONE new candidate,
#            score it, and then pick the better of the two by p.

#       2) Exactly ONE final candidate (best p among the 1 or 2) goes to
#          numerical/symbolic verification, so # of numerical verifier calls
#          stays the same as baseline.

#     We also record a small history of the candidates for analysis.
#     """
#     initial_variant_expr = variant_data.get("variant")
#     if not initial_variant_expr:
#         return None

#     candidate_history: List[Dict[str, Any]] = []
#     used_exprs = set()

#     def label_from_p(p: float) -> str:
#         return "keep" if p >= PREVERIFIER_THRESHOLD else "drop"

#     async def eval_candidate(vdata: Dict[str, Any], is_initial: bool) -> Optional[Dict[str, Any]]:
#         vexpr = vdata.get("variant")
#         if not vexpr or vexpr in used_exprs:
#             return None
#         used_exprs.add(vexpr)

#         p = preverify_prob(
#             original_integral=original_integral,
#             variant_integral=vexpr,
#             difficulty=difficulty,
#         )
#         decision = label_from_p(p)
#         cand = {
#             "variant": vexpr,
#             "reasoning": vdata.get("reasoning"),
#             "transformations_used": vdata.get("transformations_used", []),
#             "preverifier_prob": p,
#             "preverifier_decision": decision,  # "keep" or "drop"
#             "is_initial": is_initial,
#         }
#         candidate_history.append(cand)
#         return cand

#     # --- 1) First candidate (from batch LLM call) ---
#     first_cand = await eval_candidate(variant_data, is_initial=True)
#     if first_cand is None:
#         return None

#     final_candidate = first_cand
#     num_regenerations = 0

#     # --- 2) Regenerate ONCE if first was "drop" ---
#     if first_cand["preverifier_decision"] == "drop":
#         regen_raw = await generate_single_raw_variant(original_integral, difficulty)
#         if regen_raw is not None:
#             second_cand = await eval_candidate(regen_raw, is_initial=False)
#             if second_cand is not None:
#                 num_regenerations = 1
#                 if second_cand["preverifier_prob"] > final_candidate["preverifier_prob"]:
#                     final_candidate = second_cand

#     # --- 3) Run numerical / symbolic pipeline on FINAL candidate only ---
#     final_expr = final_candidate["variant"]
#     final_reasoning = final_candidate["reasoning"]
#     final_transforms = final_candidate["transformations_used"]
#     final_p = final_candidate["preverifier_prob"]
#     final_decision = final_candidate["preverifier_decision"]

#     solution, evaluations = compute_solution_and_evals(final_expr)

#     x = sp.symbols("x")
#     definite_pattern = r"integrate\((.+),\s*\(x,\s*([^,]+),\s*([^)]+)\)\)"
#     if re.search(definite_pattern, final_expr):
#         verification = True
#     else:
#         pattern = r"integrate\((.+),\s*x\)"
#         match = re.search(pattern, final_expr)
#         if match:
#             try:
#                 integrand_str = match.group(1)
#                 integrand = sp.sympify(integrand_str)
#                 if solution is not None:
#                     antideriv = sp.sympify(solution)
#                     diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
#                     verification = diff_expr == 0
#                 else:
#                     verification = None
#             except Exception:
#                 verification = None
#         else:
#             verification = None

#     # --- 4) Return final record for this slot ---
#     return {
#         "original": original_integral,
#         "requested_difficulty": difficulty,
#         "variant": final_expr,
#         "reasoning": final_reasoning,
#         "variant_response": None,
#         "verification_passed": verification,
#         "evaluation": None,
#         "transformations_used": final_transforms,
#         "solution": solution,
#         "evaluations": evaluations,
#         "timestamp": datetime.utcnow().isoformat() + "Z",
#         # Pre-verifier info for FINAL candidate:
#         "preverifier_prob": final_p,
#         "preverifier_decision": final_decision,           # "keep" or "drop"
#         "preverifier_threshold": PREVERIFIER_THRESHOLD,
#         "num_regenerations": num_regenerations,
#         # History of attempts (1 or 2 candidates) for analysis:
#         "candidate_history": candidate_history,
#     }

async def process_single_variant(
    original_integral: str, difficulty: str, variant_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Process one variant *slot* with a 2-shot drop/keep scheme.

    Behavior:
      1) Take initial candidate (from LLM batch).
         - Compute p via pre-verifier.
         - If p >= 0.25: decision = "keep", we accept this candidate,
           no second generation.

         - If p < 0.25: decision = "drop", we generate ONE new candidate,
           score it, and then pick the better of the two by p.

      2) Exactly ONE final candidate (best p among the 1 or 2) goes to
         numerical/symbolic verification (same logic as before).

    NEW: we also run the existing verify_integral(...) on every candidate in
    candidate_history to attach verification_passed labels for analysis.
    """
    initial_variant_expr = variant_data.get("variant")
    if not initial_variant_expr:
        return None

    candidate_history: List[Dict[str, Any]] = []
    used_exprs = set()

    def label_from_p(p: float) -> str:
        return "keep" if p >= PREVERIFIER_THRESHOLD else "drop"

    async def eval_candidate(vdata: Dict[str, Any], is_initial: bool) -> Optional[Dict[str, Any]]:
        vexpr = vdata.get("variant")
        if not vexpr or vexpr in used_exprs:
            return None
        used_exprs.add(vexpr)

        p = preverify_prob(
            original_integral=original_integral,
            variant_integral=vexpr,
            difficulty=difficulty,
        )
        decision = label_from_p(p)
        cand = {
            "variant": vexpr,
            "reasoning": vdata.get("reasoning"),
            "transformations_used": vdata.get("transformations_used", []),
            "preverifier_prob": p,
            "preverifier_decision": decision,  # "keep" or "drop"
            "is_initial": is_initial,
            # verification_passed will be filled later
        }
        candidate_history.append(cand)
        return cand

    # --- 1) First candidate (from batch LLM call) ---
    first_cand = await eval_candidate(variant_data, is_initial=True)
    if first_cand is None:
        return None

    final_candidate = first_cand
    num_regenerations = 0

    # --- 2) Regenerate ONCE if first was "drop" ---
    if first_cand["preverifier_decision"] == "drop":
        regen_raw = await generate_single_raw_variant(original_integral, difficulty)
        if regen_raw is not None:
            second_cand = await eval_candidate(regen_raw, is_initial=False)
            if second_cand is not None:
                num_regenerations = 1
                if second_cand["preverifier_prob"] > final_candidate["preverifier_prob"]:
                    final_candidate = second_cand

    # --- 3) Run numerical / symbolic pipeline on FINAL candidate only (UNCHANGED) ---
    final_expr = final_candidate["variant"]
    final_reasoning = final_candidate["reasoning"]
    final_transforms = final_candidate["transformations_used"]
    final_p = final_candidate["preverifier_prob"]
    final_decision = final_candidate["preverifier_decision"]

    solution, evaluations = compute_solution_and_evals(final_expr)

    x = sp.symbols("x")
    definite_pattern = r"integrate\((.+),\s*\(x,\s*([^,]+),\s*([^)]+)\)\)"
    if re.search(definite_pattern, final_expr):
        verification = True
    else:
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, final_expr)
        if match:
            try:
                integrand_str = match.group(1)
                integrand = sp.sympify(integrand_str)
                if solution is not None:
                    antideriv = sp.sympify(solution)
                    diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
                    verification = diff_expr == 0
                else:
                    verification = None
            except Exception:
                verification = None
        else:
            verification = None

    # --- 4) NEW: use EXISTING verify_integral(...) on ALL candidates in candidate_history ---
    # We reuse the already-computed verification for the final expression to avoid
    # re-doing work, but the function used is the same numerical verifier logic.
    verif_cache: Dict[str, Optional[bool]] = {final_expr: verification}
    augmented_history: List[Dict[str, Any]] = []

    for cand in candidate_history:
        expr = cand["variant"]
        if expr in verif_cache:
            vlabel = verif_cache[expr]
        else:
            # This uses the existing numerical verifier defined above.
            vlabel = verify_integral(expr)
            verif_cache[expr] = vlabel

        cand_with_verif = dict(cand)
        cand_with_verif["verification_passed"] = vlabel
        augmented_history.append(cand_with_verif)

    candidate_history = augmented_history
    # --- end of NEW block ---

    # --- 5) Return final record for this slot ---
    return {
        "original": original_integral,
        "requested_difficulty": difficulty,
        "variant": final_expr,
        "reasoning": final_reasoning,
        "variant_response": None,
        "verification_passed": verification,   # final candidate
        "evaluation": None,
        "transformations_used": final_transforms,
        "solution": solution,
        "evaluations": evaluations,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "preverifier_prob": final_p,
        "preverifier_decision": final_decision,
        "preverifier_threshold": PREVERIFIER_THRESHOLD,
        "num_regenerations": num_regenerations,
        "candidate_history": candidate_history,  # now includes verification_passed
    }



async def generate_variant_chunk(
    integral_str: str, difficulty: str, count: int
) -> List[Dict[str, Any]]:
    """
    Generate a chunk (up to `count`) of variants in a single LLM call,
    then run the pre-verifier + 2-shot scheme + numerical pipeline for each.

    This is used by process_integral as the basic unit of work per difficulty.
    """
    transformations = TRANSFORMATIONS_BY_DIFFICULTY.get(difficulty.lower(), [])
    if not transformations:
        transformations = ["make a small change"]

    num_choices = random.choice(range(3, 7))
    chosen_transforms = random.sample(
        transformations, min(num_choices, len(transformations))
    )
    transforms_text = ", ".join(chosen_transforms)

    personas = [
        "Richard Feynman who loves finding intuitive physical interpretations",
        "Leonhard Euler who excels at infinite series and creative substitutions",
        "Carl Friedrich Gauss who sees deep mathematical patterns",
        "Emmy Noether who focuses on symmetry and invariance",
        "Paul Dirac who prefers elegant mathematical beauty",
        "Isaac Newton who thinks in terms of physical motion and rates of change",
        "Gottfried Leibniz who seeks systematic notation and patterns",
        "Bernhard Riemann who explores complex geometric relationships",
        "Pierre-Simon Laplace who excels at transform methods",
        "Joseph-Louis Lagrange who loves analytical mechanics approaches",
        "Henri Poincaré who sees topological connections",
        "Srinivasa Ramanujan who has incredible intuition for identities",
        "David Hilbert who approaches problems with rigorous formalism",
        "John von Neumann who combines computational and theoretical insights",
        "Sophie Germain who finds innovative prime number relationships",
        "George Pólya who uses creative problem-solving strategies",
        "Augustin-Louis Cauchy who emphasizes rigorous analysis",
        "Évariste Galois who sees algebraic structure in everything",
        "Ada Lovelace who thinks algorithmically",
        "Alan Turing who approaches problems computationally",
        "Kurt Gödel who seeks logical foundations",
        "Edward Witten who applies physics insights to mathematics",
        "Terence Tao who combines multiple mathematical disciplines",
        "Katherine Johnson who excels at practical numerical computations",
        "Maryam Mirzakhani who thinks in terms of geometric dynamics",
    ]
    personas_str = ", ".join(personas)

    prompt_variant = get_random_prompt_template(
        integral_str, difficulty, count, transforms_text, personas_str
    )
    temperature_choice = random.choice([0.8, 1.0, 1.2, 1.4])
    response_text = await generate_text(
        MODEL, prompt_variant, temperature=temperature_choice
    )

    parsed_variants = parse_variants(response_text)
    for v in parsed_variants:
        v["transformations_used"] = chosen_transforms

    # Process each parsed variant with pre-verifier + numerical pipeline
    tasks = [
        process_single_variant(original_integral=integral_str, difficulty=difficulty, variant_data=v)
        for v in parsed_variants
    ]
    processed_variants = await asyncio.gather(*tasks)
    return [v for v in processed_variants if v is not None]


# ---------------------------------------------------------------------------
# TOP-LEVEL API: process_integral
# ---------------------------------------------------------------------------


async def process_integral(
    integral_str: str,
    difficulties: Union[List[str], Dict[str, int]],
    num_variants: int = 3,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Generate a batch of variants for the given integral and for each difficulty.

    Supports:
      - difficulties = ["easier", "equivalent"]
          -> `num_variants` variants for each listed difficulty.
      - difficulties = {"easier": 14, "equivalent": 6}
          -> per-difficulty target counts.

    Strategy:
      - For each difficulty d with target N_d:
          * We request a buffered number of variants from the LLM
            across several chunks (each chunk requests up to 10).
          * Each raw variant is processed via pre-verifier (with 2-shot
            regeneration when p < threshold) + numerical pipeline.
          * We keep up to N_d unique variants per difficulty.
    """
    final_results: List[Dict[str, Any]] = []
    seen_variants: set = set()
    buffer_multiplier = 3  # oversample by this factor, then dedupe

    # Normalize difficulties -> list of (difficulty, target_count)
    if isinstance(difficulties, dict):
        diff_items = list(difficulties.items())
    else:
        diff_items = [(d, num_variants) for d in difficulties]

    # Build tasks
    tasks: List[Tuple[str, asyncio.Future]] = []
    for difficulty, target_count in diff_items:
        if target_count <= 0:
            continue
        total_to_request = target_count * buffer_multiplier
        num_chunks = math.ceil(total_to_request / 10)
        for i in range(num_chunks):
            count = (
                10
                if i < num_chunks - 1
                else (total_to_request - 10 * (num_chunks - 1))
            )
            tasks.append(
                (difficulty, generate_variant_chunk(integral_str, difficulty, count))
            )

    # Run all chunks
    chunk_results = await asyncio.gather(*[t[1] for t in tasks])

    # Collect per difficulty, enforcing uniqueness across all difficulties
    difficulty_dict: Dict[str, List[Dict[str, Any]]] = {
        d: [] for d, _ in diff_items
    }

    for idx, (difficulty, _) in enumerate(tasks):
        for variant in chunk_results[idx]:
            variant_expr = variant.get("variant")
            if variant_expr and variant_expr not in seen_variants:
                seen_variants.add(variant_expr)
                difficulty_dict[difficulty].append(variant)

    # Truncate per difficulty to target counts
    for difficulty, target_count in diff_items:
        chosen = difficulty_dict.get(difficulty, [])[:target_count]
        final_results.extend(chosen)

    return final_results


# ---------------------------------------------------------------------------
# MAIN (for local testing)
# ---------------------------------------------------------------------------


async def _demo() -> None:
    """Simple demo for manual testing."""
    base_integral = "integrate(1/(x**2 - x + 1), (x, 0, 1))"
    difficulties = {"easier": 3, "equivalent": 3}
    print("Processing integral:", base_integral)
    variants = await process_integral(base_integral, difficulties, num_variants=3)

    with open("variants_preverifier_demo.json", "w") as outfile:
        json.dump(variants, outfile, indent=2)

    for idx, v in enumerate(variants, start=1):
        print(f"\n--- Variant {idx} ---")
        print("Requested difficulty:", v["requested_difficulty"])
        print("Variant integral:", v["variant"])
        print("preverifier_prob:", v.get("preverifier_prob"))
        print("preverifier_decision:", v.get("preverifier_decision"))
        print("num_regenerations:", v.get("num_regenerations"))
        print("verification_passed:", v["verification_passed"])
        print("solution:", v["solution"])


if __name__ == "__main__":
    asyncio.run(_demo())
