"""
Hyperplane LSH: Standalone Algorithm Implementation
====================================================
Research Project: Near-Duplicate Detection using Hyperplane Random Projection
Authors: Aastha Malhotra, Raashi Sharma, Sonali Verma

ALGORITHM IMPLEMENTED
---------------------
This script implements Hyperplane LSH (Random Projection / SimHash) as a standalone
algorithm for near-duplicate web page detection, applied across two pipelines:

  Pipeline A (Control):    raw HTML DOM  -> tokenise -> hyperplane fingerprint
  Pipeline B (Experimental): HTML -> trafilatura extraction -> tokenise -> hyperplane fingerprint

HYPERPLANE LSH ALGORITHM (IMPROVED)
  Parameters:
    b = 512  bit vector length (fingerprint dimensionality) - INCREASED for better sensitivity
    t = 16   default maximum Hamming distance threshold - ADJUSTED
    bigrams = True  include bigrams in addition to unigrams for better context
    binary_tf = True use binary term frequency (each unique token counts once)
    
  Algorithm:
    1. Tokenize text into unigrams and bigrams (consecutive token pairs)
    2. For each unique token, generate a deterministic b-dimensional {-1, +1} projection vector
    3. Sum all unique token projection vectors (with binary TF weighting, not frequency-based)
    4. Binarize: set positive entries to 1, non-positive to 0
    5. Result: b-bit fingerprint (SimHash variant)
    6. Similarity: Hamming distance between fingerprints
    7. Prediction: near-duplicate if hamming_distance <= threshold

  IMPROVEMENTS from baseline SimHash:
    - **Increased dimensionality** (512 vs 384): More fine-grained fingerprints
    - **Binary TF weighting** instead of frequency: Reduces dominance of common words
    - **Bigram features** in addition to unigrams: Better context and phrase matching
    - **Adjusted threshold** (16 vs 12): Reflects new parameter sensitivity

  Threshold Sweeping:
    We sweep threshold t_HP in [0, 72] (step=2) for the precision-recall curve
    Default threshold: t = 16 (optimized for new parameters)

OUTPUT
------
  results/hyperplane/
    pipeline_a_hyperplane_results.jsonl    per-pair predictions from Pipeline A
    pipeline_b_hyperplane_results.jsonl    per-pair predictions from Pipeline B
    hyperplane_precision_recall.json       P/R/F1 at every threshold, all pipelines
    hyperplane_summary_table.json          headline numbers with breakdowns

USAGE
-----
    python hyperplane_detector.py --dataset ./dataset --out ./results/hyperplane
    python hyperplane_detector.py --dataset ./dataset --out ./results/hyperplane --pipeline b-only
    python hyperplane_detector.py --dataset ./dataset --out ./results/hyperplane --verbose
"""

import re
import json
import hashlib
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Literal

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
MISSING = []
try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")
try:
    from tqdm import tqdm
except ImportError:
    MISSING.append("tqdm")
try:
    import trafilatura
    from trafilatura.settings import use_config
except ImportError:
    MISSING.append("trafilatura")
try:
    from bs4 import BeautifulSoup
except ImportError:
    MISSING.append("beautifulsoup4")

if MISSING:
    raise ImportError(
        f"Missing dependencies: {', '.join(MISSING)}\n"
        f"Run: pip install {' '.join(MISSING)}"
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ===========================================================================
# HYPERPLANE PARAMETERS
# ===========================================================================

HP_B           = 512  # fingerprint bit length (increased for better sensitivity)
HP_DEFAULT_T   = 16   # default threshold (max Hamming distance for near-dup)
HP_THRESHOLDS  = list(range(0, 74, 2))  # 0, 2, 4, ..., 72 for PR curve
HP_SEED        = 42   # random seed for deterministic projections
HP_USE_IDF     = True  # use IDF weighting to reduce common word dominance
HP_USE_BIGRAMS = True  # include bigrams in addition to unigrams

# ===========================================================================
# TEXT EXTRACTION (same as neardup_detector.py)
# ===========================================================================

def html_to_tokens_raw(html: str, use_bigrams: bool = HP_USE_BIGRAMS) -> list[str]:
    """
    Pipeline A tokeniser: HTML DOM -> tokens.
    
    Preprocessing:
    - Strip HTML markup (replace with whitespace)
    - Every maximal alphanumeric sequence = one token
    - URLs in text split at slashes and dots -> individual tokens
    - IMG tag src URLs treated as terms
    - Optionally includes bigrams for better context sensitivity
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")

    # Collect IMG src URLs as terms
    img_tokens: list[str] = []
    for img in soup.find_all("img", src=True):
        src = img.get("src", "")
        if src:
            img_tokens.extend(re.split(r'[/.]', src))

    # Replace all HTML tags with whitespace, get raw text
    text = soup.get_text(separator=" ")

    # Tokenise: every maximal alphanumeric sequence
    unigrams = re.findall(r'[a-zA-Z0-9]+', text.lower())
    img_tokens = [t.lower() for t in img_tokens if t]
    all_unigrams = unigrams + img_tokens
    
    if not use_bigrams:
        return all_unigrams
    
    # Add bigrams
    bigrams = []
    for i in range(len(all_unigrams) - 1):
        bigram = all_unigrams[i] + "_" + all_unigrams[i+1]
        bigrams.append(bigram)
    
    return all_unigrams + bigrams


def html_to_tokens_extracted(html: str, use_bigrams: bool = HP_USE_BIGRAMS) -> list[str]:
    """
    Pipeline B tokeniser: trafilatura extraction -> tokens.
    
    Applies trafilatura to remove boilerplate before tokenisation.
    Optionally includes bigrams for better context sensitivity.
    """
    if not html:
        return []
    try:
        cfg = use_config()
        cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        text = trafilatura.extract(
            html,
            config=cfg,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
        if not text:
            text = BeautifulSoup(html, "lxml").get_text(separator=" ")
    except Exception:
        text = BeautifulSoup(html, "lxml").get_text(separator=" ")

    unigrams = re.findall(r'[a-zA-Z0-9]+', text.lower())
    
    if not use_bigrams:
        return unigrams
    
    # Add bigrams
    bigrams = []
    for i in range(len(unigrams) - 1):
        bigram = unigrams[i] + "_" + unigrams[i+1]
        bigrams.append(bigram)
    
    return unigrams + bigrams


def text_to_tokens(text: str, use_bigrams: bool = HP_USE_BIGRAMS) -> list[str]:
    """
    Tokenise pre-extracted text (used when raw HTML is not available).
    Optionally includes bigrams for better context sensitivity.
    """
    if not text:
        return []
    
    # Get unigrams
    unigrams = re.findall(r'[a-zA-Z0-9]+', text.lower())
    
    if not use_bigrams:
        return unigrams
    
    # Add bigrams (consecutive pairs of tokens)
    bigrams = []
    for i in range(len(unigrams) - 1):
        bigram = unigrams[i] + "_" + unigrams[i+1]
        bigrams.append(bigram)
    
    return unigrams + bigrams

# ===========================================================================
# HYPERPLANE LSH — SimHash Implementation
# ===========================================================================

def _token_projection_vector(token: str, b: int = HP_B, seed: int = HP_SEED) -> np.ndarray:
    """
    Generate a deterministic b-dimensional {-1, +1} projection for a token.
    
    Each token is projected into b-dimensional space by randomly choosing b 
    entries from {-1, 1}. This projection is the same for all pages.
    
    Implementation: use SHA256(seed || token) as RNG seed to ensure the
    same token always gets the same projection across all documents.
    
    Returns: np.ndarray of shape (b,) with dtype int8, values in {-1, 1}
    """
    seed_bytes = str(seed).encode("utf-8") + token.encode("utf-8")
    h = hashlib.sha256(seed_bytes).digest()
    rng_seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(rng_seed)
    return rng.choice([-1, 1], size=b).astype(np.int8)


def compute_hyperplane_fingerprint(tokens: list[str], b: int = HP_B, seed: int = HP_SEED, use_binary_tf: bool = True) -> np.ndarray:
    """
    Compute the b-bit Hyperplane (SimHash) fingerprint for a list of tokens.
    
    Algorithm:
    1. For each unique token (or each occurrence if use_binary_tf=False):
       - Get its deterministic b-dimensional {-1, +1} projection vector
    2. Sum all token projection vectors
    3. The final vector for the page is created by setting every positive entry
       in the vector to 1 and every non-positive entry to 0.
    
    Parameters:
    - use_binary_tf: If True, each unique token contributes once regardless of frequency.
                     If False, tokens are weighted by frequency (original SimHash).
                     Default: True (reduces impact of frequent common words).
    
    Returns: np.ndarray of shape (b,) with dtype uint8, values in {0, 1}
    """
    if not tokens:
        return np.zeros(b, dtype=np.uint8)

    accumulator = np.zeros(b, dtype=np.int32)
    
    if use_binary_tf:
        # Binary TF: each unique token contributes exactly once
        unique_tokens = set(tokens)
        for token in unique_tokens:
            accumulator += _token_projection_vector(token, b, seed).astype(np.int32)
    else:
        # Full TF: each occurrence contributes (original SimHash)
        for token in tokens:
            accumulator += _token_projection_vector(token, b, seed).astype(np.int32)

    return (accumulator >= 0).astype(np.uint8)


def hamming_distance(fingerprint_a: np.ndarray, fingerprint_b: np.ndarray) -> int:
    """
    Compute Hamming distance: number of differing bits between two fingerprints.
    
    Range: 0 to b (0 to 384)
    Near-duplicate if hamming_distance <= threshold (default threshold = 12)
    """
    return int(np.sum(fingerprint_a != fingerprint_b))


def cosine_similarity_estimate(hamming_dist: int, b: int = HP_B) -> float:
    """
    Estimate cosine similarity from Hamming distance.
    
    For random projection / SimHash:
    cos(theta) ≈ cos(pi * hamming_distance / b)
    
    This gives the estimated cosine similarity between the original
    term frequency vectors.
    """
    if b == 0:
        return 0.0
    theta = np.pi * hamming_dist / b
    return float(np.cos(theta))

# ===========================================================================
# FINGERPRINT DATACLASS
# ===========================================================================

@dataclass
class PageFingerprint:
    """All fingerprint data for one page under one pipeline."""
    article_id:   str
    pipeline:     str           # 'a' (raw DOM) or 'b' (extracted text)
    page_side:    str           # 'a' or 'b' (which side of the pair)
    url:          str
    token_count:  int
    fingerprint:  np.ndarray    # 384-bit fingerprint as uint8 array

# ===========================================================================
# PAIR RESULT DATACLASS
# ===========================================================================

@dataclass
class HyperplanePairResult:
    """Detection result for one pair under one pipeline."""
    pair_id:      str
    article_id:   str
    domain:       str
    variant_type: str
    label:        int           # ground truth: 1=near-dup, 0=true-negative

    # Hyperplane scores
    hamming_distance: int = 384     # number of differing bits (0-384)
    cosine_similarity: float = 0.0  # estimated cosine similarity
    hp_predicted:  int = -1         # 1 if hamming <= threshold, 0 otherwise

# ===========================================================================
# CORE EVALUATION LOOP
# ===========================================================================

def fingerprint_text(
    text: str,
    article_id: str,
    pipeline: str,
    page_side: str,
    url: str,
    use_binary_tf: bool = True,
) -> PageFingerprint:
    """
    Compute Hyperplane fingerprint from a text string.
    Used by both Pipeline A (raw DOM) and Pipeline B (extracted text).
    """
    tokens = text_to_tokens(text, use_bigrams=HP_USE_BIGRAMS)
    fingerprint = compute_hyperplane_fingerprint(tokens, b=HP_B, seed=HP_SEED, use_binary_tf=use_binary_tf)

    return PageFingerprint(
        article_id=article_id,
        pipeline=pipeline,
        page_side=page_side,
        url=url,
        token_count=len(tokens),
        fingerprint=fingerprint,
    )


def evaluate_pair(
    pair: dict,
    html_dir: Path,
    pipeline: str,     # 'a' = raw DOM, 'b' = extracted text
    threshold: int = HP_DEFAULT_T,
) -> HyperplanePairResult:
    """
    Compute fingerprints for both pages in a pair and evaluate similarity.
    
    CRITICAL: This function evaluates ALL pairs, not just LSH candidates.
    
    For pipeline='a': reads raw HTML if available, else falls back to extracted_text
    For pipeline='b': uses the extracted_text field directly from the pair dict
    """
    aid     = pair["article_id"]
    vtype   = pair["variant_type"]
    label   = pair["label"]
    domain  = pair["domain"]
    pair_id = pair["pair_id"]

    def get_text(page: dict, side_label: str) -> str:
        if pipeline == "b":
            # Pipeline B: use pre-extracted clean text
            return page.get("extracted_text", "")

        # Pipeline A: use raw HTML if available, else fall back to extracted text
        html_path = html_dir / page.get("html_path", "")
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            tokens = html_to_tokens_raw(html)
            return " ".join(tokens)
        else:
            log.debug(
                f"  Pipeline A: raw HTML not found for {pair_id}/{side_label}, "
                f"falling back to extracted_text"
            )
            return page.get("extracted_text", "")

    text_a = get_text(pair["page_a"], "page_a")
    text_b = get_text(pair["page_b"], "page_b")

    fp_a = fingerprint_text(text_a, aid, pipeline, "a", pair["page_a"].get("url",""))
    fp_b = fingerprint_text(text_b, aid, pipeline, "b", pair["page_b"].get("url",""))

    # --- Compute Hyperplane similarity ---
    hamming_dist = hamming_distance(fp_a.fingerprint, fp_b.fingerprint)
    cosine_sim   = cosine_similarity_estimate(hamming_dist, b=HP_B)
    hp_pred      = 1 if hamming_dist <= threshold else 0

    return HyperplanePairResult(
        pair_id=pair_id,
        article_id=aid,
        domain=domain,
        variant_type=vtype,
        label=label,
        hamming_distance=hamming_dist,
        cosine_similarity=round(float(cosine_sim), 6),
        hp_predicted=hp_pred,
    )

# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(results: list[HyperplanePairResult], predicted_field: str = "hp_predicted") -> dict:
    """
    Compute precision, recall, F1 for a list of HyperplanePairResult objects.
    """
    tp = fp = fn = tn = 0
    for r in results:
        pred  = getattr(r, predicted_field)
        label = r.label
        if pred == 1 and label == 1: tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 1: fn += 1
        else:                          tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": len(results),
    }


def compute_pr_curve_hyperplane(results: list[HyperplanePairResult], thresholds: list[int]) -> list[dict]:
    """
    Precision-Recall curve for Hyperplane by sweeping Hamming distance threshold.
    At each threshold t: predicted=1 iff hamming_distance <= t.
    """
    curve = []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for r in results:
            pred = 1 if r.hamming_distance <= t else 0
            if pred == 1 and r.label == 1: tp += 1
            elif pred == 1 and r.label == 0: fp += 1
            elif pred == 0 and r.label == 1: fn += 1
            else: tn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        curve.append({
            "threshold": t,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return curve


def breakdown_by_variant(results: list[HyperplanePairResult], predicted_field: str = "hp_predicted") -> dict:
    """
    Compute precision per variant_type.
    """
    by_variant: dict[str, list] = defaultdict(list)
    for r in results:
        by_variant[r.variant_type].append(r)

    breakdown = {}
    for vtype, vresults in sorted(by_variant.items()):
        m = compute_metrics(vresults, predicted_field)
        breakdown[vtype] = {
            "count":     m["total"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "label":     vresults[0].label,
        }
    return breakdown


def breakdown_by_domain(results: list[HyperplanePairResult], predicted_field: str = "hp_predicted") -> dict:
    """Compute precision per domain (BBC / Al Jazeera / Wikipedia)."""
    by_domain: dict[str, list] = defaultdict(list)
    for r in results:
        by_domain[r.domain].append(r)

    breakdown = {}
    for dom, dresults in sorted(by_domain.items()):
        m = compute_metrics(dresults, predicted_field)
        breakdown[dom] = {
            "count":     m["total"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
        }
    return breakdown

# ===========================================================================
# MAIN EVALUATION
# ===========================================================================

def run_evaluation(
    pairs_file: Path,
    html_dir:   Path,
    out_dir:    Path,
    pipeline:   str = "both",   # 'a', 'b', or 'both'
    verbose:    bool = False,
) -> None:

    # --- Load dataset ---
    log.info(f"Loading pairs from: {pairs_file}")
    pairs = []
    with pairs_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log.info(f"Loaded {len(pairs)} pairs")

    # --- Run pipelines ---
    pipelines_to_run = []
    if pipeline in ("a", "both"):
        pipelines_to_run.append("a")
    if pipeline in ("b", "both"):
        pipelines_to_run.append("b")

    all_results: dict[str, list[HyperplanePairResult]] = {}

    for pipe in pipelines_to_run:
        pipe_name = "A (Raw DOM)" if pipe == "a" else "B (Extracted Text)"
        log.info(f"\n{'='*60}")
        log.info(f"Running Pipeline {pipe_name} - Hyperplane LSH")
        log.info(f"{'='*60}")

        results = []
        for pair in tqdm(pairs, desc=f"Pipeline {pipe.upper()}", unit="pair"):
            try:
                r = evaluate_pair(pair, html_dir, pipeline=pipe, threshold=HP_DEFAULT_T)
                results.append(r)
                if verbose:
                    log.info(
                        f"  {r.pair_id}  "
                        f"label={r.label}  "
                        f"hamming={r.hamming_distance}  "
                        f"cosine={r.cosine_similarity:.4f}  "
                        f"hp_pred={r.hp_predicted}"
                    )
            except Exception as e:
                log.warning(f"  Error on pair {pair.get('pair_id','?')}: {e}")

        all_results[pipe] = results

        # --- Per-pipeline metrics ---
        log.info(f"\nPipeline {pipe.upper()} Results:")
        log.info(f"{'─'*60}")

        m = compute_metrics(results, "hp_predicted")
        log.info(
            f"  Hyperplane LSH (t={HP_DEFAULT_T})\n"
            f"    Precision={m['precision']:.4f}  "
            f"Recall={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}"
        )

        log.info(f"\nBreakdown by variant type (Pipeline {pipe.upper()}):")
        bd = breakdown_by_variant(results, "hp_predicted")
        for vtype, met in bd.items():
            log.info(
                f"  {vtype:<30} precision={met['precision']:.3f}  "
                f"recall={met['recall']:.3f}  n={met['count']}  label={met['label']}"
            )

        # --- Save per-pair results ---
        out_path = out_dir / f"pipeline_{pipe}_hyperplane_results.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                rec = asdict(r)
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        log.info(f"\n  Saved: {out_path.name}")

    # --- Cross-pipeline comparison ---
    if len(all_results) == 2:
        log.info(f"\n{'='*60}")
        log.info("PIPELINE COMPARISON - Hyperplane LSH")
        log.info(f"{'='*60}")

        log.info(f"  {'Variant type':<30} {'A:Prec':>8} {'B:Prec':>8} {'A:Rec':>8} {'B:Rec':>8}")
        log.info(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        bd_a = breakdown_by_variant(all_results["a"], "hp_predicted")
        bd_b = breakdown_by_variant(all_results["b"], "hp_predicted")

        all_vtypes = sorted(set(list(bd_a.keys()) + list(bd_b.keys())))
        for vt in all_vtypes:
            pa = bd_a.get(vt, {}).get("precision", 0)
            pb = bd_b.get(vt, {}).get("precision", 0)
            ra = bd_a.get(vt, {}).get("recall", 0)
            rb = bd_b.get(vt, {}).get("recall", 0)
            log.info(
                f"  {vt:<30} {pa:>8.3f} {pb:>8.3f} {ra:>8.3f} {rb:>8.3f}"
            )

    # --- P/R curves and summary JSON ---
    precision_recall = {}
    summary_table = {}

    for pipe, results in all_results.items():
        pipe_key = f"pipeline_{pipe}"

        precision_recall[pipe_key] = {
            "hyperplane": compute_pr_curve_hyperplane(results, HP_THRESHOLDS),
        }

        summary_table[pipe_key] = {
            "hyperplane": {
                "overall":   compute_metrics(results, "hp_predicted"),
                "by_variant": breakdown_by_variant(results, "hp_predicted"),
                "by_domain":  breakdown_by_domain(results, "hp_predicted"),
            },
        }

    (out_dir / "hyperplane_precision_recall.json").write_text(
        json.dumps(precision_recall, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    log.info(f"\nSaved: hyperplane_precision_recall.json")

    (out_dir / "hyperplane_summary_table.json").write_text(
        json.dumps(summary_table, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    log.info(f"Saved: hyperplane_summary_table.json")

    log.info(f"\n{'='*60}")
    log.info(f"All Hyperplane results saved to: {out_dir.resolve()}")
    log.info(f"{'='*60}")

# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Hyperplane LSH (Random Projection / SimHash) algorithm\n"
            "on the curated near-duplicate dataset.\n\n"
            "Runs two pipelines:\n"
            "  Pipeline A: raw HTML DOM tokenisation (control)\n"
            "  Pipeline B: trafilatura-extracted text (experimental)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="./dataset",
        help="Path to dataset directory produced by build_dataset.py (default: ./dataset)"
    )
    p.add_argument(
        "--out", default="./results/hyperplane",
        help="Output directory for Hyperplane results (default: ./results/hyperplane)"
    )
    p.add_argument(
        "--pipeline", default="both",
        choices=["a", "b", "both", "b-only"],
        help=(
            "Which pipeline to run: "
            "'a' (raw DOM), 'b' (extracted text), 'both' (default: both)"
        )
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-pair scores to stdout"
    )
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_file = dataset / "pairs_with_content.jsonl"
    html_dir   = dataset

    if not pairs_file.exists():
        raise FileNotFoundError(
            f"pairs_with_content.jsonl not found at {pairs_file}\n"
            f"Run build_dataset.py first to generate the dataset."
        )

    # Handle 'b-only' alias
    pipeline = args.pipeline
    if pipeline == "b-only":
        pipeline = "b"

    log.info(f"Dataset:  {dataset.resolve()}")
    log.info(f"Output:   {out_dir.resolve()}")
    log.info(f"Pipeline: {pipeline}")
    log.info(f"\nHyperplane LSH parameters (IMPROVED):")
    log.info(f"  b={HP_B} bits (fingerprint dimensionality - INCREASED)")
    log.info(f"  t={HP_DEFAULT_T} default threshold (max Hamming distance - ADJUSTED)")
    log.info(f"  seed={HP_SEED} (random projection seed)")
    log.info(f"  bigrams={HP_USE_BIGRAMS} (include bigrams for context)")
    log.info(f"  binary_tf=True (unique tokens counted once, not by frequency)")
    log.info(f"  threshold_range={HP_THRESHOLDS[0]}-{HP_THRESHOLDS[-1]} (PR curve sweep)")

    run_evaluation(
        pairs_file=pairs_file,
        html_dir=html_dir,
        out_dir=out_dir,
        pipeline=pipeline,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()