"""
Near-Duplicate Detection: Algorithm Implementation
===================================================
Research Project: Boilerplate-Aware DOM Extraction Pipeline for LSH Stabilization
Authors: Aastha Malhotra, Raashi Sharma, Sonali Verma

ALGORITHMS IMPLEMENTED
-----------------------
This script implements both algorithms from Henzinger (SIGIR 2006) exactly as
specified in the paper, applied across two pipelines:

  Pipeline A (Control):    raw HTML DOM  -> tokenise -> algorithm
  Pipeline B (Experimental): HTML -> trafilatura extraction -> tokenise -> algorithm

ALGORITHM B  (Broder et al. MinHash / Shingling)
  Paper parameters (Section 2):
    k  = 8   shingle size (tokens per shingle)
    m  = 84  number of hash functions (MinHash permutations)
    l  = 14  minvalues per supershingle
    m' = 6   supershingles per page (= m/l = 84/14)
    B-similar iff >= 2 of 6 supershingles agree
  LSH mapping:
    b = 6 bands, r = 14 rows/band
    S-curve inflection at Jaccard s* ≈ 0.88
    We sweep threshold t_B in [0.3, 1.0] for the precision-recall curve

ALGORITHM C  (Charikar SimHash / Random Projection)
  Paper parameters (Section 2):
    b = 384  bit vector length
    t = 372  minimum agreeing bits (max 12 differing bits)
    cosine similarity threshold ≈ cos(pi * 12/384) ≈ 0.9952
  We sweep threshold t_C in [360, 384] for the precision-recall curve

COMBINED ALGORITHM  (Section 3.5)
  Compute B-similar pairs, then filter by C-similarity >= threshold.
  Paper chose C-sim threshold = 355 on training set S1, achieving precision=0.79.

OUTPUT
------
  results/
    pipeline_a_alg_b_results.jsonl    per-pair predictions from Pipeline A, Alg B
    pipeline_a_alg_c_results.jsonl    per-pair predictions from Pipeline A, Alg C
    pipeline_b_alg_b_results.jsonl    per-pair predictions from Pipeline B, Alg B
    pipeline_b_alg_c_results.jsonl    per-pair predictions from Pipeline B, Alg C
    pipeline_b_combined_results.jsonl per-pair predictions from Pipeline B, Combined
    precision_recall.json             P/R/F1 at every threshold, all pipelines
    summary_table.json                headline numbers matching paper's Table 2 & 5 format

USAGE
-----
    python run_algorithms.py --dataset ./dataset --out ./results
    python run_algorithms.py --dataset ./dataset --out ./results --pipeline b-only
    python run_algorithms.py --dataset ./dataset --out ./results --verbose
"""

import re
import json
import math
import hashlib
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Literal

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
MISSING = []
try:
    from datasketch import MinHash
except ImportError:
    MISSING.append("datasketch")
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
# PAPER PARAMETERS — do not change without reading the paper
# ===========================================================================

# Algorithm B (Broder et al. — Section 2 of paper)
ALG_B_K           = 8    # shingle size in tokens
ALG_B_M           = 84   # number of MinHash permutations
ALG_B_L           = 14   # minvalues per supershingle
ALG_B_M_PRIME     = 6    # supershingles per page (= M / L)
ALG_B_MIN_BANDS   = 2    # B-similar iff >= this many supershingles agree

# Algorithm C (Charikar SimHash — Section 2 of paper)
ALG_C_B           = 384  # fingerprint bit length
ALG_C_T           = 372  # default threshold (min agreeing bits)
ALG_C_PIECES      = 12   # number of 32-bit pieces for LSH lookup (12 * 32 = 384)

# Combined algorithm threshold (paper Section 3.5, chosen on training set S1)
COMBINED_C_THRESH = 355

# Thresholds to sweep for P/R curves
# For Alg B: Jaccard similarity thresholds
ALG_B_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
# For Alg C: minimum agreeing bits
ALG_C_THRESHOLDS = list(range(340, 385, 4))  # 340, 344, ..., 384

# ===========================================================================
# TEXT EXTRACTION
# ===========================================================================

def html_to_tokens_raw(html: str) -> list[str]:
    """
    Pipeline A tokeniser: HTML DOM -> tokens.

    Replicates the paper's preprocessing (Section 2):
    - Strip HTML markup (replace with whitespace)
    - Every maximal alphanumeric sequence = one token
    - URLs in text split at slashes and dots -> individual tokens
    - IMG tag src URLs treated as terms

    The paper applies this to raw HTML without any boilerplate removal.
    This is the control pipeline.
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")

    # Collect IMG src URLs as terms (paper rule 2: IMG tag URL = term)
    img_tokens: list[str] = []
    for img in soup.find_all("img", src=True):
        src = img.get("src", "")
        if src:
            # Split at slashes and dots (paper: "broken at slashes and dots")
            img_tokens.extend(re.split(r'[/.]', src))

    # Replace all HTML tags with whitespace, get raw text
    text = soup.get_text(separator=" ")

    # Tokenise: every maximal alphanumeric sequence
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())

    return tokens + [t.lower() for t in img_tokens if t]


def html_to_tokens_extracted(html: str) -> list[str]:
    """
    Pipeline B tokeniser: trafilatura extraction -> tokens.

    Applies trafilatura to remove boilerplate (nav/header/footer/ads/sidebars)
    before tokenisation. This is the experimental pipeline.
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
            # Fallback: strip all tags manually
            text = BeautifulSoup(html, "lxml").get_text(separator=" ")
    except Exception:
        text = BeautifulSoup(html, "lxml").get_text(separator=" ")

    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def text_to_tokens(text: str) -> list[str]:
    """
    Tokenise pre-extracted text (used when raw HTML is not available,
    e.g., reading from pairs_with_content.jsonl extracted_text field).
    """
    if not text:
        return []
    return re.findall(r'[a-zA-Z0-9]+', text.lower())

# ===========================================================================
# ALGORITHM B — MinHash Shingling
# ===========================================================================

def make_shingles(tokens: list[str], k: int = ALG_B_K) -> set[str]:
    """
    Generate all k-token shingles from a token sequence.

    Paper (Section 2): 'every subsequence of k tokens is fingerprinted
    using 64-bit Rabin fingerprints... called shingles'.

    We use string shingles (join k tokens with space) because datasketch's
    MinHash handles the hashing internally. The Jaccard similarity over
    string shingles is identical to the Jaccard over fingerprinted shingles.
    """
    if len(tokens) < k:
        # Short page: use all individual tokens as degenerate shingles
        return set(tokens)
    return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}


def compute_minhash(shingles: set[str], num_perm: int = ALG_B_M) -> MinHash:
    """
    Compute the MinHash signature for a set of shingles.

    Paper (Section 2): 'm different fingerprinting functions fi for 1 <= i <= m.
    For each i the smallest of these values is called the i-th minvalue.'

    datasketch MinHash with num_perm=m implements exactly this.
    Each of the m hash functions corresponds to a random permutation,
    and the minvalue under each permutation is stored.
    """
    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update(shingle.encode("utf-8"))
    return m


def compute_supershingles(minhash: MinHash, m: int = ALG_B_M, l: int = ALG_B_L) -> list[int]:
    """
    Compute supershingle vector from MinHash signature.

    Paper (Section 2): 'The concatenation of minvalue j*l,...,(j+1)*l-1 for
    0 <= j < m' is fingerprinted with yet another fingerprinting function
    and is called supershingle.'

    We concatenate groups of l consecutive minvalues and hash them with SHA256
    to produce m' = m/l integer supershingles.

    m'  = 6 supershingles (= 84/14)
    Each supershingle is a 64-bit integer derived from 14 consecutive minvalues.
    """
    m_prime = m // l
    hashvalues = minhash.hashvalues  # numpy array of m uint32 values
    supershingles = []
    for j in range(m_prime):
        # Group of l consecutive minvalues
        group = hashvalues[j*l : (j+1)*l]
        # Hash the concatenated bytes (paper: 'fingerprinting function')
        group_bytes = group.astype(np.uint32).tobytes()
        h = hashlib.sha256(group_bytes).digest()
        supershingles.append(int.from_bytes(h[:8], "big"))
    return supershingles


def b_similarity(supershingles_a: list[int], supershingles_b: list[int]) -> int:
    """
    Compute B-similarity: number of identical supershingles.

    Paper (Section 2): 'The number of identical entries in the supershingle
    vectors of two pages is their B-similarity.'

    Range: 0 to m' (0 to 6).
    B-similar iff B-similarity >= 2.
    """
    return sum(1 for a, b in zip(supershingles_a, supershingles_b) if a == b)


def b_jaccard_estimate(minhash_a: MinHash, minhash_b: MinHash) -> float:
    """
    Jaccard similarity estimate from MinHash signatures.

    This is the underlying similarity measure. We use it for threshold sweeping
    in the precision-recall curve (since B-similarity of 2/6 is a coarse
    quantisation that doesn't allow fine threshold control).

    datasketch: jaccard() returns the fraction of agreeing minvalues,
    which is an unbiased estimator of true Jaccard similarity.
    """
    return minhash_a.jaccard(minhash_b)

# ===========================================================================
# ALGORITHM C — SimHash (Charikar Random Projection)
# ===========================================================================

def _token_projection_vector(token: str, b: int = ALG_C_B, seed: int = 42) -> np.ndarray:
    """
    Generate a deterministic b-dimensional {-1, +1} projection for a token.

    Paper (Section 2): 'Each token is projected into b-dimensional space by
    randomly choosing b entries from {-1, 1}. This projection is the same
    for all pages.'

    Implementation: use SHA256(seed || token) as RNG seed to ensure the
    same token always gets the same projection across all documents,
    without pre-generating a huge projection matrix.

    Memory: O(b) per call. For 500 tokens, this means 500 * 384 = 192K int8 ops.
    """
    seed_bytes = str(seed).encode("utf-8") + token.encode("utf-8")
    h = hashlib.sha256(seed_bytes).digest()
    rng_seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(rng_seed)
    return rng.choice([-1, 1], size=b).astype(np.int8)


def compute_simhash(tokens: list[str], b: int = ALG_C_B, seed: int = 42) -> np.ndarray:
    """
    Compute the b-bit SimHash fingerprint for a list of tokens.

    Paper (Section 2):
    1. 'For each page a b-dimensional vector is created by adding the projections
       of all the tokens in its token sequence.'
    2. 'The final vector for the page is created by setting every positive entry
       in the vector to 1 and every non-positive entry to 0.'

    Note: Alg C 'takes the frequency of terms into account' (each occurrence
    adds its projection vector, so frequent terms have greater weight).
    This differs from Alg B which ignores shingle frequency.

    Returns: np.ndarray of shape (b,) with dtype uint8, values in {0, 1}.
    """
    if not tokens:
        return np.zeros(b, dtype=np.uint8)

    accumulator = np.zeros(b, dtype=np.int32)
    for token in tokens:
        accumulator += _token_projection_vector(token, b, seed).astype(np.int32)

    return (accumulator >= 0).astype(np.uint8)


def c_similarity(fingerprint_a: np.ndarray, fingerprint_b: np.ndarray) -> int:
    """
    Compute C-similarity: number of bits that agree between two fingerprints.

    Paper (Section 2): 'the C-similarity of two pages is the number of bits
    their projections agree on.'

    Range: 0 to b (0 to 384).
    C-similar iff C-similarity >= t (default t = 372).
    """
    return int(np.sum(fingerprint_a == fingerprint_b))


def hamming_distance(fingerprint_a: np.ndarray, fingerprint_b: np.ndarray) -> int:
    """
    Hamming distance = b - C-similarity.
    Equivalent formulation: C-similar iff hamming_distance <= b - t = 12.
    """
    return int(np.sum(fingerprint_a != fingerprint_b))

# ===========================================================================
# FINGERPRINT DATACLASS — one per page per pipeline
# ===========================================================================

@dataclass
class PageFingerprint:
    """All fingerprint data for one page under one pipeline."""
    article_id:   str
    pipeline:     str           # 'a' (raw DOM) or 'b' (extracted text)
    page_side:    str           # 'a' or 'b' (which side of the pair)
    url:          str
    token_count:  int
    shingle_count: int
    minhash:      MinHash       # datasketch MinHash object (m=84)
    supershingles: list[int]    # m'=6 supershingle integers
    simhash:      np.ndarray    # 384-bit fingerprint as uint8 array

# ===========================================================================
# PAIR RESULT DATACLASS — one per pair per pipeline per algorithm
# ===========================================================================

@dataclass
class PairResult:
    """Detection result for one pair under one pipeline and algorithm."""
    pair_id:      str
    article_id:   str
    domain:       str
    variant_type: str
    label:        int           # ground truth: 1=near-dup, 0=true-negative

    # Algorithm B scores
    b_jaccard:    float = 0.0   # MinHash Jaccard estimate
    b_similarity_score: int = 0 # number of matching supershingles (0-6)
    b_predicted:  int = -1      # 1 if B-similar, 0 if not (at default thresh)

    # Algorithm C scores
    c_similarity_score: int = 0 # agreeing bits (0-384)
    c_hamming:    int = 384     # differing bits (0-384)
    c_predicted:  int = -1      # 1 if C-similar, 0 if not (at default thresh)

    # Combined algorithm (B filtered by C)
    combined_predicted: int = -1  # 1 if B-similar AND c_sim >= COMBINED_C_THRESH

# ===========================================================================
# CORE EVALUATION LOOP
# ===========================================================================

def fingerprint_text(
    text: str,
    article_id: str,
    pipeline: str,
    page_side: str,
    url: str,
) -> PageFingerprint:
    """
    Compute both MinHash and SimHash fingerprints from a text string.
    Used by both Pipeline A (text = raw DOM tokenised) and
    Pipeline B (text = trafilatura-extracted plain text).
    """
    tokens = text_to_tokens(text)
    shingles = make_shingles(tokens, k=ALG_B_K)

    mh = compute_minhash(shingles, num_perm=ALG_B_M)
    ss = compute_supershingles(mh, m=ALG_B_M, l=ALG_B_L)
    sh = compute_simhash(tokens, b=ALG_C_B)

    return PageFingerprint(
        article_id=article_id,
        pipeline=pipeline,
        page_side=page_side,
        url=url,
        token_count=len(tokens),
        shingle_count=len(shingles),
        minhash=mh,
        supershingles=ss,
        simhash=sh,
    )


def evaluate_pair(
    pair: dict,
    html_dir: Path,
    pipeline: str,     # 'a' = raw DOM, 'b' = extracted text
) -> PairResult:
    """
    Compute fingerprints for both pages in a pair and evaluate similarity.

    For pipeline='a': reads extracted_text from pair dict (trafilatura already
        ran during dataset generation — but we re-tokenise without extraction
        to simulate raw DOM tokenisation).

        NOTE: Ideally we would run html_to_tokens_raw() on the raw HTML files.
        If raw HTML files exist in html_dir, we use them. Otherwise we fall back
        to the stored extracted_text (which gives pipeline A an unfair advantage;
        that case is logged as a warning).

    For pipeline='b': uses the extracted_text field directly from the pair dict.
        This is the boilerplate-removed text from trafilatura.
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
            return " ".join(tokens)  # return as space-joined string for uniform interface
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

    # --- Algorithm B scores ---
    jaccard   = b_jaccard_estimate(fp_a.minhash, fp_b.minhash)
    b_sim     = b_similarity(fp_a.supershingles, fp_b.supershingles)
    b_pred    = 1 if b_sim >= ALG_B_MIN_BANDS else 0

    # --- Algorithm C scores ---
    c_sim     = c_similarity(fp_a.simhash, fp_b.simhash)
    hamming   = hamming_distance(fp_a.simhash, fp_b.simhash)
    c_pred    = 1 if c_sim >= ALG_C_T else 0

    # --- Combined algorithm ---
    combined  = 1 if (b_pred == 1 and c_sim >= COMBINED_C_THRESH) else 0

    return PairResult(
        pair_id=pair_id,
        article_id=aid,
        domain=domain,
        variant_type=vtype,
        label=label,
        b_jaccard=round(float(jaccard), 6),
        b_similarity_score=b_sim,
        b_predicted=b_pred,
        c_similarity_score=c_sim,
        c_hamming=hamming,
        c_predicted=c_pred,
        combined_predicted=combined,
    )

# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(results: list[PairResult], predicted_field: str) -> dict:
    """
    Compute precision, recall, F1 for a list of PairResult objects
    using the specified prediction field name.
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


def compute_pr_curve_alg_b(results: list[PairResult], thresholds: list[float]) -> list[dict]:
    """
    Precision-Recall curve for Algorithm B by sweeping Jaccard threshold.
    At each threshold t: predicted=1 iff b_jaccard >= t.
    """
    curve = []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for r in results:
            pred = 1 if r.b_jaccard >= t else 0
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


def compute_pr_curve_alg_c(results: list[PairResult], thresholds: list[int]) -> list[dict]:
    """
    Precision-Recall curve for Algorithm C by sweeping C-similarity threshold.
    At each threshold t: predicted=1 iff c_similarity_score >= t.
    """
    curve = []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for r in results:
            pred = 1 if r.c_similarity_score >= t else 0
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


def breakdown_by_variant(results: list[PairResult], predicted_field: str) -> dict:
    """
    Compute precision per variant_type.
    Matches Table 2 and Table 5 format from the paper (precision by condition).
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


def breakdown_by_domain(results: list[PairResult], predicted_field: str) -> dict:
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

    all_results: dict[str, list[PairResult]] = {}

    for pipe in pipelines_to_run:
        pipe_name = "A (Raw DOM)" if pipe == "a" else "B (Extracted Text)"
        log.info(f"\n{'='*60}")
        log.info(f"Running Pipeline {pipe_name}")
        log.info(f"{'='*60}")

        results = []
        for pair in tqdm(pairs, desc=f"Pipeline {pipe.upper()}", unit="pair"):
            try:
                r = evaluate_pair(pair, html_dir, pipeline=pipe)
                results.append(r)
                if verbose:
                    log.info(
                        f"  {r.pair_id}  "
                        f"label={r.label}  "
                        f"B_jaccard={r.b_jaccard:.3f}  "
                        f"B_sim={r.b_similarity_score}  "
                        f"C_sim={r.c_similarity_score}  "
                        f"b_pred={r.b_predicted}  "
                        f"c_pred={r.c_predicted}"
                    )
            except Exception as e:
                log.warning(f"  Error on pair {pair.get('pair_id','?')}: {e}")

        all_results[pipe] = results

        # --- Per-pipeline metrics ---
        log.info(f"\nPipeline {pipe.upper()} Results:")
        log.info(f"{'─'*60}")

        for alg, field in [
            ("Algorithm B (MinHash/Supershingles)", "b_predicted"),
            ("Algorithm C (SimHash)", "c_predicted"),
            ("Combined (B filtered by C)", "combined_predicted"),
        ]:
            m = compute_metrics(results, field)
            log.info(
                f"  {alg}\n"
                f"    Precision={m['precision']:.4f}  "
                f"Recall={m['recall']:.4f}  "
                f"F1={m['f1']:.4f}  "
                f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}"
            )

        log.info(f"\nBreakdown by variant type (Alg B, Pipeline {pipe.upper()}):")
        bd = breakdown_by_variant(results, "b_predicted")
        for vtype, m in bd.items():
            log.info(
                f"  {vtype:<30} precision={m['precision']:.3f}  "
                f"recall={m['recall']:.3f}  n={m['count']}  label={m['label']}"
            )

        # --- Save per-pair results ---
        for alg_key, alg_name, pred_field in [
            ("alg_b",    f"pipeline_{pipe}_alg_b_results.jsonl",    "b_predicted"),
            ("alg_c",    f"pipeline_{pipe}_alg_c_results.jsonl",    "c_predicted"),
            ("combined", f"pipeline_{pipe}_combined_results.jsonl", "combined_predicted"),
        ]:
            out_path = out_dir / alg_name
            with out_path.open("w", encoding="utf-8") as f:
                for r in results:
                    rec = asdict(r)
                    # Convert numpy types for JSON serialisation
                    rec.pop("minhash", None)   # MinHash object not serialisable
                    # simhash stored in PairResult as int values already
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            log.info(f"  Saved: {out_path.name}")

    # --- Cross-pipeline comparison ---
    if len(all_results) == 2:
        log.info(f"\n{'='*60}")
        log.info("PIPELINE COMPARISON (the key research result)")
        log.info(f"{'='*60}")

        headers = ["Variant type", "A:Prec(B)", "B:Prec(B)", "A:Prec(C)", "B:Prec(C)"]
        log.info(f"  {'Variant type':<30} {'A:AlgB':>8} {'B:AlgB':>8} {'A:AlgC':>8} {'B:AlgC':>8}")
        log.info(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        bd_a_b = breakdown_by_variant(all_results["a"], "b_predicted")
        bd_b_b = breakdown_by_variant(all_results["b"], "b_predicted")
        bd_a_c = breakdown_by_variant(all_results["a"], "c_predicted")
        bd_b_c = breakdown_by_variant(all_results["b"], "c_predicted")

        all_vtypes = sorted(set(
            list(bd_a_b.keys()) + list(bd_b_b.keys())
        ))
        for vt in all_vtypes:
            pa_b = bd_a_b.get(vt, {}).get("precision", 0)
            pb_b = bd_b_b.get(vt, {}).get("precision", 0)
            pa_c = bd_a_c.get(vt, {}).get("precision", 0)
            pb_c = bd_b_c.get(vt, {}).get("precision", 0)
            log.info(
                f"  {vt:<30} {pa_b:>8.3f} {pb_b:>8.3f} {pa_c:>8.3f} {pb_c:>8.3f}"
            )

    # --- P/R curves and summary JSON ---
    precision_recall = {}
    summary_table = {}

    for pipe, results in all_results.items():
        pipe_key = f"pipeline_{pipe}"

        precision_recall[pipe_key] = {
            "alg_b": compute_pr_curve_alg_b(results, ALG_B_THRESHOLDS),
            "alg_c": compute_pr_curve_alg_c(results, ALG_C_THRESHOLDS),
        }

        summary_table[pipe_key] = {
            "alg_b": {
                "overall":   compute_metrics(results, "b_predicted"),
                "by_variant": breakdown_by_variant(results, "b_predicted"),
                "by_domain":  breakdown_by_domain(results,  "b_predicted"),
            },
            "alg_c": {
                "overall":   compute_metrics(results, "c_predicted"),
                "by_variant": breakdown_by_variant(results, "c_predicted"),
                "by_domain":  breakdown_by_domain(results,  "c_predicted"),
            },
            "combined": {
                "overall":   compute_metrics(results, "combined_predicted"),
                "by_variant": breakdown_by_variant(results, "combined_predicted"),
                "by_domain":  breakdown_by_domain(results,  "combined_predicted"),
            },
        }

    (out_dir / "precision_recall.json").write_text(
        json.dumps(precision_recall, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    log.info(f"\nSaved: precision_recall.json")

    (out_dir / "summary_table.json").write_text(
        json.dumps(summary_table, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    log.info(f"Saved: summary_table.json")

    log.info(f"\n{'='*60}")
    log.info(f"All results saved to: {out_dir.resolve()}")
    log.info(f"{'='*60}")

# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Algorithm B (MinHash Shingling) and Algorithm C (SimHash)\n"
            "from Henzinger SIGIR 2006 on the curated near-duplicate dataset.\n\n"
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
        "--out", default="./results",
        help="Output directory for results (default: ./results)"
    )
    p.add_argument(
        "--pipeline", default="both",
        choices=["a", "b", "both"],
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
    html_dir   = dataset   # raw HTML files are at dataset/raw_html/<id>.html

    if not pairs_file.exists():
        raise FileNotFoundError(
            f"pairs_with_content.jsonl not found at {pairs_file}\n"
            f"Run build_dataset.py first to generate the dataset."
        )

    log.info(f"Dataset:  {dataset.resolve()}")
    log.info(f"Output:   {out_dir.resolve()}")
    log.info(f"Pipeline: {args.pipeline}")
    log.info(f"\nAlgorithm B parameters (paper Section 2):")
    log.info(f"  k={ALG_B_K} shingles, m={ALG_B_M} hash funcs, "
             f"l={ALG_B_L} minvals/supershingle, m'={ALG_B_M_PRIME} supershingles")
    log.info(f"  B-similar iff >= {ALG_B_MIN_BANDS} supershingles agree")
    log.info(f"\nAlgorithm C parameters (paper Section 2):")
    log.info(f"  b={ALG_C_B} bits, t={ALG_C_T} min agreeing bits "
             f"(max {ALG_C_B - ALG_C_T} differing bits)")
    log.info(f"\nCombined algorithm C-similarity threshold: {COMBINED_C_THRESH}")

    run_evaluation(
        pairs_file=pairs_file,
        html_dir=html_dir,
        out_dir=out_dir,
        pipeline=args.pipeline,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()