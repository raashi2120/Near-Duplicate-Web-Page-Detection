from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

BANDS = 7
ROWS_PER_BAND = 12
SIGNATURE_LENGTH = BANDS * ROWS_PER_BAND


def _normalize_signatures(
    signatures: Mapping[str, Sequence[int]] | Sequence[Sequence[int]],
) -> Dict[str, Tuple[int, ...]]:
    """
    Normalize signatures into a dict of `doc_id -> tuple(signature_values)`.

    Accepts:
    - dict-like input: {doc_id: signature}
    - list-like input: [signature0, signature1, ...] (doc IDs become "0", "1", ...)
    """
    if isinstance(signatures, Mapping):
        items: Iterable[Tuple[str, Sequence[int]]] = signatures.items()
    else:
        items = ((str(idx), sig) for idx, sig in enumerate(signatures))

    normalized: Dict[str, Tuple[int, ...]] = {}
    for doc_id, signature in items:
        signature_tuple = tuple(signature)
        if len(signature_tuple) != SIGNATURE_LENGTH:
            raise ValueError(
                f"Document '{doc_id}' has signature length {len(signature_tuple)}; "
                f"expected {SIGNATURE_LENGTH}."
            )
        normalized[str(doc_id)] = signature_tuple

    return normalized


def lsh_banding(signatures: Dict[str, List[int]]) -> Set[Tuple[str, str]]:
    """
    Generate candidate document pairs using standard MinHash LSH banding.

    Parameters
    ----------
    signatures:
        Mapping of `doc_id -> MinHash signature`, where each signature has length 84.

    Returns
    -------
    Set[Tuple[str, str]]
        Unique candidate pairs where each pair is sorted `(doc_id1, doc_id2)`.
    """
    normalized = _normalize_signatures(signatures)
    candidate_pairs: Set[Tuple[str, str]] = set()

    for band_idx in range(BANDS):
        start = band_idx * ROWS_PER_BAND
        end = start + ROWS_PER_BAND

        buckets: Dict[Tuple[int, ...], List[str]] = defaultdict(list)
        for doc_id, signature in normalized.items():
            band_signature = signature[start:end]
            buckets[band_signature].append(doc_id)

        for doc_ids in buckets.values():
            if len(doc_ids) < 2:
                continue
            for doc_a, doc_b in combinations(doc_ids, 2):
                candidate_pairs.add(tuple(sorted((doc_a, doc_b))))

    return candidate_pairs


def bucket_distribution(
    signatures: Mapping[str, Sequence[int]] | Sequence[Sequence[int]],
) -> Dict[int, Dict[int, int]]:
    """
    Optional helper for debugging bucket behavior per band.

    Returns a nested dictionary:
      {
        band_index: {
          bucket_size: number_of_buckets_with_that_size
        }
      }
    """
    normalized = _normalize_signatures(signatures)
    distribution: Dict[int, Dict[int, int]] = {}

    for band_idx in range(BANDS):
        start = band_idx * ROWS_PER_BAND
        end = start + ROWS_PER_BAND

        buckets: Dict[Tuple[int, ...], int] = defaultdict(int)
        for signature in normalized.values():
            buckets[signature[start:end]] += 1

        band_histogram: Dict[int, int] = defaultdict(int)
        for bucket_size in buckets.values():
            band_histogram[bucket_size] += 1
        distribution[band_idx] = dict(sorted(band_histogram.items()))

    return distribution


def _load_signatures_from_dataset(
    dataset_dir: Path,
    pairs_file: str = "pairs_with_content.jsonl",
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Build MinHash signatures for both pipelines from dataset pairs.
    """
    from neardup_detector import (
        ALG_B_M,
        compute_minhash,
        html_to_tokens_raw,
        make_shingles,
        text_to_tokens,
    )

    pairs_path = dataset_dir / pairs_file
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    signatures_a: Dict[str, List[int]] = {}
    signatures_b: Dict[str, List[int]] = {}

    with pairs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for side in ("page_a", "page_b"):
                page = record[side]

                html_rel = page.get("html_path")
                if html_rel and html_rel not in signatures_a:
                    html = (dataset_dir / html_rel).read_text(encoding="utf-8", errors="ignore")
                    tokens_a = html_to_tokens_raw(html)
                    shingles_a = make_shingles(tokens_a)
                    mh_a = compute_minhash(shingles_a, num_perm=ALG_B_M)
                    signatures_a[html_rel] = [int(x) for x in mh_a.hashvalues.tolist()]

                text_id = page.get("text_path")
                if not text_id:
                    text_id = f"{record.get('pair_id', 'unknown')}:{side}"
                if text_id not in signatures_b:
                    text = page.get("extracted_text", "") or ""
                    tokens_b = text_to_tokens(text)
                    shingles_b = make_shingles(tokens_b)
                    mh_b = compute_minhash(shingles_b, num_perm=ALG_B_M)
                    signatures_b[text_id] = [int(x) for x in mh_b.hashvalues.tolist()]

    return signatures_a, signatures_b


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinHash LSH banding.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--pairs-file", default="pairs_with_content.jsonl")
    parser.add_argument("--pipeline", choices=("a", "b", "both"), default="both")
    parser.add_argument("--show-samples", type=int, default=10)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--write-eval",
        action="store_true",
        help="Compute and save precision_recall and summary_table style metrics for LSH.",
    )
    return parser.parse_args()


def _save_candidate_pairs(path: Path, pairs: Set[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc_a, doc_b in sorted(pairs):
            handle.write(json.dumps({"doc_id_1": doc_a, "doc_id_2": doc_b}) + "\n")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float | int]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _evaluate_lsh_for_pipeline(
    dataset_dir: Path,
    pairs_file: str,
    pipeline: str,
    candidates: Set[Tuple[str, str]],
) -> Tuple[List[Dict[str, str | int]], Dict[str, object], Dict[str, object]]:
    """
    Evaluate LSH candidate-pair predictions against labelled dataset pairs.
    """
    eval_rows: List[Dict[str, str | int]] = []
    pairs_path = dataset_dir / pairs_file

    with pairs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            label = int(record.get("label", 0))
            domain = str(record.get("domain", "unknown"))
            variant = str(record.get("variant_type", "unknown"))

            page_a = record["page_a"]
            page_b = record["page_b"]
            if pipeline == "a":
                id_a = str(page_a.get("html_path", ""))
                id_b = str(page_b.get("html_path", ""))
            else:
                id_a = str(page_a.get("text_path", ""))
                id_b = str(page_b.get("text_path", ""))

            pred = int(tuple(sorted((id_a, id_b))) in candidates)
            eval_rows.append(
                {
                    "pair_id": str(record.get("pair_id", "")),
                    "domain": domain,
                    "variant_type": variant,
                    "label": label,
                    "predicted": pred,
                }
            )

    tp = sum(1 for r in eval_rows if r["label"] == 1 and r["predicted"] == 1)
    fp = sum(1 for r in eval_rows if r["label"] == 0 and r["predicted"] == 1)
    fn = sum(1 for r in eval_rows if r["label"] == 1 and r["predicted"] == 0)
    tn = sum(1 for r in eval_rows if r["label"] == 0 and r["predicted"] == 0)

    overall = _binary_metrics(tp, fp, fn, tn)
    overall["total"] = len(eval_rows)

    by_variant: Dict[str, object] = {}
    variant_groups: Dict[str, List[Dict[str, str | int]]] = defaultdict(list)
    for row in eval_rows:
        variant_groups[str(row["variant_type"])].append(row)
    for variant, rows in sorted(variant_groups.items()):
        v_tp = sum(1 for r in rows if r["label"] == 1 and r["predicted"] == 1)
        v_fp = sum(1 for r in rows if r["label"] == 0 and r["predicted"] == 1)
        v_fn = sum(1 for r in rows if r["label"] == 1 and r["predicted"] == 0)
        v_tn = sum(1 for r in rows if r["label"] == 0 and r["predicted"] == 0)
        m = _binary_metrics(v_tp, v_fp, v_fn, v_tn)
        by_variant[variant] = {
            "count": len(rows),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "label": int(rows[0]["label"]) if rows else 0,
        }

    by_domain: Dict[str, object] = {}
    domain_groups: Dict[str, List[Dict[str, str | int]]] = defaultdict(list)
    for row in eval_rows:
        domain_groups[str(row["domain"])].append(row)
    for domain, rows in sorted(domain_groups.items()):
        d_tp = sum(1 for r in rows if r["label"] == 1 and r["predicted"] == 1)
        d_fp = sum(1 for r in rows if r["label"] == 0 and r["predicted"] == 1)
        d_fn = sum(1 for r in rows if r["label"] == 1 and r["predicted"] == 0)
        d_tn = sum(1 for r in rows if r["label"] == 0 and r["predicted"] == 0)
        m = _binary_metrics(d_tp, d_fp, d_fn, d_tn)
        by_domain[domain] = {
            "count": len(rows),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        }

    pr_entry = dict(overall)
    pr_entry.pop("total", None)
    pr_entry["threshold"] = "candidate_pair"
    precision_recall = {"lsh": [pr_entry]}
    summary_table = {"lsh": {"overall": overall, "by_variant": by_variant, "by_domain": by_domain}}
    return eval_rows, precision_recall, summary_table


if __name__ == "__main__":
    args = _parse_args()
    results_dir = Path(args.results_dir)
    signatures_a, signatures_b = _load_signatures_from_dataset(
        dataset_dir=Path(args.dataset_dir),
        pairs_file=args.pairs_file,
    )

    summary: Dict[str, Dict[str, int | str]] = {}

    if args.pipeline in ("a", "both"):
        candidate_a = lsh_banding(signatures_a)
        output_a = results_dir / "pipeline_a_lsh_candidates.jsonl"
        _save_candidate_pairs(output_a, candidate_a)
        summary["pipeline_a"] = {
            "docs": len(signatures_a),
            "candidate_pairs": len(candidate_a),
            "output_file": str(output_a),
        }
        print(f"Pipeline A docs: {len(signatures_a)}")
        print(f"Pipeline A candidate pairs: {len(candidate_a)}")
        print(f"Pipeline A output file: {output_a}")
        print(f"Pipeline A sample pairs: {sorted(candidate_a)[:args.show_samples]}")

    if args.pipeline in ("b", "both"):
        candidate_b = lsh_banding(signatures_b)
        output_b = results_dir / "pipeline_b_lsh_candidates.jsonl"
        _save_candidate_pairs(output_b, candidate_b)
        summary["pipeline_b"] = {
            "docs": len(signatures_b),
            "candidate_pairs": len(candidate_b),
            "output_file": str(output_b),
        }
        print(f"Pipeline B docs: {len(signatures_b)}")
        print(f"Pipeline B candidate pairs: {len(candidate_b)}")
        print(f"Pipeline B output file: {output_b}")
        print(f"Pipeline B sample pairs: {sorted(candidate_b)[:args.show_samples]}")

    summary_path = results_dir / "lsh_banding_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Summary file: {summary_path}")

    if args.write_eval:
        eval_precision_recall: Dict[str, object] = {}
        eval_summary_table: Dict[str, object] = {}

        if args.pipeline in ("a", "both"):
            eval_rows_a, pr_a, st_a = _evaluate_lsh_for_pipeline(
                dataset_dir=Path(args.dataset_dir),
                pairs_file=args.pairs_file,
                pipeline="a",
                candidates=candidate_a,
            )
            eval_precision_recall["pipeline_a"] = pr_a
            eval_summary_table["pipeline_a"] = st_a
            eval_rows_a_path = results_dir / "pipeline_a_lsh_results.jsonl"
            eval_rows_a_path.parent.mkdir(parents=True, exist_ok=True)
            with eval_rows_a_path.open("w", encoding="utf-8") as handle:
                for row in eval_rows_a:
                    handle.write(json.dumps(row) + "\n")
            print(f"Pipeline A eval rows: {eval_rows_a_path}")

        if args.pipeline in ("b", "both"):
            eval_rows_b, pr_b, st_b = _evaluate_lsh_for_pipeline(
                dataset_dir=Path(args.dataset_dir),
                pairs_file=args.pairs_file,
                pipeline="b",
                candidates=candidate_b,
            )
            eval_precision_recall["pipeline_b"] = pr_b
            eval_summary_table["pipeline_b"] = st_b
            eval_rows_b_path = results_dir / "pipeline_b_lsh_results.jsonl"
            eval_rows_b_path.parent.mkdir(parents=True, exist_ok=True)
            with eval_rows_b_path.open("w", encoding="utf-8") as handle:
                for row in eval_rows_b:
                    handle.write(json.dumps(row) + "\n")
            print(f"Pipeline B eval rows: {eval_rows_b_path}")

        lsh_pr_path = results_dir / "lsh_precision_recall.json"
        with lsh_pr_path.open("w", encoding="utf-8") as handle:
            json.dump(eval_precision_recall, handle, indent=2)
        print(f"LSH precision/recall file: {lsh_pr_path}")

        lsh_summary_path = results_dir / "lsh_summary_table.json"
        with lsh_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(eval_summary_table, handle, indent=2)
        print(f"LSH summary table file: {lsh_summary_path}")



