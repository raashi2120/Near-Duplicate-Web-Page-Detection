#!/usr/bin/env python3
"""
Hyperplane Results Analyzer
============================

Quick analysis script to:
1. Display summary statistics from Hyperplane results
2. Compare Pipeline A vs Pipeline B
3. Find optimal threshold from PR curve
4. Generate comparison table

Usage:
    python analyze_hyperplane.py --results ./results/hyperplane
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with filepath.open(encoding="utf-8") as f:
        return json.load(f)


def display_summary(summary: dict) -> None:
    """Display overall metrics in a nice format."""
    print("\n" + "="*80)
    print("HYPERPLANE LSH - OVERALL RESULTS")
    print("="*80)
    
    for pipeline_name, pipeline_data in summary.items():
        print(f"\n{pipeline_name.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        hp_data = pipeline_data.get("hyperplane", {})
        overall = hp_data.get("overall", {})
        
        print(f"  Precision: {overall.get('precision', 0):.4f}")
        print(f"  Recall:    {overall.get('recall', 0):.4f}")
        print(f"  F1:        {overall.get('f1', 0):.4f}")
        print(f"  TP: {overall.get('tp', 0):4d}  FP: {overall.get('fp', 0):4d}  "
              f"FN: {overall.get('fn', 0):4d}  TN: {overall.get('tn', 0):4d}")
        print(f"  Total pairs: {overall.get('total', 0)}")


def display_variant_breakdown(summary: dict) -> None:
    """Display breakdown by variant type."""
    print("\n" + "="*80)
    print("BREAKDOWN BY VARIANT TYPE")
    print("="*80)
    
    for pipeline_name, pipeline_data in summary.items():
        print(f"\n{pipeline_name.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        hp_data = pipeline_data.get("hyperplane", {})
        by_variant = hp_data.get("by_variant", {})
        
        if not by_variant:
            print("  No variant breakdown available")
            continue
        
        # Print header
        print(f"  {'Variant Type':<35} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Count':>6} {'Label':>6}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
        
        # Print each variant
        for variant_type, metrics in sorted(by_variant.items()):
            print(f"  {variant_type:<35} "
                  f"{metrics.get('precision', 0):>8.4f} "
                  f"{metrics.get('recall', 0):>8.4f} "
                  f"{metrics.get('f1', 0):>8.4f} "
                  f"{metrics.get('count', 0):>6d} "
                  f"{metrics.get('label', -1):>6d}")


def display_domain_breakdown(summary: dict) -> None:
    """Display breakdown by domain."""
    print("\n" + "="*80)
    print("BREAKDOWN BY DOMAIN")
    print("="*80)
    
    for pipeline_name, pipeline_data in summary.items():
        print(f"\n{pipeline_name.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        hp_data = pipeline_data.get("hyperplane", {})
        by_domain = hp_data.get("by_domain", {})
        
        if not by_domain:
            print("  No domain breakdown available")
            continue
        
        # Print header
        print(f"  {'Domain':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Count':>6}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        
        # Print each domain
        for domain, metrics in sorted(by_domain.items()):
            print(f"  {domain:<20} "
                  f"{metrics.get('precision', 0):>8.4f} "
                  f"{metrics.get('recall', 0):>8.4f} "
                  f"{metrics.get('f1', 0):>8.4f} "
                  f"{metrics.get('count', 0):>6d}")


def find_optimal_threshold(pr_curve: List[dict], metric: str = "f1") -> Tuple[int, dict]:
    """Find threshold that maximizes the given metric."""
    best_threshold = 0
    best_point = {}
    best_value = -1
    
    for point in pr_curve:
        value = point.get(metric, 0)
        if value > best_value:
            best_value = value
            best_threshold = point.get("threshold", 0)
            best_point = point
    
    return best_threshold, best_point


def display_pr_analysis(pr_data: dict) -> None:
    """Display precision-recall curve analysis."""
    print("\n" + "="*80)
    print("PRECISION-RECALL CURVE ANALYSIS")
    print("="*80)
    
    for pipeline_name, pipeline_curves in pr_data.items():
        print(f"\n{pipeline_name.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        hp_curve = pipeline_curves.get("hyperplane", [])
        
        if not hp_curve:
            print("  No PR curve data available")
            continue
        
        # Find optimal thresholds for different metrics
        opt_f1_thresh, opt_f1_point = find_optimal_threshold(hp_curve, "f1")
        opt_prec_thresh, opt_prec_point = find_optimal_threshold(hp_curve, "precision")
        opt_rec_thresh, opt_rec_point = find_optimal_threshold(hp_curve, "recall")
        
        print(f"\n  Optimal Threshold (F1-maximizing):")
        print(f"    Threshold:  {opt_f1_thresh}")
        print(f"    Precision:  {opt_f1_point.get('precision', 0):.4f}")
        print(f"    Recall:     {opt_f1_point.get('recall', 0):.4f}")
        print(f"    F1:         {opt_f1_point.get('f1', 0):.4f}")
        
        print(f"\n  Optimal Threshold (Precision-maximizing):")
        print(f"    Threshold:  {opt_prec_thresh}")
        print(f"    Precision:  {opt_prec_point.get('precision', 0):.4f}")
        print(f"    Recall:     {opt_prec_point.get('recall', 0):.4f}")
        
        print(f"\n  Optimal Threshold (Recall-maximizing):")
        print(f"    Threshold:  {opt_rec_thresh}")
        print(f"    Precision:  {opt_rec_point.get('precision', 0):.4f}")
        print(f"    Recall:     {opt_rec_point.get('recall', 0):.4f}")
        
        # Sample points from curve
        print(f"\n  Sample PR Curve Points:")
        print(f"    {'Thresh':>7} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
        print(f"    {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
        
        # Show every 5th point
        for i, point in enumerate(hp_curve):
            if i % 5 == 0 or i == len(hp_curve) - 1:
                print(f"    {point.get('threshold', 0):>7} "
                      f"{point.get('precision', 0):>8.4f} "
                      f"{point.get('recall', 0):>8.4f} "
                      f"{point.get('f1', 0):>8.4f} "
                      f"{point.get('tp', 0):>5d} "
                      f"{point.get('fp', 0):>5d} "
                      f"{point.get('fn', 0):>5d} "
                      f"{point.get('tn', 0):>5d}")


def compare_pipelines(summary: dict) -> None:
    """Compare Pipeline A vs Pipeline B side-by-side."""
    print("\n" + "="*80)
    print("PIPELINE COMPARISON (A vs B)")
    print("="*80)
    
    if "pipeline_a" not in summary or "pipeline_b" not in summary:
        print("  Both pipelines not available for comparison")
        return
    
    # Overall comparison
    print("\nOVERALL METRICS:")
    print("-" * 80)
    print(f"  {'Metric':<15} {'Pipeline A':>12} {'Pipeline B':>12} {'Difference':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    
    a_overall = summary["pipeline_a"]["hyperplane"]["overall"]
    b_overall = summary["pipeline_b"]["hyperplane"]["overall"]
    
    for metric in ["precision", "recall", "f1"]:
        a_val = a_overall.get(metric, 0)
        b_val = b_overall.get(metric, 0)
        diff = b_val - a_val
        print(f"  {metric.capitalize():<15} "
              f"{a_val:>12.4f} "
              f"{b_val:>12.4f} "
              f"{diff:>+12.4f}")
    
    # Variant comparison
    print("\nBY VARIANT TYPE:")
    print("-" * 80)
    print(f"  {'Variant Type':<35} {'A:Prec':>8} {'B:Prec':>8} {'Δ':>8} {'A:Rec':>8} {'B:Rec':>8} {'Δ':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    a_variants = summary["pipeline_a"]["hyperplane"]["by_variant"]
    b_variants = summary["pipeline_b"]["hyperplane"]["by_variant"]
    
    all_variants = sorted(set(list(a_variants.keys()) + list(b_variants.keys())))
    
    for variant in all_variants:
        a_metrics = a_variants.get(variant, {})
        b_metrics = b_variants.get(variant, {})
        
        a_prec = a_metrics.get("precision", 0)
        b_prec = b_metrics.get("precision", 0)
        diff_prec = b_prec - a_prec
        
        a_rec = a_metrics.get("recall", 0)
        b_rec = b_metrics.get("recall", 0)
        diff_rec = b_rec - a_rec
        
        print(f"  {variant:<35} "
              f"{a_prec:>8.4f} "
              f"{b_prec:>8.4f} "
              f"{diff_prec:>+8.4f} "
              f"{a_rec:>8.4f} "
              f"{b_rec:>8.4f} "
              f"{diff_rec:>+8.4f}")
    
    # Highlight key insight
    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("-" * 80)
    
    boilerplate_a = a_variants.get("boilerplate_only_diff", {})
    boilerplate_b = b_variants.get("boilerplate_only_diff", {})
    
    if boilerplate_a and boilerplate_b:
        a_prec = boilerplate_a.get("precision", 0)
        b_prec = boilerplate_b.get("precision", 0)
        improvement = ((b_prec - a_prec) / a_prec * 100) if a_prec > 0 else 0
        
        print(f"  On 'boilerplate_only_diff' variant (same content, different layout):")
        print(f"    Pipeline A (raw DOM):      Precision = {a_prec:.4f}")
        print(f"    Pipeline B (extracted):    Precision = {b_prec:.4f}")
        print(f"    Improvement:               {improvement:+.1f}%")
        print(f"\n  This demonstrates that text extraction (Pipeline B) is more robust")
        print(f"  to boilerplate changes than raw DOM processing (Pipeline A).")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Hyperplane LSH results"
    )
    parser.add_argument(
        "--results",
        default="./results/hyperplane",
        help="Path to Hyperplane results directory"
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["summary", "variants", "domains", "pr", "compare", "all"],
        default=["all"],
        help="Which sections to display (default: all)"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results)
    
    # Load files
    summary_file = results_dir / "hyperplane_summary_table.json"
    pr_file = results_dir / "hyperplane_precision_recall.json"
    
    if not summary_file.exists():
        print(f"Error: {summary_file} not found")
        print(f"Run hyperplane_detector.py first to generate results")
        return
    
    summary = load_json(summary_file)
    pr_data = load_json(pr_file) if pr_file.exists() else {}
    
    sections = args.sections
    if "all" in sections:
        sections = ["summary", "variants", "domains", "pr", "compare"]
    
    # Display requested sections
    if "summary" in sections:
        display_summary(summary)
    
    if "variants" in sections:
        display_variant_breakdown(summary)
    
    if "domains" in sections:
        display_domain_breakdown(summary)
    
    if "pr" in sections and pr_data:
        display_pr_analysis(pr_data)
    
    if "compare" in sections:
        compare_pipelines(summary)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()