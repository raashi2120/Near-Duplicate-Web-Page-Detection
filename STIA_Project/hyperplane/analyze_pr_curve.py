#!/usr/bin/env python3
"""Analyze PR curve to find FP and optimal thresholds"""

import json
from pathlib import Path

# Load PR curve
pr_file = Path("results/hyperplane/hyperplane_precision_recall.json")
with pr_file.open() as f:
    pr_data = json.load(f)

print("\nPipeline A - Full Precision-Recall Curve:")
print("="*85)
print(f"{'Thresh':>7} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
print("-"*85)

curve_a = pr_data.get("pipeline_a", {}).get("hyperplane", [])
first_fp_found = False
for i, point in enumerate(curve_a):
    fp = point.get("fp", 0)
    if not first_fp_found or fp > 0:
        print(f"{point['threshold']:>7} "
              f"{point['precision']:>8.4f} "
              f"{point['recall']:>8.4f} "
              f"{point['f1']:>8.4f} "
              f"{point['tp']:>5d} "
              f"{point['fp']:>5d} "
              f"{point['fn']:>5d} "
              f"{point['tn']:>5d}")
              
        if fp > 0 and not first_fp_found:
            print(f"  ^^^ FIRST FALSE POSITIVE APPEARS AT THRESHOLD {point['threshold']} ^^^")
            first_fp_found = True

# Summary
print("\n" + "="*85)
print("SUMMARY:")
print(f"Total thresholds tested: {len(curve_a)}")

best_f1_point = max(curve_a, key=lambda x: x['f1'])
best_balanced_point = max(curve_a, key=lambda x: (x['precision'] * x['recall']))
first_fp_point = next((p for p in curve_a if p['fp'] > 0), None)

print(f"\nF1-maximizing threshold: {best_f1_point['threshold']}")
print(f"  Precision={best_f1_point['precision']:.4f}, Recall={best_f1_point['recall']:.4f}, "
      f"F1={best_f1_point['f1']:.4f}")
print(f"  TP={best_f1_point['tp']}, FP={best_f1_point['fp']}, "
      f"FN={best_f1_point['fn']}, TN={best_f1_point['tn']}")

print(f"\nBest precision-recall balance: {best_balanced_point['threshold']}")
print(f"  Precision={best_balanced_point['precision']:.4f}, Recall={best_balanced_point['recall']:.4f}, "
      f"F1={best_balanced_point['f1']:.4f}")
print(f"  TP={best_balanced_point['tp']}, FP={best_balanced_point['fp']}, "
      f"FN={best_balanced_point['fn']}, TN={best_balanced_point['tn']}")

if first_fp_point:
    print(f"\nFirst appearance of FP: threshold={first_fp_point['threshold']}")
    print(f"  Precision={first_fp_point['precision']:.4f}, Recall={first_fp_point['recall']:.4f}, "
          f"F1={first_fp_point['f1']:.4f}")
    print(f"  TP={first_fp_point['tp']}, FP={first_fp_point['fp']}, "
          f"FN={first_fp_point['fn']}, TN={first_fp_point['tn']}")
else:
    print(f"\n⚠️  WARNING: NO FALSE POSITIVES at ANY threshold!")
    print(f"    The negative class (full_content_swap) is perfectly separated")
    print(f"    from the positive classes (near-duplicates).")
    print(f"\n    IMPLICATIONS:")
    print(f"    ✓ Algorithm correctly identifies true negatives with zero false positives")
    print(f"    ✗ Precision is always 1.0, but recall may be limited by overlap in")
    print(f"      the positive class itself")
