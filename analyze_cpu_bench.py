#!/usr/bin/env python3
"""
Correlate CPU benchmark results with structural equation features.

Joins cpu_bench_results.csv (timing) with equation_features.csv (pre-computed
structural features from the GPU analysis), then correlates each feature with
ns_per_call (lower = faster). Saves the merged table to cpu_equation_features.csv.

Usage:
  python analyze_cpu_bench.py
  python analyze_cpu_bench.py --results cpu_bench_results.csv --threshold 0.30 --top 20
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent.resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--results",    default=str(HERE / "cpu_bench_results.csv"))
parser.add_argument("--features",   default=str(HERE / "equation_features.csv"),
                    help="CSV with pre-computed structural features (from analyze_bench.py)")
parser.add_argument("--threshold",  type=float, default=0.30,
                    help="keep equations within this fraction above min loss (default 0.30)")
parser.add_argument("--top",        type=int,   default=20,
                    help="number of top correlating features to print (default 20)")
parser.add_argument("--output",     default=str(HERE / "cpu_equation_features.csv"),
                    help="output CSV (default: cpu_equation_features.csv)")
args = parser.parse_args()

RESULTS  = Path(args.results)
FEATURES = Path(args.features)

if not RESULTS.exists():
    raise SystemExit(f"No results file found: {RESULTS}\nRun eval_cpu_bench.py first.")

# ── load and filter ───────────────────────────────────────────────────────────

df_cpu = pd.read_csv(RESULTS)

# Drop baseline (no structural features, no loss)
df_cpu = df_cpu[df_cpu["equation"] != "baseline (2x exp)"].copy()

if df_cpu.empty:
    raise SystemExit("No non-baseline results in cpu_bench_results.csv.")

min_loss = df_cpu["loss"].min()
cutoff   = min_loss * (1 + args.threshold)
df_filt  = df_cpu[df_cpu["loss"] <= cutoff].copy()

print(f"Loaded {len(df_cpu)} CPU results.")
print(f"Min loss: {min_loss:.6f}  |  cutoff @ +{args.threshold*100:.0f}%: {cutoff:.6f}")
print(f"Equations within threshold: {len(df_filt)}")

if len(df_filt) < 3:
    print("Too few equations for meaningful correlation — showing all results instead.")
    df_filt = df_cpu.copy()

# ── merge with structural features ────────────────────────────────────────────

if FEATURES.exists():
    df_feat = pd.read_csv(FEATURES)
    # Structural feature columns (drop timing + equation metadata already in cpu results)
    drop_cols = {"equation", "loss", "throughput_gups", "us_per_seq"}
    feat_cols = [c for c in df_feat.columns if c not in drop_cols]
    df_merged = df_filt.merge(df_feat[["equation"] + feat_cols],
                              on="equation", how="left")
    print(f"Merged with {len(feat_cols)} structural features from {FEATURES.name}")
else:
    print(f"No features file at {FEATURES} — skipping structural feature merge.")
    df_merged = df_filt.copy()
    feat_cols = []

# ── summary table ─────────────────────────────────────────────────────────────

print(f"\n{'─'*70}")
print(f"{'Equation':<52} {'loss':>8} {'ns/call':>8} {'Mcps':>8}")
print(f"{'─'*70}")
for _, row in df_merged.sort_values("ns_per_call").iterrows():
    eq_short = str(row["equation"])[:50]
    print(f"{eq_short:<52} {row['loss']:>8.5f} {row['ns_per_call']:>8.1f} {row['mcps']:>8.1f}")

# Baseline row from results (if present)
df_base = df_cpu[df_cpu["equation"] == "baseline (2x exp)"]
if not df_base.empty:
    b = df_base.iloc[0]
    print(f"\n{'baseline (2x exp)':<52} {'':>8} {b['ns_per_call']:>8.1f} {b['mcps']:>8.1f}")

print(f"{'─'*70}\n")

# ── correlation analysis ──────────────────────────────────────────────────────

if feat_cols and len(df_merged) >= 3:
    numeric_feats = [c for c in feat_cols
                     if pd.api.types.is_numeric_dtype(df_merged[c])]

    corrs = {}
    for col in numeric_feats:
        series = df_merged[col].dropna()
        if series.nunique() < 2:
            continue
        aligned = df_merged.loc[series.index, "ns_per_call"]
        if aligned.nunique() < 2:
            continue
        r = np.corrcoef(series.values, aligned.values)[0, 1]
        if not np.isnan(r):
            corrs[col] = r

    if corrs:
        sorted_corrs = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top = sorted_corrs[:args.top]

        print(f"Top {len(top)} features correlated with ns_per_call  "
              f"(+r → slower on CPU, −r → faster):\n")
        print(f"  {'Feature':<35} {'r':>6}  Direction")
        print(f"  {'─'*55}")
        for feat, r in top:
            direction = "slower" if r > 0 else "faster"
            print(f"  {feat:<35} {r:>+6.3f}  More → {direction}")
        print()

        # Also show top speed-positive features (most negative r)
        fastest = sorted(corrs.items(), key=lambda kv: kv[1])[:5]
        print(f"Strongest speed-positive features (most negative r):")
        for feat, r in fastest:
            print(f"  {feat:<35} {r:>+6.3f}")
        print()
    else:
        print("Not enough variance in features for correlation analysis.")
else:
    print("Skipping correlation (need structural features + ≥3 equations).")

# ── save merged output ────────────────────────────────────────────────────────

out = Path(args.output)
df_merged.to_csv(out, index=False)
print(f"Saved merged results → {out.name}  ({len(df_merged)} rows, {len(df_merged.columns)} cols)")
