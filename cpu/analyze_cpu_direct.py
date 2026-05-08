#!/usr/bin/env python3
"""
Correlate CPU throughput results with AST structural features.
Reads cpu_direct_results.csv, parses each equation, extracts tree features,
and correlates them with f32_gcps (higher = faster).

Usage:
  python analyze_cpu_direct.py
  python analyze_cpu_direct.py --threshold 0.30 --top 20
"""

import argparse, re, math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

HERE      = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent

parser = argparse.ArgumentParser()
parser.add_argument("--results",   default=str(HERE / "cpu_direct_results.csv"))
parser.add_argument("--threshold", type=float, default=0.30,
                    help="keep equations within this fraction above min loss")
parser.add_argument("--top",       type=int, default=30)
parser.add_argument("--output",    default=str(HERE / "cpu_direct_features.csv"))
args = parser.parse_args()

RESULTS = Path(args.results)
if not RESULTS.exists():
    raise SystemExit(f"No results at {RESULTS}\nRun eval_cpu_direct.py first.")


# ── AST parser (copied from gpu/analyze_bench.py) ────────────────────────────

@dataclass
class Node:
    kind: str
    value: object = None
    children: list = field(default_factory=list)

    @property
    def left(self):  return self.children[0] if len(self.children) > 0 else None
    @property
    def right(self): return self.children[1] if len(self.children) > 1 else None


@dataclass
class Tok:
    type: str
    value: str

def tokenize(s: str) -> list[Tok]:
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in ' \t': i += 1
        elif c == '(': tokens.append(Tok('LP', '(')); i += 1
        elif c == ')': tokens.append(Tok('RP', ')')); i += 1
        elif c == ',': tokens.append(Tok('COMMA', ',')); i += 1
        elif c == '+': tokens.append(Tok('PLUS', '+')); i += 1
        elif c == '*': tokens.append(Tok('STAR', '*')); i += 1
        elif c == '-':
            prev = tokens[-1].type if tokens else None
            if prev in (None, 'LP', 'PLUS', 'MINUS', 'STAR', 'SLASH', 'COMMA'):
                j = i + 1
                while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                    j += 1
                if j < len(s) and s[j] in 'eE':
                    j += 1
                    if j < len(s) and s[j] in '+-': j += 1
                    while j < len(s) and s[j].isdigit(): j += 1
                tokens.append(Tok('NUM', s[i:j])); i = j
            else:
                tokens.append(Tok('MINUS', '-')); i += 1
        elif c.isdigit() or c == '.':
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'): j += 1
            if j < len(s) and s[j] in 'eE':
                j += 1
                if j < len(s) and s[j] in '+-': j += 1
                while j < len(s) and s[j].isdigit(): j += 1
            tokens.append(Tok('NUM', s[i:j])); i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'): j += 1
            tokens.append(Tok('ID', s[i:j])); i = j
        else:
            i += 1
    return tokens


BINOPS = {'+': 'add', '-': 'sub', '*': 'mul'}
VARS   = {'m', 's', 'x'}
FUNCS  = {'abs', 'exp', 'log', 'sqrt', 'greater', 'neg'}

class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos  = 0

    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else Tok('EOF', '')

    def eat(self, ttype=None):
        t = self.toks[self.pos]
        if ttype and t.type != ttype:
            raise ValueError(f"Expected {ttype}, got {t}")
        self.pos += 1
        return t

    def parse_expr(self):
        left = self.parse_term()
        while self.peek().type in ('PLUS', 'MINUS'):
            op   = self.eat()
            op_s = 'add' if op.type == 'PLUS' else 'sub'
            right = self.parse_term()
            left  = Node('binop', op_s, [left, right])
        return left

    def parse_term(self):
        left = self.parse_unary()
        while self.peek().type == 'STAR':
            self.eat()
            right = self.parse_unary()
            left  = Node('binop', 'mul', [left, right])
        return left

    def parse_unary(self):
        if self.peek().type == 'MINUS':
            self.eat()
            child = self.parse_atom()
            return Node('func', 'neg', [child])
        return self.parse_atom()

    def parse_atom(self):
        t = self.peek()
        if t.type == 'NUM':
            self.eat()
            return Node('num', float(t.value))
        if t.type == 'LP':
            self.eat('LP')
            e = self.parse_expr()
            self.eat('RP')
            return e
        if t.type == 'ID':
            name = self.eat('ID').value
            if self.peek().type == 'LP':
                self.eat('LP')
                args = [self.parse_expr()]
                while self.peek().type == 'COMMA':
                    self.eat('COMMA')
                    args.append(self.parse_expr())
                self.eat('RP')
                return Node('func', name, args)
            if name in VARS:
                return Node('var', name)
            return Node('num', float('nan'))
        raise ValueError(f"Unexpected token: {t}")


def parse_equation(s: str) -> Node | None:
    try:
        toks   = tokenize(s)
        parser = Parser(toks)
        tree   = parser.parse_expr()
        return tree
    except Exception:
        return None


# ── feature extraction ────────────────────────────────────────────────────────

def count_nodes(n: Node) -> int:
    if n is None: return 0
    return 1 + sum(count_nodes(c) for c in n.children)

def count_leaves(n: Node) -> int:
    if n is None: return 0
    if not n.children: return 1
    return sum(count_leaves(c) for c in n.children)

def tree_height(n: Node) -> int:
    if n is None or not n.children: return 0
    return 1 + max(tree_height(c) for c in n.children)

def count_kind(n: Node, kind: str, val=None) -> int:
    if n is None: return 0
    match = (n.kind == kind) and (val is None or n.value == val)
    return int(match) + sum(count_kind(c, kind, val) for c in n.children)

def var_depths(n: Node, depth=0) -> list[int]:
    if n is None: return []
    if n.kind == 'var': return [depth]
    out = []
    for c in n.children:
        out.extend(var_depths(c, depth + 1))
    return out

def var_depths_for(n: Node, var: str, depth=0) -> list[int]:
    if n is None: return []
    if n.kind == 'var' and n.value == var: return [depth]
    out = []
    for c in n.children:
        out.extend(var_depths_for(c, var, depth + 1))
    return out

def const_leaf_fraction(n: Node) -> float:
    if n is None: return 0.0
    leaves = count_leaves(n)
    consts = count_kind(n, 'num')
    return consts / leaves if leaves else 0.0

def distinct_funcs(n: Node) -> set:
    if n is None: return set()
    s = set()
    if n.kind in ('func', 'binop'): s.add(n.value)
    for c in n.children: s |= distinct_funcs(c)
    return s

def x_products(n: Node) -> int:
    """Count nodes where * has x in both subtrees."""
    if n is None: return 0
    c = sum(x_products(ch) for ch in n.children)
    if n.kind == 'binop' and n.value == 'mul':
        lx = count_kind(n.left, 'var', 'x') > 0
        rx = count_kind(n.right, 'var', 'x') > 0
        if lx and rx: c += 1
    return c

def extract_features(eq: str) -> dict:
    tree = parse_equation(eq)
    if tree is None:
        return {}

    n_nodes   = count_nodes(tree)
    n_leaves  = count_leaves(tree)
    n_internal = n_nodes - n_leaves
    height    = tree_height(tree)

    n_add     = count_kind(tree, 'binop', 'add')
    n_sub     = count_kind(tree, 'binop', 'sub')
    n_mul     = count_kind(tree, 'binop', 'mul')
    n_abs     = count_kind(tree, 'func', 'abs')
    n_exp     = count_kind(tree, 'func', 'exp')
    n_greater = count_kind(tree, 'func', 'greater')
    n_consts  = count_kind(tree, 'num')

    frac_const = const_leaf_fraction(tree)
    n_ops      = len(distinct_funcs(tree))

    all_d  = var_depths(tree)
    x_d    = var_depths_for(tree, 'x')
    m_d    = var_depths_for(tree, 'm')
    s_d    = var_depths_for(tree, 's')

    x_avg_depth = float(np.mean(x_d)) if x_d else 0.0
    x_poly = x_products(tree)

    all_depth_std = float(np.std(all_d)) if len(all_d) > 1 else 0.0
    leaf_ratio    = n_leaves / n_nodes if n_nodes else 0

    return dict(
        n_nodes=n_nodes, n_leaves=n_leaves, n_internal=n_internal,
        tree_height=height, leaf_ratio=leaf_ratio,
        n_add=n_add, n_sub=n_sub, n_mul=n_mul,
        n_abs=n_abs, n_exp=n_exp, n_greater=n_greater,
        n_constants=n_consts, frac_const=frac_const,
        n_distinct_ops=n_ops, x_avg_depth=x_avg_depth,
        x_poly_degree=x_poly, all_var_depth_std=all_depth_std,
        n_x=len(x_d), n_m=len(m_d), n_s=len(s_d),
        total_arith_ops=n_add + n_sub + n_mul,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(RESULTS)
    df = df[df["equation"].notna()].copy()

    # Filter baselines
    df = df[~df["equation"].str.startswith("baseline")].copy()

    if df.empty:
        print("No equations found.")
        return

    min_loss = df["loss"].dropna().min()
    cutoff   = min_loss * (1 + args.threshold)
    df_filt  = df[df["loss"] <= cutoff].copy()

    print(f"Total equations: {len(df)}")
    print(f"Min loss: {min_loss:.6f}  cutoff @ +{args.threshold*100:.0f}%: {cutoff:.6f}")
    print(f"Equations within threshold: {len(df_filt)}")

    if len(df_filt) < 3:
        print("Too few — using all equations.")
        df_filt = df.copy()

    # Extract features
    feat_rows = []
    for _, row in df_filt.iterrows():
        feats = extract_features(str(row["equation"]))
        if feats:
            feat_rows.append({**row.to_dict(), **feats})

    df_feat = pd.DataFrame(feat_rows)
    if df_feat.empty:
        print("Feature extraction failed for all equations.")
        return

    # Summary table sorted by f32_gcps
    print(f"\n{'─'*80}")
    print(f"{'Equation'[:45]:<45} {'loss':>8} {'f32 Gcps':>9} {'f64 Gcps':>9} {'speedup':>8}")
    print(f"{'─'*80}")
    for _, row in df_feat.sort_values("f32_gcps", ascending=False).iterrows():
        eq_short = str(row["equation"])[:43]
        f32sp = row.get("f32_speedup", float("nan"))
        print(f"{eq_short:<45} {row['loss']:>8.5f} {row['f32_gcps']:>9.4f} "
              f"{row['f64_gcps']:>9.4f} {f32sp:>7.2f}x")
    print(f"{'─'*80}\n")

    # Correlation with f32_gcps (higher = faster = good)
    feat_cols = [c for c in df_feat.columns
                 if c not in {"equation", "loss", "complexity",
                               "f32_gcps", "f64_gcps", "f32_speedup", "f64_speedup",
                               "baseline_f32_gcps", "baseline_f64_gcps", "n_rows", "outer"}
                 and pd.api.types.is_numeric_dtype(df_feat[c])]

    corrs = {}
    for col in feat_cols:
        series = df_feat[col].dropna()
        if series.nunique() < 2: continue
        aligned = df_feat.loc[series.index, "f32_gcps"].dropna()
        common  = series.index.intersection(aligned.index)
        if len(common) < 3: continue
        r = np.corrcoef(series.loc[common].values, aligned.loc[common].values)[0, 1]
        if not np.isnan(r):
            corrs[col] = r

    if corrs:
        sorted_corrs = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top = sorted_corrs[:args.top]
        print(f"Top {len(top)} features correlated with f32 Gcalls/s  "
              f"(+r = more → faster, −r = more → slower):\n")
        print(f"  {'Feature':<35} {'r':>7}  Direction")
        print(f"  {'─'*55}")
        for feat, r in top:
            direction = "faster" if r > 0 else "slower"
            print(f"  {feat:<35} {r:>+7.3f}  More → {direction}")

        print(f"\nTop speed-BOOSTING features (positive r):")
        for feat, r in sorted(corrs.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {feat:<35} {r:>+7.3f}")
        print(f"\nTop speed-HURTING features (negative r):")
        for feat, r in sorted(corrs.items(), key=lambda kv: kv[1])[:5]:
            print(f"  {feat:<35} {r:>+7.3f}")
    else:
        print("Not enough variance for correlation analysis.")

    # Save
    out = Path(args.output)
    df_feat.to_csv(out, index=False)
    print(f"\nSaved → {out}  ({len(df_feat)} rows, {len(df_feat.columns)} cols)")


if __name__ == "__main__":
    main()
