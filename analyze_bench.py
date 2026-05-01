#!/usr/bin/env python3
"""
Filter bench_results.csv to high-accuracy equations, extract structural features
from each equation's AST, then correlate features with GPU throughput.

Usage:
  python analyze_bench.py [--results bench_results.csv] [--threshold 0.30]
"""

import argparse, re, math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

HERE = Path(__file__).parent.resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--results",   default=str(HERE / "bench_results.csv"))
parser.add_argument("--threshold", type=float, default=0.30)
parser.add_argument("--top",       type=int,   default=40)
args = parser.parse_args()

# ── AST ───────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    kind: str       # 'var' | 'num' | 'binop' | 'func'
    value: object = None
    children: list = field(default_factory=list)

    @property
    def left(self):  return self.children[0] if len(self.children) > 0 else None
    @property
    def right(self): return self.children[1] if len(self.children) > 1 else None


# ── tokeniser ─────────────────────────────────────────────────────────────────

@dataclass
class Tok:
    type: str
    value: str

def tokenize(s: str) -> list[Tok]:
    tokens: list[Tok] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in ' \t':
            i += 1
        elif c == '(':
            tokens.append(Tok('LP', '(')); i += 1
        elif c == ')':
            tokens.append(Tok('RP', ')')); i += 1
        elif c == ',':
            tokens.append(Tok('COMMA', ',')); i += 1
        elif c == '+':
            tokens.append(Tok('PLUS', '+')); i += 1
        elif c == '*':
            tokens.append(Tok('STAR', '*')); i += 1
        elif c == '-':
            # negative literal if preceded by an operator / open-paren / nothing
            prev = tokens[-1].type if tokens else None
            if prev in (None, 'LP', 'COMMA', 'PLUS', 'MINUS', 'STAR') \
                    and i + 1 < len(s) and (s[i+1].isdigit() or s[i+1] == '.'):
                j = i + 1
                while j < len(s) and (s[j].isdigit() or s[j] in '.eE'):
                    if s[j] in 'eE' and j + 1 < len(s) and s[j+1] in '+-':
                        j += 2
                    else:
                        j += 1
                tokens.append(Tok('NUM', s[i:j])); i = j
            else:
                tokens.append(Tok('MINUS', '-')); i += 1
        elif c.isdigit() or c == '.':
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] in '.eE'):
                if s[j] in 'eE' and j + 1 < len(s) and s[j+1] in '+-':
                    j += 2
                else:
                    j += 1
            tokens.append(Tok('NUM', s[i:j])); i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(Tok('IDENT', s[i:j])); i = j
        else:
            i += 1
    return tokens


# ── recursive-descent parser ──────────────────────────────────────────────────

class Parser:
    def __init__(self, tokens: list[Tok]):
        self.t = tokens
        self.i = 0

    def peek(self) -> Tok:
        return self.t[self.i] if self.i < len(self.t) else Tok('EOF', '')

    def eat(self, type: Optional[str] = None) -> Tok:
        tok = self.t[self.i]
        if type and tok.type != type:
            raise ValueError(f"Expected {type}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        return self.expr()

    def expr(self) -> Node:
        left = self.term()
        while self.peek().type in ('PLUS', 'MINUS'):
            op = self.eat().value
            right = self.term()
            left = Node('binop', op, [left, right])
        return left

    def term(self) -> Node:
        left = self.factor()
        while self.peek().type == 'STAR':
            self.eat()
            right = self.factor()
            left = Node('binop', '*', [left, right])
        return left

    def factor(self) -> Node:
        tok = self.peek()
        if tok.type == 'LP':
            self.eat('LP')
            node = self.expr()
            self.eat('RP')
            return node
        if tok.type == 'NUM':
            self.eat()
            return Node('num', float(tok.value))
        if tok.type == 'IDENT':
            name = self.eat().value
            if self.peek().type == 'LP':          # function call
                self.eat('LP')
                args = [self.expr()]
                while self.peek().type == 'COMMA':
                    self.eat('COMMA')
                    args.append(self.expr())
                self.eat('RP')
                return Node('func', name, args)
            return Node('var', name)              # plain variable
        if tok.type == 'MINUS':                   # unary minus
            self.eat()
            child = self.factor()
            if child.kind == 'num':
                return Node('num', -child.value)
            return Node('func', 'neg', [child])
        raise ValueError(f"Unexpected token {tok}")


def parse_eq(eq: str) -> Optional[Node]:
    try:
        return Parser(tokenize(eq)).parse()
    except Exception:
        return None


# ── tree queries ──────────────────────────────────────────────────────────────

def height(n: Node) -> int:
    if not n.children: return 0
    return 1 + max(height(c) for c in n.children)

def n_nodes(n: Node) -> int:
    return 1 + sum(n_nodes(c) for c in n.children)

def n_leaves(n: Node) -> int:
    if not n.children: return 1
    return sum(n_leaves(c) for c in n.children)

def vars_in(n: Node) -> set[str]:
    if n.kind == 'var': return {n.value}
    r: set[str] = set()
    for c in n.children: r |= vars_in(c)
    return r

def var_depths(n: Node, var: str, d: int = 0) -> list[int]:
    if n.kind == 'var' and n.value == var: return [d]
    r: list[int] = []
    for c in n.children: r += var_depths(c, var, d + 1)
    return r

def all_consts(n: Node) -> list[float]:
    if n.kind == 'num': return [n.value]
    r: list[float] = []
    for c in n.children: r += all_consts(c)
    return r

def count_kind(n: Node, kind: str, val: object = None) -> int:
    match = (n.kind == kind and (val is None or n.value == val))
    return int(match) + sum(count_kind(c, kind, val) for c in n.children)

def count_func(n: Node, name: str) -> int:
    return count_kind(n, 'func', name)

def count_binop(n: Node, op: str) -> int:
    return count_kind(n, 'binop', op)

def is_var(n: Node, v: str) -> bool:
    return n.kind == 'var' and n.value == v

# Is the root a +/- with one child being bare `s`?
def s_plus_correction(n: Node) -> bool:
    if n.kind != 'binop' or n.value not in ('+', '-'): return False
    return is_var(n.left, 's') or is_var(n.right, 's')

def correction_subtree(n: Node) -> Optional[Node]:
    if not s_plus_correction(n): return None
    return n.right if is_var(n.left, 's') else n.left

# Longest chain of * nodes (mul depth)
def max_mul_chain(n: Node) -> int:
    if n.kind != 'binop' or n.value != '*': return 0
    return 1 + max(max_mul_chain(n.left), max_mul_chain(n.right))

# Count (a*b)+c or a+(b*c) patterns  → FMA-friendly
def count_fma(n: Node) -> int:
    c = 0
    if n.kind == 'binop' and n.value in ('+', '-'):
        if n.left.kind  == 'binop' and n.left.value  == '*': c += 1
        if n.right.kind == 'binop' and n.right.value == '*': c += 1
    return c + sum(count_fma(ch) for ch in n.children)

# Estimate polynomial degree in x  (multiplication adds degrees; other ops take max)
def x_degree(n: Node) -> int:
    if n.kind == 'var':  return 1 if n.value == 'x' else 0
    if n.kind == 'num':  return 0
    if n.kind == 'binop' and n.value == '*':
        return x_degree(n.left) + x_degree(n.right)
    if n.kind in ('binop', 'func'):
        return max((x_degree(c) for c in n.children), default=0)
    return 0

# Is every occurrence of x inside an abs()?
def all_x_in_abs(n: Node, in_abs: bool = False) -> bool:
    if n.kind == 'var' and n.value == 'x': return in_abs
    if n.kind == 'func' and n.value == 'abs':
        return all(all_x_in_abs(c, True) for c in n.children)
    return all(all_x_in_abs(c, in_abs) for c in n.children)

# Depth of the deepest abs() node
def max_abs_depth(n: Node, d: int = 0) -> int:
    best = d if (n.kind == 'func' and n.value == 'abs') else 0
    return max([best] + [max_abs_depth(c, d + 1) for c in n.children])

# Number of distinct constant magnitudes (rounded to 3 sig figs)
def n_distinct_consts(n: Node) -> int:
    cs = all_consts(n)
    rounded = {round(abs(c), 4) for c in cs}
    return len(rounded)

# Does this subtree contain m and x interacting (in same multiply)?
def m_x_in_product(n: Node) -> bool:
    if n.kind == 'binop' and n.value == '*':
        l_vars, r_vars = vars_in(n.left), vars_in(n.right)
        if 'm' in l_vars and 'x' in r_vars: return True
        if 'x' in l_vars and 'm' in r_vars: return True
    return any(m_x_in_product(c) for c in n.children)

# How many abs() calls wrap only constants (no variables)?
def n_abs_const(n: Node) -> int:
    if n.kind == 'func' and n.value == 'abs' and not vars_in(n):
        return 1
    return sum(n_abs_const(c) for c in n.children)

# Number of subtraction nodes where one side is a variable
def n_var_diffs(n: Node) -> int:
    c = 0
    if n.kind == 'binop' and n.value == '-':
        if vars_in(n.left) or vars_in(n.right): c = 1
    return c + sum(n_var_diffs(ch) for ch in n.children)

# Maximum constant absolute value
def max_abs_const(n: Node) -> float:
    cs = all_consts(n)
    return max(abs(c) for c in cs) if cs else 0.0

# Ratio of negative to total constants
def neg_const_ratio(n: Node) -> float:
    cs = all_consts(n)
    if not cs: return 0.0
    return sum(1 for c in cs if c < 0) / len(cs)

# Does any path from root to leaf pass through more than k abs() nodes?
def max_abs_nesting(n: Node, depth: int = 0) -> int:
    d = depth + (1 if n.kind == 'func' and n.value == 'abs' else 0)
    if not n.children: return d
    return max(max_abs_nesting(c, d) for c in n.children)

# Number of nodes that are direct children of a multiply
def n_mul_children(n: Node) -> int:
    c = 0
    if n.kind == 'binop' and n.value == '*':
        c += 2
    return c + sum(n_mul_children(ch) for ch in n.children)

# ── new tree queries ──────────────────────────────────────────────────────────

# Strahler number: classic register-pressure / tree-complexity measure.
# A leaf = 1. If both children have same number k → k+1, else max(k_l, k_r).
def strahler(n: Node) -> int:
    if not n.children: return 1
    vals = sorted([strahler(c) for c in n.children], reverse=True)
    if len(vals) == 1: return vals[0]
    return vals[0] + 1 if vals[0] == vals[1] else vals[0]

# Node counts at each depth level → breadth profile
def breadth_profile(n: Node) -> list[int]:
    from collections import deque
    counts: dict[int, int] = {}
    q: deque = deque([(n, 0)])
    while q:
        node, d = q.popleft()
        counts[d] = counts.get(d, 0) + 1
        for c in node.children: q.append((c, d + 1))
    return [counts[i] for i in range(max(counts) + 1)]

def max_width(n: Node) -> int:
    return max(breadth_profile(n))

def avg_width(n: Node) -> float:
    b = breadth_profile(n)
    return sum(b) / len(b)

# Parallelism ratio: total nodes / height  (wide = parallel, tall = serial)
def parallelism_ratio(n: Node) -> float:
    h = height(n)
    return n_nodes(n) / (h + 1)

# Constant-only subtrees (no variables; compiler can fold to a single constant)
def is_const_only(n: Node) -> bool:
    if n.kind == 'var': return False
    if n.kind == 'num': return True
    return all(is_const_only(c) for c in n.children)

def n_foldable_ops(n: Node) -> int:
    """Operators/funcs whose entire subtree is constant → 1 fold each."""
    if n.kind in ('var', 'num'): return 0
    if is_const_only(n): return 1
    return sum(n_foldable_ops(c) for c in n.children)

def frac_const_leaves(n: Node) -> float:
    nl = n_leaves(n)
    if nl == 0: return 0.0
    return sum(1 for c in all_consts(n) if True) / nl   # all_consts returns one per leaf-num

def frac_var_leaves(n: Node) -> float:
    all_leaf_vars: list[str] = []
    def _collect(node):
        if not node.children and node.kind == 'var': all_leaf_vars.append(node.value)
        for c in node.children: _collect(c)
    _collect(n)
    nl = n_leaves(n)
    return len(all_leaf_vars) / nl if nl else 0.0

# Affine-in-x pattern: (c * x) +/- d  anywhere in tree
def n_affine_x(n: Node) -> int:
    c = 0
    if n.kind == 'binop' and n.value in ('+', '-'):
        for child in n.children:
            if child.kind == 'binop' and child.value == '*':
                lv, rv = vars_in(child.left), vars_in(child.right)
                if (lv == {'x'} and not rv) or (rv == {'x'} and not lv):
                    c += 1
    return c + sum(n_affine_x(ch) for ch in n.children)

# Quadratic bracket pattern: (c1 ± x) * (c2 ± x)  (product of two linear-in-x terms)
def _is_linear_x_term(n: Node) -> bool:
    """(const ± x) or (x ± const)"""
    if n.kind == 'binop' and n.value in ('+', '-'):
        lv, rv = vars_in(n.left), vars_in(n.right)
        return (lv == {'x'} and not rv) or (rv == {'x'} and not lv)
    return False

def n_quadratic_bracket(n: Node) -> int:
    c = 0
    if n.kind == 'binop' and n.value == '*':
        if _is_linear_x_term(n.left) and _is_linear_x_term(n.right):
            c += 1
    return c + sum(n_quadratic_bracket(ch) for ch in n.children)

# Is x only ever used linearly (never x*x or x inside a * with another x subtree)?
def x_is_linear(n: Node) -> bool:
    if n.kind == 'binop' and n.value == '*':
        if 'x' in vars_in(n.left) and 'x' in vars_in(n.right):
            return False
    return all(x_is_linear(c) for c in n.children)

# Is m only ever used linearly (never m*m or m inside * with another m subtree)?
def m_is_linear(n: Node) -> bool:
    if n.kind == 'binop' and n.value == '*':
        if 'm' in vars_in(n.left) and 'm' in vars_in(n.right):
            return False
    return all(m_is_linear(c) for c in n.children)

# Count products where one side has m and the other has x (m×x interactions)
def n_mx_products(n: Node) -> int:
    c = 0
    if n.kind == 'binop' and n.value == '*':
        lv, rv = vars_in(n.left), vars_in(n.right)
        if 'm' in lv and 'x' in rv: c += 1
        elif 'x' in lv and 'm' in rv: c += 1
    return c + sum(n_mx_products(ch) for ch in n.children)

# Count const*const multiplications (compiler can fold)
def n_const_mul(n: Node) -> int:
    c = 0
    if n.kind == 'binop' and n.value == '*':
        if is_const_only(n.left) and is_const_only(n.right): c += 1
    return c + sum(n_const_mul(ch) for ch in n.children)

# Count const+const / const-const additions (also foldable)
def n_const_add(n: Node) -> int:
    c = 0
    if n.kind == 'binop' and n.value in ('+', '-'):
        if is_const_only(n.left) and is_const_only(n.right): c += 1
    return c + sum(n_const_add(ch) for ch in n.children)

# Variables at exactly depth k
def n_vars_at_depth(n: Node, k: int, d: int = 0) -> int:
    if d == k:
        return 1 if n.kind == 'var' else 0
    return sum(n_vars_at_depth(c, k, d + 1) for c in n.children)

# All variable occurrences (any var) with depths
def all_var_depths(n: Node, d: int = 0) -> list[int]:
    r = [d] if n.kind == 'var' else []
    for c in n.children: r += all_var_depths(c, d + 1)
    return r

# Correction-subtree specific queries
def correction_root_op(n: Node) -> Optional[str]:
    corr = correction_subtree(n)
    if corr is None: return None
    return corr.value if corr.kind in ('binop', 'func') else 'leaf'

def correction_has_abs(n: Node) -> bool:
    corr = correction_subtree(n)
    return corr is not None and count_func(corr, 'abs') > 0

def correction_has_greater(n: Node) -> bool:
    corr = correction_subtree(n)
    return corr is not None and count_func(corr, 'greater') > 0

def correction_is_product(n: Node) -> bool:
    corr = correction_subtree(n)
    return corr is not None and corr.kind == 'binop' and corr.value == '*'

def correction_is_sum(n: Node) -> bool:
    corr = correction_subtree(n)
    return corr is not None and corr.kind == 'binop' and corr.value in ('+', '-')

def correction_var_depths(n: Node, var: str) -> list[int]:
    corr = correction_subtree(n)
    return var_depths(corr, var) if corr is not None else []

# Fraction of internal nodes that have at least one variable descendant
def frac_internal_with_vars(n: Node) -> float:
    internals = []; has_var = []
    def visit(node):
        if node.children:
            internals.append(1)
            has_var.append(1 if vars_in(node) else 0)
        for c in node.children: visit(c)
    visit(n)
    return (sum(has_var) / len(internals)) if internals else 0.0

# Longest path that only passes through + and - nodes (add-chain depth)
def max_add_chain(n: Node) -> int:
    if n.kind == 'binop' and n.value in ('+', '-'):
        return 1 + max(max_add_chain(n.left), max_add_chain(n.right))
    return 0

# Does root-level operation mix + and * (vs pure + tree or pure * tree)?
def uses_both_add_and_mul(n: Node) -> bool:
    has_add = count_binop(n, '+') + count_binop(n, '-') > 0
    has_mul = count_binop(n, '*') > 0
    return has_add and has_mul

# Depth of s relative to tree height (0 = root, 1 = leaf level)
def s_relative_depth(n: Node) -> float:
    h = height(n)
    sd = var_depths(n, 's')
    if not sd or h == 0: return 0.0
    return min(sd) / h

# Depth of x relative to tree height
def x_relative_depth(n: Node) -> float:
    h = height(n)
    xd = var_depths(n, 'x')
    if not xd or h == 0: return 0.0
    return float(np.mean(xd)) / h

# Depth of m relative to tree height
def m_relative_depth(n: Node) -> float:
    h = height(n)
    md = var_depths(n, 'm')
    if not md or h == 0: return 0.0
    return float(np.mean(md)) / h

# Ratio of x depth to m depth (x shallower than m → positive signal from earlier results)
def x_to_m_depth_ratio(n: Node) -> float:
    xd = var_depths(n, 'x'); md = var_depths(n, 'm')
    if not xd or not md: return 1.0
    return float(np.mean(xd)) / float(np.mean(md))

# Number of abs() calls that wrap a subtree containing both m and x
def n_abs_mx(n: Node) -> int:
    c = 0
    if n.kind == 'func' and n.value == 'abs':
        v = vars_in(n)
        if 'm' in v and 'x' in v: c += 1
    return c + sum(n_abs_mx(ch) for ch in n.children)

# How many distinct operator types appear (variety of ops)
def n_distinct_ops(n: Node) -> int:
    ops: set = set()
    def visit(node):
        if node.kind == 'binop': ops.add(node.value)
        if node.kind == 'func':  ops.add(node.value)
        for c in node.children: visit(c)
    visit(n)
    return len(ops)


# ── full feature extraction ───────────────────────────────────────────────────

def extract_features(eq: str) -> dict:
    # ── regex / string features (fast, always available) ──────────────────────
    n_add      = eq.count('+')
    n_sub      = eq.count('-') - len(re.findall(r'-\d', eq))  # subtract negative literals
    n_mul      = eq.count('*')
    n_abs      = eq.count('abs(')
    n_exp      = eq.count('exp(')
    n_greater  = eq.count('greater(')
    n_m        = len(re.findall(r'\bm\b', eq))
    n_x        = len(re.findall(r'\bx\b', eq))
    n_constants= len(re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', eq))
    depth_paren= 0; mx = 0; d = 0
    for ch in eq:
        if ch == '(': d += 1; mx = max(mx, d)
        elif ch == ')': d -= 1
    depth_paren = mx
    n_parens   = eq.count('(')
    eq_len     = len(eq)
    n_total_ops= n_add + n_sub + n_mul
    n_calls    = n_abs + n_exp + n_greater

    s_additive      = int(bool(re.search(r'^\s*s\s*[+-]|[+-]\s*s\s*$', eq)))
    abs_s           = int('abs(s' in eq)
    abs_x_direct    = int(bool(re.search(r'abs\(\s*x\s*\)', eq)))
    abs_x_plus_x    = int(bool(re.search(r'abs\(\s*x\s*\)\s*\+\s*x|x\s*\+\s*abs\(\s*x\s*\)', eq)))
    m_x_diff        = int(bool(re.search(r'\bm\s*-\s*x\b|\bx\s*-\s*m\b', eq)))
    nested_abs      = int(bool(re.search(r'abs\([^)]*abs\(', eq)))
    x_repeated      = int(n_x > 1)
    m_repeated      = int(n_m > 1)
    s_plus_const    = int(bool(re.match(r'^\s*s\s*[+-]\s*-?[\d.e+\-]+\s*$', eq.strip())))
    outer_abs       = int(bool(re.match(r'^\s*abs\(', eq.strip())))
    m_times_x       = int(bool(re.search(r'\bm\s*\*\s*x\b|\bx\s*\*\s*m\b', eq)))
    uses_m_and_x    = int(n_m > 0 and n_x > 0)
    double_neg      = int('- -' in eq or '--' in eq)
    x_sq_pattern    = int(bool(re.search(r'\bx\s*\*\s*x\b', eq)))
    m_sq_pattern    = int(bool(re.search(r'\bm\s*\*\s*m\b', eq)))
    s_in_mul        = int(bool(re.search(r'\bs\s*\*|\*\s*s\b', eq)))

    ops_per_depth   = n_total_ops / (depth_paren + 1)
    calls_per_depth = n_calls     / (depth_paren + 1)
    const_per_op    = n_constants / (n_total_ops + 1)
    vars_used       = int(n_m > 0) + int(n_x > 0)
    abs_x_depth     = n_abs * depth_paren
    ops_x_depth     = n_total_ops * depth_paren
    m_x_ops         = n_m * n_x

    regex_feats = dict(
        n_add=n_add, n_sub=n_sub, n_mul=n_mul,
        n_abs=n_abs, n_exp=n_exp, n_greater=n_greater,
        n_m=n_m, n_x=n_x, n_constants=n_constants,
        depth_paren=depth_paren, n_parens=n_parens, eq_len=eq_len,
        n_calls=n_calls, n_total_ops=n_total_ops,
        has_abs=int(n_abs > 0), has_exp=int(n_exp > 0), has_greater=int(n_greater > 0),
        s_additive=s_additive, abs_s=abs_s, abs_x_direct=abs_x_direct,
        abs_x_plus_x=abs_x_plus_x, m_x_diff=m_x_diff, nested_abs=nested_abs,
        x_repeated=x_repeated, m_repeated=m_repeated, s_plus_const=s_plus_const,
        outer_abs=outer_abs, m_times_x=m_times_x, uses_m_and_x=uses_m_and_x,
        double_neg=double_neg, x_sq_pattern=x_sq_pattern, m_sq_pattern=m_sq_pattern,
        s_in_mul=s_in_mul,
        ops_per_depth=ops_per_depth, calls_per_depth=calls_per_depth,
        const_per_op=const_per_op, vars_used=vars_used,
        abs_x_depth=abs_x_depth, ops_x_depth=ops_x_depth, m_x_ops=m_x_ops,
    )

    # ── AST features ──────────────────────────────────────────────────────────
    tree = parse_eq(eq)
    if tree is None:
        return {**regex_feats,
                **{k: float('nan') for k in [
                    'tree_n_nodes','tree_n_leaves','tree_n_internal','tree_leaf_ratio',
                    'tree_height','tree_balance','s_min_depth','m_avg_depth','x_avg_depth',
                    'x_min_depth','m_min_depth','root_is_add','root_is_mul','root_is_sub',
                    'root_is_abs','root_is_greater','s_plus_correction','s_at_root_child',
                    'correction_height','correction_n_nodes','correction_uses_m',
                    'correction_uses_x','correction_uses_both','correction_x_degree',
                    'n_fma','x_poly_degree','max_mul_chain',
                    'const_max_abs','const_min_abs','const_mean_abs',
                    'n_small_consts','n_large_consts','const_range','neg_const_ratio',
                    'n_distinct_consts','all_x_in_abs','max_abs_nesting',
                    'n_abs_const','n_var_diffs','m_x_in_product','n_mul_children',
                ]}}

    h      = height(tree)
    nn     = n_nodes(tree)
    nl     = n_leaves(tree)
    ni     = nn - nl
    s_d    = var_depths(tree, 's')
    m_d    = var_depths(tree, 'm')
    x_d    = var_depths(tree, 'x')
    consts = all_consts(tree)
    abs_cs = [abs(c) for c in consts]

    corr   = correction_subtree(tree)
    spc    = s_plus_correction(tree)

    root_val = tree.value if tree.kind in ('binop', 'func') else None

    ast_feats = dict(
        # tree shape
        tree_n_nodes    = nn,
        tree_n_leaves   = nl,
        tree_n_internal = ni,
        tree_leaf_ratio = nl / nn,
        tree_height     = h,
        tree_balance    = h / (math.log2(nl) + 1) if nl > 1 else 1.0,

        # variable depths
        s_min_depth  = min(s_d) if s_d else 0,
        m_avg_depth  = float(np.mean(m_d)) if m_d else 0.0,
        x_avg_depth  = float(np.mean(x_d)) if x_d else 0.0,
        x_min_depth  = min(x_d) if x_d else 99,
        m_min_depth  = min(m_d) if m_d else 99,

        # root operation
        root_is_add     = int(root_val == '+'),
        root_is_mul     = int(root_val == '*'),
        root_is_sub     = int(root_val == '-'),
        root_is_abs     = int(root_val == 'abs'),
        root_is_greater = int(root_val == 'greater'),

        # s structure
        s_plus_correction = int(spc),
        s_at_root_child   = int(any(d == 1 for d in s_d)),

        # correction subtree (the non-s part when root is s + correction)
        correction_height   = height(corr)    if corr else 0,
        correction_n_nodes  = n_nodes(corr)   if corr else 0,
        correction_uses_m   = int('m' in vars_in(corr)) if corr else 0,
        correction_uses_x   = int('x' in vars_in(corr)) if corr else 0,
        correction_uses_both= int({'m','x'} <= vars_in(corr)) if corr else 0,
        correction_x_degree = x_degree(corr) if corr else 0,

        # arithmetic patterns
        n_fma           = count_fma(tree),
        x_poly_degree   = x_degree(tree),
        max_mul_chain   = max_mul_chain(tree),

        # constant statistics
        const_max_abs   = max(abs_cs) if abs_cs else 0.0,
        const_min_abs   = min(abs_cs) if abs_cs else 0.0,
        const_mean_abs  = float(np.mean(abs_cs)) if abs_cs else 0.0,
        n_small_consts  = sum(1 for c in abs_cs if c < 0.1),
        n_large_consts  = sum(1 for c in abs_cs if c > 1.0),
        const_range     = (max(abs_cs) - min(abs_cs)) if abs_cs else 0.0,
        neg_const_ratio = neg_const_ratio(tree),
        n_distinct_consts = n_distinct_consts(tree),

        # structural patterns
        all_x_in_abs    = int(all_x_in_abs(tree)),
        max_abs_nesting = max_abs_nesting(tree),
        n_abs_const     = n_abs_const(tree),
        n_var_diffs     = n_var_diffs(tree),
        m_x_in_product  = int(m_x_in_product(tree)),
        n_mul_children  = n_mul_children(tree),

        # register pressure / parallelism
        strahler           = strahler(tree),
        max_width          = max_width(tree),
        avg_width          = avg_width(tree),
        parallelism_ratio  = parallelism_ratio(tree),

        # constant folding opportunities
        n_foldable_ops    = n_foldable_ops(tree),
        n_const_mul       = n_const_mul(tree),
        n_const_add       = n_const_add(tree),
        frac_const_leaves = frac_const_leaves(tree),
        frac_var_leaves   = frac_var_leaves(tree),
        const_std         = float(np.std(abs_cs)) if len(abs_cs) > 1 else 0.0,

        # polynomial / algebraic patterns
        n_affine_x         = n_affine_x(tree),
        n_quadratic_bracket= n_quadratic_bracket(tree),
        x_is_linear        = int(x_is_linear(tree)),
        m_is_linear        = int(m_is_linear(tree)),
        n_mx_products      = n_mx_products(tree),
        n_abs_mx           = n_abs_mx(tree),

        # variable depth distribution
        all_var_depth_mean  = float(np.mean(all_var_depths(tree))) if all_var_depths(tree) else 0.0,
        all_var_depth_std   = float(np.std(all_var_depths(tree)))  if len(all_var_depths(tree)) > 1 else 0.0,
        all_var_depth_max   = max(all_var_depths(tree)) if all_var_depths(tree) else 0,
        x_depth_std         = float(np.std(x_d)) if len(x_d) > 1 else 0.0,
        m_depth_std         = float(np.std(m_d)) if len(m_d) > 1 else 0.0,
        n_vars_at_depth_0   = n_vars_at_depth(tree, 0),
        n_vars_at_depth_1   = n_vars_at_depth(tree, 1),
        n_vars_at_depth_2   = n_vars_at_depth(tree, 2),
        n_vars_at_depth_3   = n_vars_at_depth(tree, 3),
        s_relative_depth    = s_relative_depth(tree),
        x_relative_depth    = x_relative_depth(tree),
        m_relative_depth    = m_relative_depth(tree),
        x_to_m_depth_ratio  = x_to_m_depth_ratio(tree),

        # correction subtree deep features
        correction_is_product  = int(correction_is_product(tree)),
        correction_is_sum      = int(correction_is_sum(tree)),
        correction_has_abs     = int(correction_has_abs(tree)),
        correction_has_greater = int(correction_has_greater(tree)),
        correction_root_is_mul = int(correction_root_op(tree) == '*'),
        correction_root_is_add = int(correction_root_op(tree) in ('+', '-')),
        correction_root_is_abs = int(correction_root_op(tree) == 'abs'),
        correction_x_min_depth = min(correction_var_depths(tree, 'x')) if correction_var_depths(tree, 'x') else 99,
        correction_m_min_depth = min(correction_var_depths(tree, 'm')) if correction_var_depths(tree, 'm') else 99,

        # operation diversity & chain features
        n_distinct_ops       = n_distinct_ops(tree),
        uses_both_add_mul    = int(uses_both_add_and_mul(tree)),
        max_add_chain        = max_add_chain(tree),
        frac_internal_w_vars = frac_internal_with_vars(tree),
    )

    return {**regex_feats, **ast_feats}


# ── load & filter ─────────────────────────────────────────────────────────────

df = pd.read_csv(args.results)
df = df[df["loss"].notna()].copy()

min_loss    = df["loss"].min()
cutoff      = min_loss * (1 + args.threshold)
df_filtered = df[df["loss"] <= cutoff].copy()

print(f"Total equations with loss data : {len(df)}")
print(f"Min loss                       : {min_loss:.6f}")
print(f"Cutoff (×{1+args.threshold:.2f})               : {cutoff:.6f}")
print(f"Equations within threshold     : {len(df_filtered)}")
print()

feat_df = df_filtered["equation"].apply(extract_features).apply(pd.Series)
feat_df["complexity"] = df_filtered["complexity"].values

analysis = pd.concat([
    df_filtered[["equation","loss","throughput_gups","us_per_seq"]].reset_index(drop=True),
    feat_df.reset_index(drop=True),
], axis=1)

# ── correlation ───────────────────────────────────────────────────────────────

feature_cols = feat_df.columns.tolist()
corr = analysis[feature_cols + ["throughput_gups"]].corr()["throughput_gups"].drop("throughput_gups")

# drop NaN and zero-variance features
corr = corr.dropna()
corr_sorted = corr.abs().sort_values(ascending=False)

print(f"── Feature correlations with throughput_gups (|r|, top {args.top}) ──")
print(f"{'feature':<26}  {'r':>8}  {'|r|':>8}")
print("─" * 48)
for feat in corr_sorted.index[:args.top]:
    r = corr[feat]
    print(f"{feat:<26}  {r:+8.4f}  {abs(r):8.4f}")

# ── summary table ─────────────────────────────────────────────────────────────

print(f"\n── Top 15 high-accuracy equations by throughput ──")
cols = ["equation","loss","throughput_gups","complexity",
        "s_plus_correction","correction_x_degree","max_mul_chain","n_fma","tree_height"]
top = (analysis.sort_values("throughput_gups", ascending=False)
               .head(15)[cols])
pd.set_option("display.max_colwidth", 55)
pd.set_option("display.width", 140)
print(top.to_string(index=False))

# ── save ──────────────────────────────────────────────────────────────────────

out = HERE / "equation_features.csv"
analysis.to_csv(out, index=False)
print(f"\nFull feature table → {out}  ({len(analysis)} rows, {len(feature_cols)} features)")
