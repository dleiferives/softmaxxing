import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_eq(arg):
    # accept full iter line: "iter  27  mae=0.00413  eq=((x1+...))"
    m = re.search(r'eq=(.+)$', arg.strip())
    return m.group(1).strip() if m else arg.strip()

def eval_eq(eq_str, x):
    expr = eq_str.replace('abs', 'np.abs')
    return eval(expr, {"np": np, "x1": x, "__builtins__": {}})

if len(sys.argv) < 2:
    print("usage: python plot_approx.py '<eq or iter line>' [xmin] [xmax]")
    sys.exit(1)

eq_str = parse_eq(sys.argv[1])
xmin = float(sys.argv[2]) if len(sys.argv) > 2 else -3
xmax = float(sys.argv[3]) if len(sys.argv) > 3 else 3

x = np.linspace(xmin, xmax, 1000)
approx = eval_eq(eq_str, x)

mae = np.mean(np.abs(np.exp(x) - approx))
print(f"mae over [{xmin}, {xmax}]: {mae:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(x, np.exp(x), label="$e^x$")
plt.plot(x, approx, label=eq_str[:60] + ("..." if len(eq_str) > 60 else ""), linestyle="--")
plt.legend(fontsize=8)
plt.xlabel("x")
plt.title(f"$e^x$ vs approx  (mae={mae:.5f})")
plt.tight_layout()
out = "scripts/plot_approx.png"
plt.savefig(out, dpi=150)
plt.show()
print(f"saved {out}")
