#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

csv_path = sys.argv[1] if len(sys.argv) > 1 else "sweep.csv"
name = csv_path.split(".")[0]

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
x, ref, jit = data[:, 0], data[:, 1], data[:, 2]
# jit == vm always; only need one
rel_err_pct = data[:, 4] * 100.0   # relative error as %

# thin to ~5k points for clean rendering (log-uniform subsample)
idx = np.unique(np.round(np.linspace(0, len(x) - 1, 5000)).astype(int))
xs, rs, js, ep = x[idx], ref[idx], jit[idx], rel_err_pct[idx]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
fig.suptitle(f"{name} vs  1 / √x", fontsize=13, y=0.98)

# ── panel 1: value comparison, log–log ──────────────────────────────────────
good = np.abs(ep) < 10
ax1.fill_betweenx([1e-10, 1e10], xs[good].min(), xs[good].max(),
                  color="tab:green", alpha=0.08, label="|err| < 10 % region")
ax1.loglog(xs, rs, color="black", lw=1.5, label="1/√x  (reference)")
ax1.loglog(xs, js, color="tab:blue", lw=1.0, alpha=0.85, label="evolved (jit_eval)")
ax1.set_ylabel("value")
ax1.set_ylim(js[js > 0].min() * 0.5, rs.max() * 2)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, which="both", alpha=0.25)
ax1.set_title("Function values  (log–log)", fontsize=10)

# ── panel 2: relative error % ────────────────────────────────────────────────
ax2.fill_betweenx([-110, 110], xs[good].min(), xs[good].max(),
                  color="tab:green", alpha=0.08)
for level, style, label in [(10, "-", "±10 %")]:
    ax2.axhline( level, color="gray", lw=0.8, ls=style)
    ax2.axhline(-level, color="gray", lw=0.8, ls=style, label=label if level == 1 else f"±{level} %")
ax2.axhline(0, color="black", lw=0.8)
ax2.semilogx(xs, ep, color="tab:blue", lw=0.9, alpha=0.9)
ax2.set_xlabel("x")
ax2.set_ylabel("relative error  (%)")
ax2.set_ylim(-105, 105)
ax2.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
ax2.legend(fontsize=9, loc="lower right")
ax2.grid(True, which="both", alpha=0.25)
ax2.set_title("Relative error  (evolved − ref) / ref", fontsize=10)

# annotate the good-region boundaries
x_lo, x_hi = xs[good].min(), xs[good].max()
ax2.annotate(f"x ≈ {x_lo:.2g}", xy=(x_lo, -50), xytext=(x_lo * 0.15, -70),
             arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
             fontsize=8, color="green")
ax2.annotate(f"x ≈ {x_hi:.2g}", xy=(x_hi, -50), xytext=(x_hi * 3, -70),
             arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
             fontsize=8, color="green")

# ── panel 3: error scaled to its own range ──────────────────────────────────
# y-axis limits ARE the data min/max — full panel height = full error excursion.
# a value halfway up is literally halfway between worst and best.
e_min, e_max = ep.min(), ep.max()
e_range = e_max - e_min

ax3.fill_betweenx([e_min, e_max], xs[good].min(), xs[good].max(),
                  color="tab:green", alpha=0.08)
ax3.semilogx(xs, ep, color="tab:blue", lw=0.9, alpha=0.9)
ax3.axhline(0, color="black", lw=0.9)

# draw reference lines only if they fall inside the actual data range
for level, style in [(1, "--"), (10, "-"), (25, ":"),
                     (-1, "--"), (-10, "-"), (-25, ":"), (-50, ":"), (-75, ":")]:
    if e_min <= level <= e_max:
        ax3.axhline(level, color="gray", lw=0.7, ls=style)
        ax3.text(xs.min() * 1.3, level + e_range * 0.01,
                 f"{level:+}%", fontsize=7, color="gray", va="bottom")

ax3.set_ylim(e_min, e_max)          # tight — no padding
ax3.set_xlabel("x")
ax3.set_ylabel("relative error  (%)")
ax3.set_title("Error — axis scaled to actual error range  "
              f"[{e_min:.1f}%, {e_max:.1f}%]", fontsize=10)
ax3.grid(True, which="both", alpha=0.25)

# stats
fin = ep[np.isfinite(ep)]
stat = (f"mean |err| = {np.mean(np.abs(fin)):.1f}%   "
        f"median |err| = {np.median(np.abs(fin)):.1f}%   "
        f"|err|<10%: {(np.abs(fin)<10).mean()*100:.0f}% of range")
fig.text(0.5, 0.01, stat, ha="center", fontsize=8.5, color="#444")

plt.tight_layout(rect=[0, 0.03, 1, 1])
out = csv_path.replace(".csv", "_plot.png")
plt.savefig(out, dpi=150)
print(f"saved → {out}")
plt.show()
