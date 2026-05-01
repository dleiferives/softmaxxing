import argparse
import numpy as np
import sklearn.base
from sklearn.utils.validation import validate_data

# sklearn 1.7 removed _validate_data; patch it back for HROCH compatibility
if not hasattr(sklearn.base.BaseEstimator, "_validate_data"):
    sklearn.base.BaseEstimator._validate_data = lambda self, *a, **kw: validate_data(self, *a, **kw)

from HROCH import SymbolicRegressor

DATASET_SIZE = 100_000
NUM_THREADS  = 8


def sample_x(rng, normal_std):
    if rng.random() < 0.1:
        x = np.nan
        while not np.isfinite(x):
            x = rng.integers(0, 2**32, dtype=np.uint32).view(np.float32).item()
        return float(x)
    return rng.normal(0.0, normal_std)


def gen_data(n: int, rng=None, min_seq=2, max_seq=1000):
    if rng is None:
        rng = np.random.default_rng()
    # normal std so that 80% of values fall in [-2, 2]: 2/1.2816 ≈ 1.56
    normal_std = 2.0 / 1.2816
    X = np.empty((n, 3), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        # log-uniform sequence length
        seq_len = int(np.exp(rng.uniform(np.log(min_seq), np.log(max_seq))))
        seq_len = min(seq_len, n - i)
        m, s = 0.0, 1.0
        for _ in range(seq_len):
            x = sample_x(rng, normal_std)
            m_new = max(m, x)
            if m_new > m:
                s *= np.exp(m - m_new)
                m = m_new
            s_new = s + np.exp(x - m)
            X[i] = [m, s, x]
            y[i] = s_new
            s = s_new
            i += 1
    return X, y


def refresh(X, y, frac: float, rng):
    n_drop = int(len(y) * frac)
    X_new, y_new = gen_data(n_drop, rng, max_seq=len(y) // 10)
    return np.concatenate([X[n_drop:], X_new]), np.concatenate([y[n_drop:], y_new])


def print_hof(reg, X, y, iteration: int):
    models = reg.get_models()
    if not models:
        return
    best = min(models, key=lambda m: np.mean(np.abs(m.predict(X) - y)))
    mae = np.mean(np.abs(best.predict(X) - y))
    print(f"iter {iteration:4d}  mae={mae:.6g}  eq={best.equation}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=3.0, help="seconds per fit call")
    parser.add_argument("--refresh-frac", type=float, default=1.0, help="fraction of dataset to swap each iteration")
    args = parser.parse_args()

    rng = np.random.default_rng()
    print(f"Generating {DATASET_SIZE} samples...")
    X, y = gen_data(DATASET_SIZE, rng, max_seq=DATASET_SIZE // 10)

    reg = SymbolicRegressor(
        num_threads=NUM_THREADS,
        time_limit=args.time_limit,
        precision="f32",
        problem={'add': 1.0, 'mul': 1.0, 'sub': 1.0, 'abs': 1.0},
        code_settings={"min_size": 10, "max_size": 20, "const_size": 8},
        metric="MAE",
        verbose=0,
        warm_start=True,
    )

    print(f"time_limit={args.time_limit}s  refresh_frac={args.refresh_frac}  threads={NUM_THREADS}")
    print("Ctrl-C to stop\n")

    i = 0
    try:
        while True:
            i += 1
            reg.fit(X, y)
            X, y = refresh(X, y, args.refresh_frac, rng)
            print_hof(reg, X, y, i)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
