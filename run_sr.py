import numpy as np
import pandas as pd
import time
import signal
import atexit
import pickle
from pathlib import Path
from pysr import PySRRegressor

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_SIZE  = 10_000
REFRESH_EVERY = 10      # SR iterations between data refresh
REFRESH_FRAC  = 0.20    # fraction replaced (None = full regen)
SAVE_EVERY    = 100     # SR iterations between model saves
model_file    = Path('model.pkl')
# ─────────────────────────────────────────────────────────────────────────────

def gen_chunk(n: int) -> pd.DataFrame:
    data, m, s = [], 0.0, 0.0
    for _ in range(n):
        x = np.random.randn()
        m_new = max(m, x)
        if m_new > m:
            s *= np.exp(m - m_new)
            m = m_new
        s_new = s + np.exp(x - m)
        data.append([m, s, x, s_new])
        s = s_new
    return pd.DataFrame(data, columns=['m', 's', 'x', 's_new'])

def refresh(df: pd.DataFrame) -> pd.DataFrame:
    if REFRESH_FRAC is None:
        return gen_chunk(DATASET_SIZE)
    n_new = int(DATASET_SIZE * REFRESH_FRAC)
    return pd.concat([df.iloc[n_new:], gen_chunk(n_new)], ignore_index=True)

model = None

def save_model():
    if model is not None:
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ saved @ {time.strftime('%H:%M:%S')}")

def load_model() -> bool:
    global model
    for path in [model_file, Path('hall_of_fame.pkl')]:
        if path.exists():
            model = (pickle.load(open(path, 'rb')) if path.suffix == '.pkl' and path == model_file
                     else PySRRegressor.from_file(str(path)))
            print(f"✓ resumed from {path}")
            return True
    return False

signal.signal(signal.SIGINT, lambda s, f: (save_model(), exit(0)))
atexit.register(save_model)

if not load_model():
    model = PySRRegressor(
        niterations=1,          # ← 1 SR iteration per .fit() call
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        binary_operators=['+', '-', '*'],
        unary_operators=['abs', 'neg'],
        maxsize=20,
        warm_start=True,        # ← state carries over between .fit() calls
        batching=True,
        batch_size=1000,
        progress=False,         # too noisy at 1 iter/call
        verbosity=0,
    )

df = gen_chunk(DATASET_SIZE)
print(f"✓ initial dataset: {len(df)} rows")

for i in range(100_000):
    model.fit(df[['m', 's', 'x']], df['s_new'])  # 1 SR iteration

    if i % REFRESH_EVERY == 0 and i > 0:
        df = refresh(df)
        best = model.equations_.iloc[-1]
        print(f"[iter {i:>6}] refreshed data | best loss={best['loss']:.4f} eq={best['equation']}")
