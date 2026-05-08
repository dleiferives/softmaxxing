import numpy as np
import pandas as pd
import time
import signal
import atexit
import pickle
from pathlib import Path
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=100,          # ← 1 SR iteration per .fit() call
    elementwise_loss="loss(prediction, target) = abs(prediction - target)",
    binary_operators=['+', '-', '*'],
    unary_operators=['abs', 'neg'],
    maxsize=20,
    warm_start=True,        # ← state carries over between .fit() calls
    batching=True,
    batch_size=1000,
    progress=True,         # too noisy at 1 iter/call
    verbosity=1,
)

def gen_chunk(n: int) -> pd.DataFrame:
    x = np.random.randn(n).astype("float32")
    y = 1 / np.sqrt(x)

    df = pd.DataFrame({'x': x, 'y': y}, dtype="float32")
    df = df[np.isfinite(df['y'])]
    return df

df = gen_chunk(10000);

model.fit(df[['x']], df['y'])
