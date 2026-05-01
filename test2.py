import numpy as np
import pandas as pd
import time
import signal
import threading
import atexit
import pickle
from pathlib import Path
from pysr import PySRRegressor

# Dataset (load if exists)
dataset_file = Path('softmax_incremental.csv')
if not dataset_file.exists():
    print("Generating dataset...")
    n_steps = 10000
    data = []
    m = s = 0.0
    for _ in range(n_steps):
        x = np.random.randn()
        m_new = max(m, x)
        if m_new > m:
            s *= np.exp(m - m_new)
            m = m_new
        s_new = s + np.exp(x - m)
        data.append([m, s, x, s_new])
        s = s_new
    df = pd.DataFrame(data, columns=['m', 's', 'x', 's_new'])
    df.to_csv(dataset_file, index=False)
    print(f"✓ Dataset created: {len(df)} rows")
else:
    df = pd.read_csv(dataset_file)
    print(f"✓ Dataset loaded: {len(df)} rows")

model_file = Path('model.pkl')
HOF_FILE = Path('hall_of_fame.csv')  # PySR auto-saves this during fit
SAVE_INTERVAL = 3600  # 1h
model = None

def save_model():
    """Save model via pickle."""
    if model is not None:
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved model.pkl at {time.strftime('%H:%M:%S')}")

def load_model():
    """Load model: prefer our pickle, fall back to PySR's auto-saved HOF pkl."""
    global model
    # Check for PySR's auto-saved hall-of-fame pkl (written during fit)
    hof_pkl = Path('hall_of_fame.pkl')
    if model_file.exists():
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print("✓ Loaded model.pkl (will resume)")
        return True
    elif hof_pkl.exists():
        # PySR writes this automatically — from_file() reconstructs the model
        model = PySRRegressor.from_file(str(hof_pkl))
        print("✓ Loaded hall_of_fame.pkl via from_file (will resume)")
        return True
    return False

# Graceful exit
signal.signal(signal.SIGINT, lambda s, f: (save_model(), exit(0)))
atexit.register(save_model)

# Create/load model
if not load_model():
    print("New PySR session...")
    model = PySRRegressor(
        niterations=100,
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        binary_operators=['+', '-', '*'],
        unary_operators=['abs', 'neg'],
        maxsize=20,
        warm_start=True,
        batching=True,
        batch_size=1000,
        progress=True,
        verbosity=1,
    )

# Auto-save thread
def auto_save():
    while True:
        time.sleep(SAVE_INTERVAL)
        save_model()

threading.Thread(target=auto_save, daemon=True).start()

# Run!
print("🚀 PySR fit starting/resuming...")
model.fit(df[['m', 's', 'x']], df['s_new'])

print("\n" + "=" * 60)
print("BEST EQUATIONS:")
print(model.equations_)
print("=" * 60)

save_model()
print("✅ Done!")
