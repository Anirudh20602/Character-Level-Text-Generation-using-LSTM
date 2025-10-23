
# Making runs reproducible so my results are consistent
import os, random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Loading libraries I'll use
import matplotlib.pyplot as plt
from typing import Dict, Any

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import Callback
    TF_AVAILABLE = True
    print("TensorFlow version:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    print("TensorFlow isn't installed in this runtime. I'll still set up the notebook;")
    print("install it when you want to train: pip install tensorflow==2.15.*")



# Using the uppercase English alphabet
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
n_classes = len(alphabet)

# Mappings
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for i, c in enumerate(alphabet)}

# Build input->target pairs
X_idx, y_idx = [], []
for i in range(n_classes - 1):
    X_idx.append(char_to_int[alphabet[i]])
    y_idx.append(char_to_int[alphabet[i+1]])

import numpy as np
X_idx = np.array(X_idx, dtype=np.int32)
y_idx = np.array(y_idx, dtype=np.int32)

# One-hot inputs instead of normalized indices (this trains faster & cleaner)
X_onehot = np.eye(n_classes, dtype=np.float32)[X_idx]     # (25, 26)
X_lstm = X_onehot.reshape((-1, 1, n_classes))             # (25, 1, 26)

# One-hot targets
if 'tf' in globals() and TF_AVAILABLE:
    y_onehot = tf.keras.utils.to_categorical(y_idx, num_classes=n_classes)
else:
    y_onehot = np.eye(n_classes, dtype=np.float32)[y_idx]

print("X_lstm shape:", X_lstm.shape, "| y_onehot shape:", y_onehot.shape)



def build_lstm_model(hidden_size: int = 96, dropout: float = 0.0, lr: float = 1e-3):
    if 'tf' not in globals() or not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build and train the model.")
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(1, X_lstm.shape[-1])))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model



from tensorflow.keras.callbacks import Callback

class TargetAccuracy(Callback):
    def __init__(self, target=0.95):
        super().__init__()
        self.target = target
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy")
        if acc is not None and acc >= self.target:
            print(f"\nReached target accuracy {acc:.3f} at epoch {epoch+1}. Stopping.")
            self.model.stop_training = True



if 'tf' in globals() and TF_AVAILABLE:
    model = build_lstm_model(hidden_size=96, dropout=0.0, lr=1e-3)
    cb = TargetAccuracy(target=0.95)
    history = model.fit(
        X_lstm, y_onehot,
        epochs=600,
        batch_size=1,
        shuffle=True,
        verbose=0,
        callbacks=[cb]
    )
    loss, acc = model.evaluate(X_lstm, y_onehot, verbose=0)
    print(f"Baseline training accuracy: {acc:.4f}")
else:
    print("Skipping training because TensorFlow isn't available here.")



if 'tf' in globals() and TF_AVAILABLE:
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
else:
    print("No plots since training didn't run in this environment.")



def predict_next(model, ch: str) -> str:
    idx = char_to_int[ch]
    x = np.eye(n_classes, dtype=np.float32)[[idx]]
    x = x.reshape((1,1,n_classes))
    probs = model.predict(x, verbose=0)[0]
    return int_to_char[int(np.argmax(probs))]

if 'tf' in globals() and TF_AVAILABLE:
    test_letters = list("ABCDWXYZ")
    print("Predictions (input -> predicted next):")
    for t in test_letters:
        print(f"{t} -> {predict_next(model, t)}")



import pandas as pd
if 'tf' in globals() and TF_AVAILABLE:
    trial_cfgs = [
        {"hidden": 64, "lr": 1e-3},
        {"hidden": 96, "lr": 1e-3},
        {"hidden": 128, "lr": 5e-4},
    ]
    rows = []
    for cfg in trial_cfgs:
        m = build_lstm_model(hidden_size=cfg["hidden"], lr=cfg["lr"])
        h = m.fit(X_lstm, y_onehot, epochs=400, batch_size=1, shuffle=True, verbose=0, callbacks=[TargetAccuracy(0.95)])
        _, a = m.evaluate(X_lstm, y_onehot, verbose=0)
        rows.append({"hidden": cfg["hidden"], "lr": cfg["lr"], "accuracy": float(a)})
    df_trials = pd.DataFrame(rows)
    print(df_trials)
else:
    print("Skip ablation if TF isn't installed.")
