import os, json, numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _find_out():
    d = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "docs")) or os.path.isdir(os.path.join(d, ".git")):
            return os.path.join(d, "docs", "data")
        d = os.path.dirname(d)
    raise RuntimeError("Could not locate the repo root — set OUT manually.")
OUT = _find_out(); os.makedirs(OUT, exist_ok=True)

# --- config ------------------------------------------------------------------
DIMS      = [1, 2, 8, 32]     # 1 is included because the failure is instructive
EPOCHS    = 8
BATCH     = 128
N_SHOW    = 12                # test images shown per row
HIDDEN    = 128               # pre-latent width — wide enough not to bottleneck
DATA      = mnist_digits      # (N,28,28,1) in [0,1]
# -----------------------------------------------------------------------------

# reuse the notebook's Sampling / VAE if they exist, else define equivalents
try:
    Sampling
except NameError:
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            eps = tf.random.normal(tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * eps

try:
    VAE
except NameError:
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kw):
            super().__init__(**kw)
            self.encoder, self.decoder = encoder, decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        @property
        def metrics(self):
            return [self.total_loss_tracker, self.reconstruction_loss_tracker,
                    self.kl_loss_tracker]
        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                rec = self.decoder(z)
                r = tf.reduce_mean(tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, rec), axis=(1, 2)))
                k = tf.reduce_mean(tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
                loss = r + k
            g = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(g, self.trainable_weights))
            self.total_loss_tracker.update_state(loss)
            self.reconstruction_loss_tracker.update_state(r)
            self.kl_loss_tracker.update_state(k)
            return {m.name: m.result() for m in self.metrics}

def build(latent_dim):
    i = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(i)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(HIDDEN, activation="relu")(x)
    zm = layers.Dense(latent_dim, name="z_mean")(x)
    zv = layers.Dense(latent_dim, name="z_log_var")(x)
    enc = keras.Model(i, [zm, zv, Sampling()([zm, zv])], name=f"enc{latent_dim}")

    li = keras.Input(shape=(latent_dim,))
    y = layers.Dense(7 * 7 * 64, activation="relu")(li)
    y = layers.Reshape((7, 7, 64))(y)
    y = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(y)
    y = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(y)
    y = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(y)
    dec = keras.Model(li, y, name=f"dec{latent_dim}")
    return enc, dec

rng = np.random.default_rng(7)
show_idx = rng.permutation(len(DATA))[:N_SHOW]
originals = DATA[show_idx]

def u8(a): return (np.clip(a, 0, 1) * 255).astype("uint8")
def tile(rows, pad=1, bg=40):
    """rows: list of (n,28,28) float -> one sheet, one row per entry."""
    R, N = len(rows), rows[0].shape[0]
    H = W = rows[0].shape[1]
    sheet = np.full((R*(H+pad)-pad, N*(W+pad)-pad), bg, "uint8")
    for r, row in enumerate(rows):
        for c in range(N):
            sheet[r*(H+pad):r*(H+pad)+H, c*(W+pad):c*(W+pad)+W] = u8(row[c])
    return sheet

recon_rows = [originals.squeeze(-1)]
prior_rows, records = [], []

for d in DIMS:
    print(f"\n=== latent_dim = {d} ===", flush=True)
    enc, dec = build(d)
    m = VAE(enc, dec)
    m.compile(optimizer=keras.optimizers.Adam())
    h = m.fit(DATA, epochs=EPOCHS, batch_size=BATCH, verbose=2)

    zm, _, _ = enc.predict(originals, verbose=0)
    recon_rows.append(dec.predict(zm, verbose=0).squeeze(-1))
    prior_rows.append(
        dec.predict(rng.normal(size=(N_SHOW, d)).astype("float32"), verbose=0).squeeze(-1))

    hist = h.history
    key = "reconstruction_loss" if "reconstruction_loss" in hist else "loss"
    records.append({
        "dim": d,
        "recon": round(float(hist[key][-1]), 2),
        "kl": round(float(hist["kl_loss"][-1]), 2) if "kl_loss" in hist else None,
        "total": round(float((hist["total_loss"] if "total_loss" in hist else hist["loss"])[-1]), 2),
        "compression": round(784 / d, 1),
        "params": int(enc.count_params() + dec.count_params())
    })

Image.fromarray(tile(recon_rows)).save(f"{OUT}/latent_dims_recon.png")
Image.fromarray(tile(prior_rows)).save(f"{OUT}/latent_dims_prior.png")
json.dump({"dims": records, "cols": N_SHOW, "tile": 28, "pad": 1,
           "epochs": EPOCHS, "hidden": HIDDEN,
           "note": "recon sheet row 0 is the originals; prior sheet has no originals row"},
          open(f"{OUT}/latent_dims.json", "w"), indent=1)

print(f"\n{'dim':>4} {'compression':>12} {'recon loss':>11} {'KL':>8}")
for r in records:
    kl = f"{r['kl']:8.2f}" if r["kl"] is not None else "       —"
    print(f"{r['dim']:>4} {r['compression']:>11.1f}x {r['recon']:>11.2f} {kl}")

lo, hi = records[0], records[-1]
print(f"\nGoing from {lo['dim']} to {hi['dim']} latent dimensions changes compression from "
      f"{lo['compression']:.0f}x to {hi['compression']:.0f}x and reconstruction loss from "
      f"{lo['recon']:.0f} to {hi['recon']:.0f}.")
print("Reconstruction improves with more dimensions; compression falls. That trade-off is the")
print("point. Note also that only the 2-D model can be drawn as a map — which is why the rest")
print("of the tool uses it.")
print("\nlatent_dims export OK ->", OUT)