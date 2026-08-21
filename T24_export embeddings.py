"""T25 Stage 1 export cell.

Retrains three VAE comparison models (latent_dim = 2, 8, 32) and saves everything
Stage 1 (and later Stage 2) needs: the trained models, a PCA fit per dim, a background
scatter, and 20 selectable test-digit picks with their projected positions.

Standalone script (not notebook-cell pasted like T2/T17/T19/T20/T23) because it needs to
run inside the WSL GPU environment, separate from the VAE.ipynb kernel. Run from the repo
root: `python "T24_export embeddings.py"`.
"""
import os, json, numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "docs", "data")
MODELS_OUT = os.path.join(REPO, "lab", "embed_models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS_OUT, exist_ok=True)

# --- config ------------------------------------------------------------------
DIMS       = [2, 8, 32]
EPOCHS     = 8
BATCH      = 128
HIDDEN     = 128        # pre-latent width, matches T20's comparison models
N_PICKS    = 20         # selectable digits, 2 per class
N_SCATTER  = 3000        # background scatter points shown per panel
N_PCA_FIT  = 10000       # sample used to fit PCA (kept separate from the display sample)
SEED       = 7
# -----------------------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
mnist_digits = np.expand_dims(
    np.concatenate([x_train, x_test], axis=0).astype("float32") / 255.0, -1)
labels = np.concatenate([y_train, y_test], axis=0)
n_train = len(x_train)  # picks are drawn from the test portion, index >= n_train


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kw):
        super().__init__(**kw)
        self.encoder, self.decoder = encoder, decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

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


def u8(a):
    return (np.clip(a, 0, 1) * 255).astype("uint8")


def fit_pca(z, k=2):
    """Fit PCA on centered z, return (mean, components[k,D], explained_ratio[k])."""
    d = z.shape[1]
    if d == k:
        # Trivial case: projecting D->D is not a projection at all. Use the identity
        # explicitly (mean=0, components=I) rather than an arbitrary PCA rotation, so
        # dim=2 is a literal identity map -- a free control case where the same point
        # plotted two ways must land in the same place.
        return np.zeros(d, "float32"), np.eye(d, dtype="float32"), np.ones(k, "float32") / k
    mean = z.mean(axis=0)
    centered = z - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    var = (s ** 2) / (len(z) - 1)
    ratio = var[:k] / var.sum()
    return mean.astype("float32"), vt[:k].astype("float32"), ratio.astype("float32")


rng = np.random.default_rng(SEED)
pca_fit_idx = rng.permutation(len(mnist_digits))[:N_PCA_FIT]
scatter_idx = rng.permutation(len(mnist_digits))[:N_SCATTER]

# 2 picks per digit class, drawn from the test portion only
pick_idx = []
for c in range(10):
    cand = np.where(labels[n_train:] == c)[0] + n_train
    pick_idx.extend(rng.choice(cand, size=2, replace=False))
pick_idx = np.array(sorted(pick_idx))

pca_records, scatter_out, picks_out = [], {}, {"picks": []}

for d in DIMS:
    print(f"\n=== latent_dim = {d} ===", flush=True)
    enc, dec = build(d)
    vae = VAE(enc, dec)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(mnist_digits, epochs=EPOCHS, batch_size=BATCH, verbose=2)

    # Mean-only encoder for Stage 2: plain Conv/Dense graph, no custom Sampling layer,
    # so it converts to tfjs cleanly and gives deterministic live-drawing encodes.
    enc_mean = keras.Model(enc.input, enc.output[0], name=f"enc{d}_mean")

    dim_dir = os.path.join(MODELS_OUT, f"dim{d}")
    os.makedirs(dim_dir, exist_ok=True)
    enc_mean.save(os.path.join(dim_dir, "encoder.h5"))
    dec.save(os.path.join(dim_dir, "decoder.h5"))

    zm_fit = enc_mean.predict(mnist_digits[pca_fit_idx], verbose=0, batch_size=512)
    mean, components, explained = fit_pca(zm_fit, k=2)
    compression = round(784 / d, 1)
    pca_records.append({
        "dim": d, "mean": mean.tolist(), "components": components.tolist(),
        "explained": explained.tolist(), "compression": compression,
        "params": int(enc_mean.count_params() + dec.count_params()),
    })

    zm_scatter = enc_mean.predict(mnist_digits[scatter_idx], verbose=0, batch_size=512)
    proj_scatter = (zm_scatter - mean) @ components.T
    scatter_out[str(d)] = [[round(float(a), 3), round(float(b), 3), int(labels[i])]
                            for (a, b), i in zip(proj_scatter, scatter_idx)]

    zm_picks = enc_mean.predict(mnist_digits[pick_idx], verbose=0, batch_size=512)
    proj_picks = (zm_picks - mean) @ components.T
    for k_i, i in enumerate(pick_idx):
        if d == DIMS[0]:
            picks_out["picks"].append({"index": int(i), "label": int(labels[i]), "proj": {}})
        picks_out["picks"][k_i]["proj"][str(d)] = [round(float(proj_picks[k_i][0]), 3),
                                                     round(float(proj_picks[k_i][1]), 3)]

json.dump({"dims": pca_records}, open(os.path.join(OUT, "embed_pca.json"), "w"), indent=1)
json.dump(scatter_out, open(os.path.join(OUT, "embed_scatter.json"), "w"))
json.dump(picks_out, open(os.path.join(OUT, "embed_picks.json"), "w"), indent=1)

# sprite sheet of the 20 picks, one row, 28px tiles
tiles = mnist_digits[pick_idx].squeeze(-1)
pad, bg = 1, 40
sheet = np.full((28, len(tiles) * (28 + pad) - pad), bg, "uint8")
for c, t in enumerate(tiles):
    sheet[:, c * (28 + pad):c * (28 + pad) + 28] = u8(t)
Image.fromarray(sheet).save(os.path.join(OUT, "embed_picks.png"))

print(f"\n{'dim':>4} {'compression':>12} {'explained (2 PCs)':>18}")
for r in pca_records:
    print(f"{r['dim']:>4} {r['compression']:>11.1f}x {sum(r['explained']):>17.1%}")
print("\nembed_pca / embed_scatter / embed_picks export OK ->", OUT)
print("embed_models export OK ->", MODELS_OUT)
