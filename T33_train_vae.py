"""T33: train the VAE and export everything the page shows of it, in one process.

Replaces running the lab notebook and then pasting T2/T17/T19/T23/T31 into it.
One process means one set of weights, so the figures, the scatter, the decoder
stations, the live decoder and the loss curves all come from the same run, and
the per-epoch history is kept (vae_losses.json was previously from an earlier
run than the figures). Layers are named explicitly, so the names on the page
do not depend on how many models a session happened to build first.

The model is the lab notebook's (the Keras VAE example): latent_dim = 2,
Dense(16) before the latent layer, BCE + analytic KL, trained on MNIST train
and test together.

Writes, under --root (default: this repo):
  docs/data/vae_*.{png,json}, layers_vae.*, layers_vae_decoder.*
  lab/checkpoints/vae_encoder.weights.h5, vae_decoder.weights.h5
  lab/vae_decoder/decoder.h5          for tensorflowjs_converter (T31)

Run:  python T33_train_vae.py              (CPU is fine; a few minutes on GPU)
      python T33_train_vae.py --smoke --root /tmp/smoke
"""
import argparse, os, runpy, sys

import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops

REPO = os.path.dirname(os.path.abspath(__file__))
LATENT_DIM = 2


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


def build():
    i = keras.Input(shape=(28, 28, 1), name="image")
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same", name="conv2d")(i)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", name="conv2d_1")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(16, activation="relu", name="dense")(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling(name="sampling")([z_mean, z_log_var])
    encoder = keras.Model(i, [z_mean, z_log_var, z], name="encoder")

    li = keras.Input(shape=(LATENT_DIM,), name="z")
    y = layers.Dense(7 * 7 * 64, activation="relu", name="dense")(li)
    y = layers.Reshape((7, 7, 64), name="reshape")(y)
    y = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same",
                               name="conv2d_transpose")(y)
    y = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same",
                               name="conv2d_transpose_1")(y)
    y = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same",
                               name="conv2d_transpose_2")(y)
    decoder = keras.Model(li, y, name="decoder")
    return encoder, decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(ops.sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)   # the Keras example's value; the notebook ran 5
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--root", default=REPO, help="where docs/ and lab/ are written")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch: checks the pipeline, not the model")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1

    keras.utils.set_random_seed(0)
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.expand_dims(np.concatenate([x_train, x_test], 0), -1).astype("float32") / 255

    encoder, decoder = build()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    history = vae.fit(mnist_digits, epochs=args.epochs, batch_size=args.batch, verbose=2)

    ckpt = os.path.join(args.root, "lab", "checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    enc_path = os.path.join(ckpt, "vae_encoder.weights.h5")
    dec_path = os.path.join(ckpt, "vae_decoder.weights.h5")
    encoder.save_weights(enc_path)
    decoder.save_weights(dec_path)
    e2, d2 = build()
    e2.load_weights(enc_path); d2.load_weights(dec_path)
    zs = np.array([[0.5, -1.0], [2.0, 0.3]], "float32")
    gap = np.abs(decoder.predict(zs, verbose=0) - d2.predict(zs, verbose=0)).max()
    gap = max(gap, np.abs(encoder.predict(mnist_digits[:4], verbose=0)[0] -
                          e2.predict(mnist_digits[:4], verbose=0)[0]).max())
    assert gap < 1e-5, f"weights did not round-trip (max diff {gap})"
    print("saved ->", ckpt)

    # every export, against these weights. They find docs/ from the cwd.
    g = {"encoder": encoder, "decoder": decoder, "mnist_digits": mnist_digits,
         "latent_dim": LATENT_DIM, "history": history}
    os.makedirs(os.path.join(args.root, "docs", "data"), exist_ok=True)
    cwd = os.getcwd()
    os.chdir(args.root)
    try:
        for cell in ("T2_export_vae_cell.py", "T17_export_vae_layers.py",
                     "T19_export_scatter_labels.py", "T23_export decoder layers.py",
                     "T31_export_vae_decoder_tfjs.py"):
            print(f"\n--- {cell}")
            runpy.run_path(os.path.join(REPO, cell), init_globals=dict(g))
    finally:
        os.chdir(cwd)

    print("\nVAE export OK. Next, in the tfjs venv (T25_setup_tfjs_venv.sh):")
    print("  tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model "
          f"{os.path.join(args.root, 'lab', 'vae_decoder', 'decoder.h5')} docs/tfjs/vae_decoder")
    print("  python T25_fix_tfjs_topology.py")


if __name__ == "__main__":
    main()
