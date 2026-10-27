"""T32: train the diffusion model and export everything the page shows of it.

Replaces the lab notebook's training cells and T2_export_diffusion_cell.py. The
U-Net is the notebook's own, unchanged in architecture; the schedule, forward
process and sampler come from diffusion_schedule.py, which fixes the notebook's
schedule (it never reached pure noise) and its off-by-one reverse step. See the
docstring there, and tests/test_diffusion_schedule.py.

Writes, under --root (default: this repo):
  docs/data/schedule.json            the schedule, at full precision
  docs/data/diffusion_losses.json    mean training loss per epoch
  docs/data/forward.png              one image noised at every t
  docs/data/reverse.png, eps_pred.png, x0hat.png, reverse_meta.json, reverse_start.png
  docs/data/unet_check.json          known inputs and outputs, for the browser's self-check
  docs/data/layers_diffusion.*       via T17_export_diffusion_layers.py
  lab/diffusion/unet.weights.h5      the trained weights
  lab/diffusion/unet_savedmodel/     for tensorflowjs_converter (see the end of this file)

Run from anywhere:  python T32_train_diffusion.py            (GPU recommended)
Quick end-to-end check on CPU, written to a scratch root:
                    python T32_train_diffusion.py --smoke --root /tmp/smoke
"""
import argparse, json, math, os, runpy, sys, time
from functools import partial
from inspect import isfunction

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras, einsum
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as nn
from einops import rearrange

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import diffusion_schedule as ds

# --- the lab notebook's U-Net ---------------------------------------------------
# Unchanged in what it computes. LayerNorm registers its parameters with
# add_weight (a raw tf.Variable is not saved by Keras 3's save_weights);
# Attention subtracts each row's max rather than its argmax before the softmax
# (softmax ignores any per-row constant, and max is what was intended); GELU is
# rearranged for the browser (see below). The notebook also kept its
# checkpoint with tf.train.Checkpoint, which does not track layers held in
# nested lists: its ckpt-3 held 10.4M of the 32.3M parameters, missing every
# down and up block. save_weights is used here, and checked by a cold reload.

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class SinusoidalPosEmb(Layer):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        return tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

class Identity(Layer):
    def call(self, x, training=True):
        return tf.identity(x)

class Residual(Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x, training=True):
        return self.fn(x, training=training) + x

def Upsample(dim):
    return nn.Conv2DTranspose(filters=dim, kernel_size=4, strides=2, padding='SAME')

def Downsample(dim):
    return nn.Conv2D(filters=dim, kernel_size=4, strides=2, padding='SAME')

class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.g = self.add_weight(shape=(1, 1, 1, dim), initializer="ones", name="g")
        self.b = self.add_weight(shape=(1, 1, 1, dim), initializer="zeros", name="b")

    def call(self, x, training=True):
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        return (x - mean) / tf.sqrt((var + self.eps)) * self.g + self.b

class PreNorm(Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def call(self, x, training=True):
        return self.fn(self.norm(x))

class SiLU(Layer):
    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)

class GELU(Layer):
    # Exact GELU, written as x * (0.5 + 0.5 erf(x / sqrt 2)). The textbook
    # 0.5 x (1 + erf(...)) is the same function, but tensorflowjs_converter fuses
    # that pattern into a MatMul op TF.js's WebGL backend cannot run.
    def call(self, x, training=True):
        return x * (0.5 + 0.5 * tf.math.erf(x * 0.7071067811865476))

class Block(Layer):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.proj = nn.Conv2D(dim, kernel_size=3, strides=1, padding='SAME')
        self.norm = nn.GroupNormalization(groups, epsilon=1e-05)
        self.act = SiLU()

    def call(self, x, gamma_beta=None, training=True):
        x = self.proj(x)
        x = self.norm(x, training=training)
        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1) + beta
        return self.act(x)

class ResnetBlock(Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = Sequential([SiLU(), nn.Dense(units=dim_out * 2)]) if exists(time_emb_dim) else None
        self.block1 = Block(dim_out, groups=groups)
        self.block2 = Block(dim_out, groups=groups)
        self.res_conv = nn.Conv2D(filters=dim_out, kernel_size=1, strides=1) if dim != dim_out else Identity()

    def call(self, x, time_emb=None, training=True):
        gamma_beta = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 1 c')
            gamma_beta = tf.split(time_emb, num_or_size_splits=2, axis=-1)
        h = self.block1(x, gamma_beta=gamma_beta, training=training)
        h = self.block2(h, training=training)
        return h + self.res_conv(x)

class LinearAttention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = Sequential([nn.Conv2D(filters=dim, kernel_size=1, strides=1), LayerNorm(dim)])

    def call(self, x, training=True):
        b, h, w, c = x.shape
        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)
        q = tf.nn.softmax(q, axis=-2)
        k = tf.nn.softmax(k, axis=-1)
        q = q * self.scale
        context = einsum('b h d n, b h e n -> b h d e', k, v)
        out = einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b x y (h c)', h=self.heads, x=h, y=w)
        return self.to_out(out, training=training)

class Attention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = nn.Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        b, h, w, c = x.shape
        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - tf.stop_gradient(tf.reduce_max(sim, axis=-1, keepdims=True))
        attn = tf.nn.softmax(sim, axis=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=h, y=w)
        return self.to_out(out, training=training)

class Unet(Model):
    def __init__(self, dim=64, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8),
                 channels=3, resnet_block_groups=8, learned_variance=False,
                 sinusoidal_cond_mlp=True):
        super().__init__()
        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='SAME')
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4
        self.sinusoidal_cond_mlp = sinusoidal_cond_mlp
        self.time_mlp = Sequential([SinusoidalPosEmb(dim), nn.Dense(units=time_dim), GELU(),
                                    nn.Dense(units=time_dim)], name="time_embeddings")
        self.downs, self.ups = [], []
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else Identity()])
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity()])
        self.out_dim = default(out_dim, channels * (1 if not learned_variance else 2))
        self.final_conv = Sequential([block_klass(dim * 2, dim),
                                      nn.Conv2D(filters=self.out_dim, kernel_size=1, strides=1)],
                                     name="output")

    def call(self, x, time=None, training=True, **kwargs):
        x = self.init_conv(x)
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = tf.concat([x, h.pop()], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = tf.concat([x, h.pop()], axis=-1)
        return self.final_conv(x)

# --- helpers --------------------------------------------------------------------

def u8(a):                              # [-1,1] -> uint8
    return (np.clip((np.asarray(a) + 1) / 2, 0, 1) * 255).astype("uint8")

def tile(imgs, cols, pad=0, bg=30):
    n, H, W = imgs.shape
    rows = (n + cols - 1) // cols
    sheet = np.full((rows*(H+pad)-pad, cols*(W+pad)-pad), bg, "uint8")
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        sheet[r*(H+pad):r*(H+pad)+H, c*(W+pad):c*(W+pad)+W] = im
    return sheet

def load_mnist(n=None):
    """The notebook's preprocessing: train split, [-1, 1], resized to 32x32."""
    (x, _), _ = keras.datasets.mnist.load_data()
    x = x[:n] if n else x
    x = tf.image.resize(tf.cast(x[..., None], tf.float32) / 127.5 - 1, (32, 32))
    return x.numpy()

# --- main -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)   # the tutorial's value; the notebook ran 3
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--root", default=REPO, help="where docs/ and lab/ are written")
    ap.add_argument("--smoke", action="store_true",
                    help="512 images, 1 epoch: checks the pipeline, not the model")
    ap.add_argument("--skip-train", action="store_true",
                    help="load lab/diffusion/unet.weights.h5 and only re-run the exports")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1

    OUT = os.path.join(args.root, "docs", "data")
    LAB = os.path.join(args.root, "lab", "diffusion")
    os.makedirs(OUT, exist_ok=True); os.makedirs(LAB, exist_ok=True)

    s = ds.make_schedule()
    T = s["T"]
    SAB = tf.constant(s["sqrt_alpha_bar"], tf.float32)
    S1M = tf.constant(s["sqrt_one_minus_abar"], tf.float32)

    data = load_mnist(512 if args.smoke else None)
    train = (tf.data.Dataset.from_tensor_slices(data)
             .shuffle(len(data), reshuffle_each_iteration=True)
             .batch(args.batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE))

    unet = Unet(channels=1)
    unet(tf.zeros([1, 32, 32, 1]), tf.zeros([1], tf.int32))
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    print(f"U-Net parameters: {unet.count_params():,}")

    @tf.function
    def train_step(x0):
        n = tf.shape(x0)[0]
        t = tf.random.uniform([n], 0, T, tf.int32)
        eps = tf.random.normal(tf.shape(x0))
        x_t = tf.gather(SAB, t)[:, None, None, None] * x0 + tf.gather(S1M, t)[:, None, None, None] * eps
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((eps - unet(x_t, t, training=True)) ** 2)
        opt.apply_gradients(zip(tape.gradient(loss, unet.trainable_variables),
                                unet.trainable_variables))
        return loss

    wpath = os.path.join(LAB, "unet.weights.h5")
    history = []
    for e in range(1, 0 if args.skip_train else args.epochs + 1):
        t0, losses = time.time(), []
        for i, batch in enumerate(train):
            losses.append(float(train_step(batch)))
            if i % 100 == 0:
                print(f"  epoch {e} step {i}: loss {np.mean(losses):.4f}", flush=True)
        history.append({"epoch": e, "loss": round(float(np.mean(losses)), 5)})
        print(f"epoch {e}/{args.epochs}: loss {history[-1]['loss']}  ({time.time() - t0:.0f}s)", flush=True)
    if args.skip_train:
        unet.load_weights(wpath)
        print("loaded <-", wpath)
    else:
        json.dump(history, open(os.path.join(OUT, "diffusion_losses.json"), "w"))
        unet.save_weights(wpath)

    # a cold reload, to prove the weights round-trip
    fresh = Unet(channels=1)
    fresh(tf.zeros([1, 32, 32, 1]), tf.zeros([1], tf.int32))
    fresh.load_weights(wpath)
    probe = np.random.default_rng(0).normal(size=(2, 32, 32, 1)).astype("float32")
    pt = np.array([5, 150], "int32")
    gap = np.abs(np.asarray(unet(probe, pt, training=False)) - np.asarray(fresh(probe, pt, training=False))).max()
    assert gap < 1e-5, f"weights did not round-trip (max diff {gap})"
    print("saved ->", wpath)

    predict = lambda x, t: np.asarray(unet(x.astype("float32"), np.array([t], "int32"), training=False),
                                      "float64")

    # schedule, at full precision (the browser's sampler reads it)
    json.dump({"T": T, **{k: [float(v) for v in s[k]] for k in
               ("beta", "alpha", "alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_abar")}},
              open(os.path.join(OUT, "schedule.json"), "w"))

    # forward process: exact, no model involved
    x0 = data[:1].astype("float64")
    eps0 = np.random.default_rng(0).standard_normal(x0.shape)
    fwd = [u8(ds.forward_noise(s, x0, np.array([t]), eps0)[0, ..., 0]) for t in range(T)]
    Image.fromarray(tile(np.stack(fwd), cols=T)).save(os.path.join(OUT, "forward.png"))

    # reverse runs. Column k holds the state after step k, as before; x_T itself
    # goes to reverse_start.png.
    rev_all, eps_all, x0_all, starts, checks = [], [], [], [], []
    for sd in range(args.seeds):
        rng = np.random.default_rng(100 + sd)
        x = rng.standard_normal((1, 32, 32, 1))
        starts.append(u8(x[0, ..., 0]))
        rev, epsv, x0v = [], [], []
        for t in reversed(range(T)):
            e_hat = predict(x, t)
            if sd == 0 and t in (T - 1, T // 2, 10):
                checks.append({"t": t, "x": np.round(x.ravel(), 5).tolist(),
                               "eps": np.round(e_hat.ravel(), 5).tolist()})
            x0v.append(u8(np.clip(ds.predict_x0(s, x, np.array([t]), e_hat)[0, ..., 0], -1, 1)))
            epsv.append(u8(e_hat[0, ..., 0]))
            x = ds.reverse_step(s, x, t, e_hat, rng)
            rev.append(u8(x[0, ..., 0]))
        rev_all.append(np.stack(rev)); eps_all.append(np.stack(epsv)); x0_all.append(np.stack(x0v))

    stack = lambda sheets: np.concatenate([tile(sh, cols=T) for sh in sheets], 0)
    Image.fromarray(stack(rev_all)).save(os.path.join(OUT, "reverse.png"))
    Image.fromarray(stack(eps_all)).save(os.path.join(OUT, "eps_pred.png"))
    Image.fromarray(stack(x0_all)).save(os.path.join(OUT, "x0hat.png"))
    Image.fromarray(np.concatenate(starts, 0)).save(os.path.join(OUT, "reverse_start.png"))
    json.dump({"T": T, "seeds": args.seeds, "tile": 32,
               "note": "rows = seed, cols = timestep, ordered t = T-1 down to 0; "
                       "each column is the state after that step"},
              open(os.path.join(OUT, "reverse_meta.json"), "w"))
    json.dump({"checks": checks}, open(os.path.join(OUT, "unet_check.json"), "w"))

    # layer activations, by the existing export (it finds docs/ from the cwd)
    cwd = os.getcwd()
    os.chdir(args.root)
    try:
        runpy.run_path(os.path.join(REPO, "T17_export_diffusion_layers.py"),
                       init_globals={"unet": unet, "timesteps": T})
    finally:
        os.chdir(cwd)

    # a SavedModel with one fixed signature, for tensorflowjs_converter
    # Only the variables are tracked, not the Keras model: saving the model object
    # makes Keras trace its own default signature, which cannot pass the timestep.
    sm = os.path.join(LAB, "unet_savedmodel")
    mod = tf.Module()
    mod.vars = list(unet.variables)
    mod.serve = tf.function(lambda x, t: {"eps": unet(x, t, training=False)},
                            input_signature=[tf.TensorSpec([1, 32, 32, 1], tf.float32, name="x"),
                                             tf.TensorSpec([1], tf.int32, name="t")])
    tf.saved_model.save(mod, sm, signatures={"serving_default": mod.serve})
    reloaded = tf.saved_model.load(sm).signatures["serving_default"]
    eager_out = np.asarray(unet(probe[:1], pt[:1], training=False))
    sm_out = reloaded(x=tf.constant(probe[:1]), t=tf.constant(pt[:1]))["eps"].numpy()
    gap = np.abs(sm_out - eager_out).max()
    # A fixed absolute tolerance doesn't scale to this depth: eager and the traced
    # signature run the SAME weights through different op orderings/fusions (the
    # oneDNN/XLA kernels TF itself warns about), and across 39 layers of conv,
    # GroupNorm and attention that alone measured ~0.1-0.3% of the output's peak
    # magnitude here. A tracked-variable bug (the one this check exists to catch --
    # the notebook's checkpoint silently dropping every down/up block) produces a
    # gap orders of magnitude larger than that, so judge relative to scale instead.
    rel = gap / (np.abs(eager_out).max() + 1e-8)
    assert rel < 1e-2, (f"SavedModel does not reproduce the model (max diff {gap}, "
                        f"{rel:.1%} of eager's peak magnitude -- expected well under 1%)")
    print("saved ->", sm)

    if not args.smoke:
        # Raw pixel std is not a sharpness proxy here: MNIST digits are mostly
        # background with a thin stroke, so a correct, clean x0-hat can have LOWER
        # std than an early, salt-and-pepper-noisy one (extreme bimodal values
        # spread over the whole tile maximise variance). What "sharpens" actually
        # means is that x0-hat converges toward its final estimate as t falls, so
        # measure that directly: distance to the last (most-denoised) frame.
        final = x0_all[0][-1].astype(float)
        early, late = x0_all[0][10].astype(float), x0_all[0][-10].astype(float)
        mse_early = ((early - final) ** 2).mean()
        mse_late = ((late - final) ** 2).mean()
        assert mse_late < mse_early, (
            f"x0-hat should converge toward its final estimate as t falls "
            f"(MSE to final: early {mse_early:.1f}, late {mse_late:.1f})")

    print("\ndiffusion export OK ->", OUT)
    print("Next, in the tfjs venv (T25_setup_tfjs_venv.sh):")
    print(f"  tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model "
          f"--signature_name=serving_default --quantize_float16='*' {sm} docs/tfjs/unet")


if __name__ == "__main__":
    main()
