import os, json, numpy as np
from PIL import Image

OUT = "docs/data"; os.makedirs(OUT, exist_ok=True)


ENCODER = encoder          # keras.Model -> [z_mean, z_log_var, z]
DECODER = decoder          # keras.Model: z -> image
DATA    = mnist_digits     # (N,28,28,1) float in [0,1]
LABELS  = None             # set to y_train/np.concatenate([y_train,y_test]) if available
HISTORY = None             # set to the object returned by vae.fit(...) if you kept it
# -------------------------------------------------------------------------------

def u8(a):                              # float [0,1] -> uint8
    return (np.clip(a, 0, 1) * 255).astype("uint8")

def tile(imgs, cols, pad=1, bg=40):     # (n,H,W) uint8 -> one sprite sheet
    n, H, W = imgs.shape
    rows = (n + cols - 1) // cols
    sheet = np.full((rows*(H+pad)-pad, cols*(W+pad)-pad), bg, "uint8")
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        sheet[r*(H+pad):r*(H+pad)+H, c*(W+pad):c*(W+pad)+W] = im
    return sheet

# 1. latent manifold: decode a G x G grid over z in [-3,3]^2  (needs latent_dim == 2)
G, LIM = 30, 3.0
g = np.linspace(-LIM, LIM, G)
zs = np.array([[a, b] for b in g[::-1] for a in g], "float32")   # row 0 = top = max b
imgs = DECODER.predict(zs, verbose=0).squeeze(-1)
Image.fromarray(tile(u8(imgs), cols=G, pad=0)).save(f"{OUT}/vae_manifold.png")
json.dump({"grid": G, "range": [-LIM, LIM], "tile": int(imgs.shape[1])},
          open(f"{OUT}/vae_manifold.json", "w"))

# 2. latent scatter: where real images land
S = 5000
si = np.random.default_rng(0).permutation(len(DATA))[:S]
z_mean = ENCODER.predict(DATA[si], verbose=0)[0]
pts = [[round(float(a), 3), round(float(b), 3)] for a, b in z_mean]
if LABELS is not None:
    pts = [p + [int(LABELS[j])] for p, j in zip(pts, si)]
json.dump({"points": pts, "labelled": LABELS is not None},
          open(f"{OUT}/vae_scatter.json", "w"))

# 3. reconstructions: original / decode(mean) / 3 stochastic draws
R = 16
ri = np.random.default_rng(1).permutation(len(DATA))[:R]
xb = DATA[ri]
z_mean, z_log_var, _ = ENCODER.predict(xb, verbose=0)
rows = [xb.squeeze(-1), DECODER.predict(z_mean, verbose=0).squeeze(-1)]
rng = np.random.default_rng(2)
for _ in range(3):
    z = z_mean + np.exp(0.5*z_log_var) * rng.normal(size=z_mean.shape)
    rows.append(DECODER.predict(z.astype("float32"), verbose=0).squeeze(-1))
grid = u8(np.array([rows[k][j] for k in range(5) for j in range(R)]))
Image.fromarray(tile(grid, cols=R)).save(f"{OUT}/vae_recon.png")
json.dump({"rows": ["original", "decode(mean)", "draw 1", "draw 2", "draw 3"], "cols": R},
          open(f"{OUT}/vae_recon.json", "w"))

# 4. interpolations between encoded pairs
PAIRS, STEPS = 6, 12
pi = np.random.default_rng(3).permutation(len(DATA))[:PAIRS*2]
mu = ENCODER.predict(DATA[pi], verbose=0)[0]
seq = [((1-t)*mu[2*p] + t*mu[2*p+1])
       for p in range(PAIRS) for t in np.linspace(0, 1, STEPS)]
imgs = DECODER.predict(np.array(seq, "float32"), verbose=0).squeeze(-1)
Image.fromarray(tile(u8(imgs), cols=STEPS)).save(f"{OUT}/vae_interp.png")
json.dump({"pairs": PAIRS, "steps": STEPS}, open(f"{OUT}/vae_interp.json", "w"))

# 5. loss history for the ELBO panel (optional)
if HISTORY is not None:
    h = HISTORY.history
    json.dump([{"epoch": i+1,
                "recon": round(float(h["reconstruction_loss"][i]), 3),
                "kl":    round(float(h["kl_loss"][i]), 3),
                "total": round(float(h["loss"][i]), 3)}
               for i in range(len(h["loss"]))],
              open(f"{OUT}/vae_losses.json", "w"))
else:
    print("! vae_losses.json skipped -- set HISTORY = the object returned by vae.fit(...)")
    print("  (re-run fit for a few epochs with history = vae.fit(...) if you did not keep it)")

# self-check: fails loudly if an export is empty or the wrong shape
assert os.path.getsize(f"{OUT}/vae_manifold.png") > 1000, "manifold export looks empty"
assert len(json.load(open(f"{OUT}/vae_scatter.json"))["points"]) == S, "scatter wrong length"
assert latent_dim == 2, "manifold view assumes latent_dim == 2"
print("VAE export OK ->", os.path.abspath(OUT))
print(sorted(os.listdir(OUT)))