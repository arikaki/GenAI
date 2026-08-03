import os, json, numpy as np
from PIL import Image

OUT = "docs/data"; os.makedirs(OUT, exist_ok=True)


MODEL = unet               # trained U-Net, predicts noise
T     = timesteps          # 200
SEEDS = 3                  # how many reverse trajectories to export
# -------------------------------------------------------------------------------

def u8(a):                              # [-1,1] -> uint8
    return (np.clip((a + 1) / 2, 0, 1) * 255).astype("uint8")

def tile(imgs, cols, pad=0, bg=30):
    n, H, W = imgs.shape
    rows = (n + cols - 1) // cols
    sheet = np.full((rows*(H+pad)-pad, cols*(W+pad)-pad), bg, "uint8")
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        sheet[r*(H+pad):r*(H+pad)+H, c*(W+pad):c*(W+pad)+W] = im
    return sheet

# 1. schedule -- straight from the notebook's own arrays
json.dump({"T": int(T),
           "beta":        [round(float(v), 6) for v in beta],
           "alpha":       [round(float(v), 6) for v in alpha],
           "alpha_bar":   [round(float(v), 6) for v in alpha_bar],
           "sqrt_alpha_bar":       [round(float(v), 4) for v in sqrt_alpha_bar],
           "sqrt_one_minus_abar":  [round(float(v), 4) for v in one_minus_sqrt_alpha_bar]},
          open(f"{OUT}/schedule.json", "w"))

# 2. forward process -- exact, no model involved. One fixed image noised at every t.
x0 = np.expand_dims(sample_mnist, 0) if "sample_mnist" in dir() else next(iter(get_datasets()))[:1]
x0 = np.asarray(x0, "float32")
fwd = [u8(forward_noise(0, x0, np.array([t]))[0].squeeze()) for t in range(T)]
Image.fromarray(tile(np.stack(fwd), cols=T)).save(f"{OUT}/forward.png")

# 3. reverse trajectories + the noise / x0-hat views  (this is the C2 export)
IMG = x0.shape[1]
rev_all, eps_all, x0_all = [], [], []
for s in range(SEEDS):
    np.random.seed(100 + s)
    x = np.random.normal(size=(1, IMG, IMG, 1)).astype("float32")
    rev, epsv, x0v = [], [], []
    for t in reversed(range(T)):
        pred_noise = np.asarray(MODEL(x, np.array([t], "int32")))   # adjust if your call differs
        sab  = np.take(sqrt_alpha_bar, t)
        somb = np.take(one_minus_sqrt_alpha_bar, t)
        x0hat = (x - somb * pred_noise) / sab            # implied clean image
        x = ddpm(x, pred_noise, t) if t > 0 else (x - somb * pred_noise) / sab
        rev.append(u8(np.asarray(x).squeeze()))
        epsv.append(u8(np.asarray(pred_noise).squeeze()))          # noise map
        x0v.append(u8(np.clip(np.asarray(x0hat).squeeze(), -1, 1)))
    rev_all.append(np.stack(rev)); eps_all.append(np.stack(epsv)); x0_all.append(np.stack(x0v))

stack = lambda sheets: np.concatenate([tile(s, cols=T) for s in sheets], 0)
Image.fromarray(stack(rev_all)).save(f"{OUT}/reverse.png")
Image.fromarray(stack(eps_all)).save(f"{OUT}/eps_pred.png")
Image.fromarray(stack(x0_all)).save(f"{OUT}/x0hat.png")
json.dump({"T": int(T), "seeds": SEEDS, "tile": int(IMG),
           "note": "rows = seed, cols = timestep, ordered t = T-1 down to 0"},
          open(f"{OUT}/reverse_meta.json", "w"))

# self-check: the whole point of the C2 view is that x0-hat sharpens as t falls.
early = np.asarray(x0_all[0][10], "float32")     # t near T (very noisy)
late  = np.asarray(x0_all[0][-10], "float32")    # t near 0
assert late.std() > early.std(), "x0-hat should get sharper as t decreases -- check MODEL call signature"
print("diffusion export OK ->", os.path.abspath(OUT))
print(sorted(os.listdir(OUT)))