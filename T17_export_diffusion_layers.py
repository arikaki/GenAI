import os, json, numpy as np
from PIL import Image

OUT = None
def _find_out():
    """Notebooks run with cwd=lab/, so a relative "docs/data" lands in the wrong
    place. Walk up until the repo root (the folder holding docs/ or .git) is found."""
    d = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "docs")) or os.path.isdir(os.path.join(d, ".git")):
            return os.path.join(d, "docs", "data")
        d = os.path.dirname(d)
    raise RuntimeError("Could not locate the repo root — set OUT manually.")
OUT = _find_out(); os.makedirs(OUT, exist_ok=True)
TILE = 64


MODEL     = unet                # trained U-Net
T_STEPS   = timesteps           # 200
PROBE_T   = 100                 # timestep to probe at (mid-trajectory)
IMG       = 32
SKIP_TYPES = ("Activation", "Dropout")   # not interesting as stations
# -----------------------------------------------------------------------------

def walk(layer, depth=0, seen=None):
    """Depth-first list of leaf layers — the U-Net nests convs inside blocks."""
    seen = seen if seen is not None else set()
    if id(layer) in seen: return []
    seen.add(id(layer))
    kids = getattr(layer, "layers", None) or []
    if not kids:
        return [(layer, depth)]
    out = []
    for k in kids:
        out += walk(k, depth + 1, seen)
    return out

leaves = [(l, d) for l, d in walk(MODEL)
          if l.__class__.__name__ not in SKIP_TYPES]

records, originals = [], {}
try:
    for l, d in leaves:
        originals[id(l)] = l.call
        def make(layer, orig, depth):
            def wrapped(*a, **kw):
                out = orig(*a, **kw)
                try:
                    arr = np.asarray(out)
                    if arr.ndim >= 2:
                        records.append((layer.name, layer.__class__.__name__, depth, arr))
                except Exception:
                    pass
                return out
            return wrapped
        l.call = make(l, originals[id(l)], d)

    x = np.random.normal(size=(1, IMG, IMG, 1)).astype("float32")
    _ = MODEL(x, np.array([PROBE_T], "int32"))     # match your training call
finally:
    for l, _d in leaves:
        if id(l) in originals: l.call = originals[id(l)]

if not records:
    raise RuntimeError("No activations captured — check the MODEL(...) call signature above.")

def to_tile(a):
    a = np.asarray(a)
    a = a[0] if a.ndim == 4 else a
    if a.ndim == 3:
        img = a.mean(-1)
    else:
        d = a.reshape(-1); w = int(np.ceil(np.sqrt(d.size)))
        img = np.zeros(w * w, "float32"); img[:d.size] = d; img = img.reshape(w, w)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return np.array(Image.fromarray((img * 255).astype("uint8"))
                    .resize((TILE, TILE), Image.NEAREST))

rows, tiles = [], []
rows.append({"name": "input x_t", "type": "Input", "shape": [IMG, IMG, 1],
             "elements": IMG * IMG, "spatial": True, "depth": 0})
tiles.append(to_tile(x))

for name, cls, depth, arr in records:
    shape = list(arr.shape[1:])
    rows.append({"name": name, "type": cls, "shape": shape,
                 "elements": int(np.prod(shape)), "spatial": arr.ndim == 4,
                 "depth": depth})
    tiles.append(to_tile(arr))

# Mark the stations where the spatial size actually changes. 39 stations is too
# many to read; these ~8 trace the hourglass and carry the whole argument.
#
# Also separate the timestep-embedding branch. sinusoidal_pos_emb / dense / gelu
# feed t into the network alongside the image — the image never passes through
# them. Left unmarked they appear inline in the sequence, implying a flow that
# does not exist, and their small size (64) would be mistaken for a bottleneck.
prev = None
for i, r in enumerate(rows):
    r["branch"] = "main" if r["spatial"] else "time"
    side = r["shape"][0] if r["spatial"] else None
    r["key"] = bool(side is not None and (prev is None or side != prev))
    if side is not None: prev = side
rows[-1]["key"] = True                       # always show the output

Image.fromarray(np.concatenate(tiles, 1)).save(f"{OUT}/layers_diffusion.png")
json.dump({"tile": TILE, "model": "diffusion_unet", "probe_t": PROBE_T,
           "stations": rows},
          open(f"{OUT}/layers_diffusion.json", "w"), indent=1)

print(f"{'station':28} {'shape':18} {'elements':>9}")
for r in rows:
    print(f"{r['name']:28} {str(r['shape']):18} {r['elements']:>9}")

spatial = [r["shape"][0] for r in rows if r["spatial"]]
inp = IMG * IMG
# The bottleneck is an INTERIOR property. The input and the final output are both
# 32x32x1 by construction — an epsilon-predicting U-Net must return to input shape —
# so including them makes min() trivially equal to the input size and measures nothing.
interior = [r["elements"] for r in rows[1:-1] if r["spatial"]]
tightest = min(interior)
narrowest = next(r for r in rows[1:-1] if r.get("spatial") and r["elements"] == tightest)

print(f"\nSpatial size: starts {spatial[0]}, minimum {min(spatial)}, ends {spatial[-1]}")
print(f"Key stations marked: {sum(r['key'] for r in rows)} of {len(rows)}")
print(f"\nInput holds {inp} values.")
print(f"Narrowest interior layer: {narrowest['name']} {narrowest['shape']} = "
      f"{tightest} values ({tightest/inp:.1f}x the input).")
print("The U-Net reduces spatial size and then restores it. It never compresses")
print("below the input, and never produces an embedding vector — that is the")
print("contrast with the VAE encoder, which ends at 2 numbers.")
print("\nlayers_diffusion export OK ->", os.path.abspath(OUT))