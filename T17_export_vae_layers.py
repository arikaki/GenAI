import os, json, numpy as np
from PIL import Image
from tensorflow import keras

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
TILE = 64                       # each layer is rendered at this size


ENCODER = encoder               # keras functional Model
DECODER = decoder
DATA    = mnist_digits          # (N,28,28,1) in [0,1]
# -----------------------------------------------------------------------------

x = DATA[:1]                    # one representative image

# every layer output except the input placeholder
probe_layers = [l for l in ENCODER.layers if not isinstance(l, keras.layers.InputLayer)]
probe = keras.Model(ENCODER.input, [l.output for l in probe_layers])
acts  = probe.predict(x, verbose=0)
if not isinstance(acts, list): acts = [acts]

def to_tile(a):
    """One activation -> a TILE x TILE uint8 image.
    Spatial layers are shown as the mean across channels; vectors are folded
    into the nearest square block so that a 2-D embedding is still visible."""
    a = np.asarray(a)[0]
    if a.ndim == 3:                       # (H, W, C)
        img = a.mean(-1)
    else:                                 # (D,) dense / embedding
        d = a.reshape(-1); w = int(np.ceil(np.sqrt(d.size)))
        img = np.zeros(w * w, "float32"); img[:d.size] = d; img = img.reshape(w, w)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return np.array(Image.fromarray((img * 255).astype("uint8"))
                    .resize((TILE, TILE), Image.NEAREST))

rows, tiles = [], []

# the input image itself is the first station
# Every station in the encoder is on the main data path — unlike the U-Net,
# whose timestep embedding is a side branch. Marked explicitly so the two JSON
# files share a schema and the front end can treat them the same way.
rows.append({"name": "input", "type": "Image", "shape": list(x.shape[1:]),
             "elements": int(np.prod(x.shape[1:])), "spatial": True,
             "branch": "main"})
tiles.append(to_tile(x))

for layer, a in zip(probe_layers, acts):
    a = np.asarray(a)
    shape = list(a.shape[1:])
    rows.append({
        "name": layer.name,
        "type": layer.__class__.__name__,
        "shape": shape,
        "elements": int(np.prod(shape)),
        "spatial": a.ndim == 4,
        "branch": "main"
    })
    tiles.append(to_tile(a))

# sprite sheet: one row, one tile per station
sheet = np.concatenate(tiles, 1)
Image.fromarray(sheet).save(f"{OUT}/layers_vae.png")

json.dump({"tile": TILE, "model": "vae_encoder", "stations": rows},
          open(f"{OUT}/layers_vae.json", "w"), indent=1)

# --- the finding worth reading before you build the view ---------------------
print(f"{'station':22} {'shape':18} {'elements':>9}")
for r in rows:
    print(f"{r['name']:22} {str(r['shape']):18} {r['elements']:>9}")

inp   = rows[0]["elements"]
final = rows[-1]["elements"]
peak  = max(r["elements"] for r in rows)
print(f"\nInput holds {inp} values. Peak is {peak} ({peak/inp:.1f}x the input).")
print(f"The encoder ends at {final} — a {inp/final:.0f}x reduction from the input.")
print("Element count is NOT monotonic: the first convolution increases it (spatial")
print("size halves, channels grow). Real compression happens at the dense layers,")
print("and unlike the diffusion U-Net the encoder ends at an embedding and stops.")
print("\nlayers_vae export OK ->", os.path.abspath(OUT))