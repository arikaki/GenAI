import os, json, numpy as np
from PIL import Image
from tensorflow import keras

def _find_out():
    d = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "docs")) or os.path.isdir(os.path.join(d, ".git")):
            return os.path.join(d, "docs", "data")
        d = os.path.dirname(d)
    raise RuntimeError("Could not locate the repo root — set OUT manually.")
OUT = _find_out(); os.makedirs(OUT, exist_ok=True)

# --- config ------------------------------------------------------------------
DECODER = decoder
GRID    = 12                    # 12x12 = 144 latent positions
LIM     = 3.0                   # same [-3,3] range as the manifold and the map
TILE    = 40
# -----------------------------------------------------------------------------

# grid in the SAME order as vae_manifold: row 0 is the top, z2 descending
g  = np.linspace(-LIM, LIM, GRID)
zs = np.array([[a, b] for b in g[::-1] for a in g], "float32")   # (GRID*GRID, 2)

probe_layers = [l for l in DECODER.layers
                if not isinstance(l, keras.layers.InputLayer)]
probe = keras.Model(DECODER.input, [l.output for l in probe_layers])
acts  = probe.predict(zs, verbose=0)
if not isinstance(acts, list): acts = [acts]

def to_tile(a):
    """One sample's activation -> TILE x TILE uint8, normalised per tile."""
    a = np.asarray(a)
    if a.ndim == 3:                       # (H, W, C)
        img = a.mean(-1)
    else:                                 # (D,) dense
        d = a.reshape(-1); w = int(np.ceil(np.sqrt(d.size)))
        img = np.zeros(w * w, "float32"); img[:d.size] = d; img = img.reshape(w, w)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return np.array(Image.fromarray((img * 255).astype("uint8"))
                    .resize((TILE, TILE), Image.NEAREST))

# station metadata — z itself is the first station
stations = [{"name": "z", "type": "LatentVector", "shape": [zs.shape[1]],
             "elements": int(zs.shape[1]), "spatial": False, "branch": "main"}]
for layer, a in zip(probe_layers, acts):
    shape = list(np.asarray(a).shape[1:])
    stations.append({"name": layer.name, "type": layer.__class__.__name__,
                     "shape": shape, "elements": int(np.prod(shape)),
                     "spatial": np.asarray(a).ndim == 4, "branch": "main"})

# sheet: one row per latent position, one column per station
n_pos, n_st = len(zs), len(stations)
sheet = np.zeros((n_pos * TILE, n_st * TILE), "uint8")
for p in range(n_pos):
    sheet[p*TILE:(p+1)*TILE, 0:TILE] = to_tile(zs[p])
    for s, a in enumerate(acts, start=1):
        sheet[p*TILE:(p+1)*TILE, s*TILE:(s+1)*TILE] = to_tile(np.asarray(a)[p])

Image.fromarray(sheet).save(f"{OUT}/layers_vae_decoder.png")
json.dump({"tile": TILE, "grid": GRID, "range": [-LIM, LIM],
           "model": "vae_decoder", "stations": stations},
          open(f"{OUT}/layers_vae_decoder.json", "w"), indent=1)

print(f"{'station':22} {'shape':18} {'elements':>9}")
for s in stations:
    print(f"{s['name']:22} {str(s['shape']):18} {s['elements']:>9}")
first, last = stations[0]["elements"], stations[-1]["elements"]
print(f"\nThe decoder runs the encoder in reverse: {first} -> {last}, "
      f"a {last/first:.0f}x expansion.")
print(f"Grid: {GRID}x{GRID} = {n_pos} latent positions, {n_st} stations each.")
print(f"Sheet: {sheet.shape[1]}x{sheet.shape[0]} px")
print("layers_vae_decoder export OK ->", OUT)