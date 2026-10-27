"""T31: bundle the lab VAE decoder for the browser, so the decoder stepper can
decode the exact z the user chose instead of snapping to T23's 12x12 grid.

Paste into VAE.ipynb after the model is trained or restored from
lab/checkpoints (it uses the notebook's `decoder`), like T2/T17/T19/T23.

It refuses to export a decoder that does not reproduce the precomputed grid
in docs/data/layers_vae_decoder.png. Weights from any other training run would
put a different model on the page than the one behind every other VAE figure.
The page runs the same check again on load and stays on the grid if it fails.

After it runs, convert and patch in the tfjs venv (see T25_setup_tfjs_venv.sh):
    tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model \
        lab/vae_decoder/decoder.h5 docs/tfjs/vae_decoder
    python T25_fix_tfjs_topology.py
"""
import os, json, numpy as np
from PIL import Image
from tensorflow import keras

def _find_root():
    d = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "docs")) or os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Could not locate the repo root -- set ROOT manually.")
ROOT = _find_root()

# --- config ------------------------------------------------------------------
DECODER = decoder
TOL     = 0.01                  # mean abs error per pixel, 0..1; DEC_TOL in docs/app.js
# -----------------------------------------------------------------------------

meta  = json.load(open(os.path.join(ROOT, "docs", "data", "layers_vae_decoder.json")))
sheet = np.asarray(Image.open(os.path.join(ROOT, "docs", "data", "layers_vae_decoder.png")),
                   "float32") / 255
G, (LO, HI), TILE = meta["grid"], meta["range"], meta["tile"]

# the stations must line up with the model's layers, in order (names may differ)
layers_ = [l for l in DECODER.layers if not isinstance(l, keras.layers.InputLayer)]
shapes  = [list(l.output.shape[1:]) for l in layers_]
assert shapes == [s["shape"] for s in meta["stations"][1:]], \
    f"decoder layers {shapes} do not match the exported stations"

# same grid and tile rendering as T23
g  = np.linspace(LO, HI, G)
zs = np.array([[a, b] for b in g[::-1] for a in g], "float32")
out = DECODER.predict(zs, verbose=0)
col = len(meta["stations"]) - 1
errs = []
for p, im in enumerate(out):
    im = im.mean(-1)
    lo, hi = float(im.min()), float(im.max())
    im = (im - lo) / (hi - lo) if hi > lo else np.zeros_like(im)
    tile = np.array(Image.fromarray((im * 255).astype("uint8"))
                    .resize((TILE, TILE), Image.NEAREST), "float32") / 255
    errs.append(np.abs(tile - sheet[p*TILE:(p+1)*TILE, col*TILE:(col+1)*TILE]).mean())
err = float(np.mean(errs))
print(f"grid check: mean error {err:.4f} over {len(errs)} points (tolerance {TOL})")
assert err < TOL, ("this decoder is not the one T23 exported -- restore the matching "
                   "weights, or re-run every VAE export from these weights first")

out_dir = os.path.join(ROOT, "lab", "vae_decoder")
os.makedirs(out_dir, exist_ok=True)
DECODER.save(os.path.join(out_dir, "decoder.h5"))
print("saved ->", os.path.join(out_dir, "decoder.h5"))
print("now convert it with tensorflowjs_converter, then run T25_fix_tfjs_topology.py")
