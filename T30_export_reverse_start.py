"""T30: export x_T, the starting noise of the three reverse runs.

reverse.png stores the state *after* each denoising step, so its first column is
x_198, not the noise the run started from. The page needs x_T itself for the
starting-noise card and for the x_t tile of the "all three at once" strip.

No model is needed: T2_export_diffusion_cell.py drew x_T as
    np.random.seed(100 + s); np.random.normal(size=(1, 32, 32, 1))
and NumPy's legacy seeded generator is stable across versions, so the same
draws can be rebuilt here exactly. As a check, each rebuilt x_T must correlate
strongly with its own run's x_198 (one step of sigma = sqrt(0.02) noise apart)
and not with the other runs'.

Run from the repo root: `python T30_export_reverse_start.py`.
"""
import os, json, numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "docs", "data")
meta = json.load(open(os.path.join(OUT, "reverse_meta.json")))
SEEDS, IMG = meta["seeds"], meta["tile"]

def u8(a):                              # [-1,1] -> uint8, as in T2
    return (np.clip((a + 1) / 2, 0, 1) * 255).astype("uint8")

rev = np.asarray(Image.open(os.path.join(OUT, "reverse.png")), "float64")
first = [rev[s*IMG:(s+1)*IMG, :IMG].ravel() for s in range(SEEDS)]

tiles = []
for s in range(SEEDS):
    np.random.seed(100 + s)
    x = np.random.normal(size=(1, IMG, IMG, 1)).astype("float32")
    t = u8(x.squeeze())
    own = np.corrcoef(t.ravel(), first[s])[0, 1]
    others = max(abs(np.corrcoef(t.ravel(), first[o])[0, 1]) for o in range(SEEDS) if o != s)
    print(f"seed {s}: corr with own x_198 {own:.3f}, best other {others:.3f}")
    assert own > 0.95 and others < 0.2, "rebuilt x_T does not match the exported run"
    tiles.append(t)

# one column, row = seed, the same layout as reverse.png
Image.fromarray(np.concatenate(tiles, 0)).save(os.path.join(OUT, "reverse_start.png"))
print("reverse_start export OK ->", OUT)
