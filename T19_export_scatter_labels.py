import os, json, numpy as np
from tensorflow import keras

def _find_out():
    d = os.path.abspath(os.getcwd())
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "docs")) or os.path.isdir(os.path.join(d, ".git")):
            return os.path.join(d, "docs", "data")
        d = os.path.dirname(d)
    raise RuntimeError("Could not locate the repo root — set OUT manually.")
OUT = _find_out()


(_, y_train), (_, y_test) = keras.datasets.mnist.load_data()
LABELS = np.concatenate([y_train, y_test], axis=0)

ENCODER = encoder
DATA    = mnist_digits
# -----------------------------------------------------------------------------

assert len(LABELS) == len(DATA), (
    f"labels ({len(LABELS)}) and data ({len(DATA)}) differ in length — check that "
    "mnist_digits was built from train+test in that order")

S  = 5000
si = np.random.default_rng(0).permutation(len(DATA))[:S]
z_mean = ENCODER.predict(DATA[si], verbose=0)[0]

points = [[round(float(a), 3), round(float(b), 3), int(LABELS[j])]
          for (a, b), j in zip(z_mean, si)]

json.dump({"points": points, "labelled": True},
          open(f"{OUT}/vae_scatter.json", "w"))

# sanity: do the classes actually separate, or is the map uniform?
import collections
cent = collections.defaultdict(list)
for a, b, c in points: cent[c].append((a, b))
print("class centroids in latent space:")
for c in sorted(cent):
    pts = np.array(cent[c])
    print(f"  {c}: ({pts[:,0].mean():+.2f}, {pts[:,1].mean():+.2f})   n={len(pts)}")
spread = np.array([[np.mean([p[0] for p in cent[c]]), np.mean([p[1] for p in cent[c]])]
                   for c in sorted(cent)])
print(f"\ncentroid spread: {spread.std(0).round(2)}  "
      "(near zero would mean the classes do not separate)")
print("scatter re-exported ->", OUT)