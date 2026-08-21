#!/usr/bin/env bash
# Builds .venv_tfjs — a Windows-native venv used ONLY for tensorflowjs_converter
# (CPU-only, no GPU needed for conversion). Model training happens elsewhere (WSL2 GPU).
#
# Why this isn't a plain `pip install tensorflowjs`: latest tensorflowjs (4.x) pulls in
# tensorflow-decision-forests, jax, and flax as hard unconditional imports, none of which
# our conversion (plain Keras -> tfjs_layers_model) ever calls. On Windows/Python 3.12:
#   - tensorflow-decision-forests 1.8.1 ships no compatible native op (inference.so) for
#     tensorflow>=2.16 on this platform -> ImportError at module load.
#   - flax pulls in orbax, whose test fixtures have paths that exceed Windows MAX_PATH
#     unless long-path support is enabled -> install fails outright.
#   - jax installs but an old resolved version (0.11.1) hits a circular-import bug on
#     import.
# None of these are needed for our conversion path, so each is replaced with a minimal
# stub package that satisfies `import` without providing real functionality.
#
# Verified end-to-end: trained a tiny Keras model, model.save('x.h5'), converted with
# `tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model`,
# inspected model.json — full topology + weights round-tripped correctly.
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv_tfjs
source .venv_tfjs/Scripts/activate
python -m pip install --upgrade pip -q

# Core: tensorflow pulled in automatically (>=2.13,<3 — resolves to latest CPU wheel).
pip install tensorflowjs --no-deps -q
pip install six packaging importlib_resources tensorflow-hub -q
pip install "setuptools<81" -q   # tensorflow_hub still imports the deprecated pkg_resources API

# tensorflow-decision-forests: stub out. Only ever referenced via its bare `import`
# statement (tf_saved_model_conversion_v2.py) — never called.
pip install tensorflow-decision-forests --no-deps -q || true
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "$SITE/tensorflow_decision_forests"
mkdir -p "$SITE/tensorflow_decision_forests"
: > "$SITE/tensorflow_decision_forests/__init__.py"

# jax: stub out. keras 3's orbax_checkpoint callback probes `jax.monitoring` at import
# time (unrelated to tensorflowjs), and jax_conversion.py imports jax2tf unconditionally
# — neither is exercised by a Keras -> tfjs_layers_model conversion.
rm -rf "$SITE/jax"
mkdir -p "$SITE/jax/experimental/jax2tf"
printf 'from . import monitoring\n' > "$SITE/jax/__init__.py"
: > "$SITE/jax/monitoring.py"
: > "$SITE/jax/experimental/__init__.py"
printf 'def convert(*a, **kw):\n    raise NotImplementedError("jax2tf stub - not needed for Keras model conversion")\n' \
  > "$SITE/jax/experimental/jax2tf/__init__.py"

python -c "import tensorflowjs as tfjs; print('tensorflowjs', tfjs.__version__, 'OK')"
echo "Setup OK. Activate with: source .venv_tfjs/Scripts/activate"
