"""T25 Stage 2: patch tensorflowjs_converter output for Keras-3 topology.

tensorflowjs_converter (pip, latest as of writing: 4.22.0) was built against Keras 2's
functional-model serialization format. TensorFlow 2.21 bundles Keras 3, which changed that
format in three ways the converter's JS-side deserializer does not understand:

  1. InputLayer: `batch_input_shape` renamed to `batch_shape`.
  2. Every layer's `inbound_nodes`: Keras 2 used a plain array of
     [layer_name, node_index, tensor_index, kwargs] tuples. Keras 3 uses a list of
     {"args": [...], "kwargs": {...}} dicts, with connectivity buried in each arg's
     `keras_history`.
  3. Functional model `input_layers` / `output_layers`: Keras 2 wrapped these in an outer
     list (supporting multi-input/output); Keras 3 emits a bare tuple for the single-io case.

The .h5 weights themselves are fine — this is a topology-JSON-only mismatch. Rather than
retrain against the legacy `tf_keras` package (which would avoid this by emitting the old
format directly), this rewrites tensorflowjs_converter's output back to the old format it
expects. Run after every `tensorflowjs_converter` call, before the model is used from JS.

Verified against a live tf.loadLayersModel() + predict() call, output compared to the
original Keras model's prediction on the same input.
"""
import glob, json, os, sys

REPO = os.path.dirname(os.path.abspath(__file__))


def fix_inbound_nodes(nodes):
    """Keras-3 dict-of-args nodes -> Keras-2 array-of-tuples nodes."""
    if not nodes or isinstance(nodes[0], list):
        return nodes  # already old format (e.g. InputLayer's empty [])
    fixed = []
    for node in nodes:
        tuples = []
        for arg in node.get("args", []):
            if isinstance(arg, dict) and "config" in arg and "keras_history" in arg["config"]:
                hist = arg["config"]["keras_history"]
                layer_name, node_idx, tensor_idx = hist[0], hist[1], hist[2]
                tuples.append([layer_name, node_idx, tensor_idx, node.get("kwargs", {}) or {}])
        fixed.append(tuples)
    return fixed


def fix_dtype(cfg):
    dt = cfg.get("dtype")
    if isinstance(dt, dict):
        cfg["dtype"] = dt.get("config", {}).get("name", "float32")


def fix_layer(layer):
    cfg = layer.get("config", {})
    if layer.get("class_name") == "InputLayer" and "batch_shape" in cfg:
        cfg["batch_input_shape"] = cfg.pop("batch_shape")
    fix_dtype(cfg)
    if "inbound_nodes" in layer:
        layer["inbound_nodes"] = fix_inbound_nodes(layer["inbound_nodes"])


def fix_model_json(path):
    m = json.load(open(path))
    cfg = m["modelTopology"]["model_config"]["config"]
    for layer in cfg.get("layers", []):
        fix_layer(layer)
    for key in ("input_layers", "output_layers"):
        v = cfg.get(key)
        if v and isinstance(v[0], str):   # bare tuple -> wrap in outer list
            cfg[key] = [v]
    json.dump(m, open(path, "w"))
    return path


if __name__ == "__main__":
    paths = glob.glob(os.path.join(REPO, "docs", "tfjs", "dim*", "*", "model.json"))
    for p in paths:
        fix_model_json(p)
        print("fixed", os.path.relpath(p, REPO))
    print(f"\n{len(paths)} model.json files patched.")
