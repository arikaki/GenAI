#!/usr/bin/env bash
# Sets up a WSL2 (Ubuntu) venv for GPU-accelerated training, used for T25 Stage 1
# retraining. Run this *inside* WSL: `wsl -d Ubuntu-24.04 -- bash T25_setup_wsl_gpu.sh`
#
# Why WSL and not native Windows: pip tensorflow dropped native-Windows GPU support
# after 2.10 (Windows GPU users must go through WSL2 or DirectML). WSL2 was already
# installed on this machine with an RTX 4090 Laptop GPU passthrough-visible via nvidia-smi.
#
# The nontrivial part: `pip install tensorflow[and-cuda]` installs the right CUDA/cuDNN
# pip wheels (nvidia-cusolver-cu12 etc.) but TF's own dlopen search does not reliably
# find them, nor WSL's own libcuda.so.1 (at /usr/lib/wsl/lib) -- both need to be on
# LD_LIBRARY_PATH explicitly. Diagnosed via TF_CPP_VMODULE=dso_loader=3, which prints
# per-library dlopen results instead of TF's generic "Cannot dlopen some GPU libraries".
set -euo pipefail
python3 -m venv ~/genai-tfgpu
source ~/genai-tfgpu/bin/activate
pip install --upgrade pip -q
pip install "tensorflow[and-cuda]" pillow -q

SITE=~/genai-tfgpu/lib/python3.12/site-packages
cat > ~/genai-tfgpu/gpu_env.sh <<EOF
source ~/genai-tfgpu/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/nccl/lib
EOF

source ~/genai-tfgpu/gpu_env.sh
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
echo "Setup OK. Before running any TF script: source ~/genai-tfgpu/gpu_env.sh"
