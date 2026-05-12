#!/usr/bin/env bash
# End-to-end smoke: synthetic CSVs -> MLPerf LoadGen + main.py (needs CUDA).
set -euo pipefail

DLRM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_ROOT="$(cd "$DLRM/../.." && pwd)"

# Python for this repo (override with PYTHON=...).
: "${PYTHON:=/home/comp/cswjyu/anaconda3/envs/dlrmv3/bin/python}"
: "${UV:=/home/comp/cswjyu/anaconda3/bin/uv}"

# Some shells set HTTPS_PROXY=http://127.0.0.1:7890 (local proxy) which breaks uv/curl
# when that service is down; lowercase https_proxy (e.g. campus) still works.
UV_PIP=(env -u HTTPS_PROXY -u HTTP_PROXY "$UV" pip)

# tiny data: --preset smoke (~seconds–minutes). Use "sampled" for 50k-user shard.
: "${DATA_PRESET:=smoke}"
DATA_DIR="${DATA_DIR:-$DLRM/_demo_streaming_data}"

echo "==> Data dir: $DATA_DIR (preset=$DATA_PRESET)"
mkdir -p "$DATA_DIR"
cd "$DLRM"

if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "==> Installing dlrm_v3 requirements (PyTorch + torchrec + fbgemm; CUDA cu124 wheel index)..."
  "${UV_PIP[@]}" install --python "$PYTHON" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match \
    -r "$DLRM/requirements.txt"
fi

if ! "$PYTHON" -c "import mlperf_loadgen" 2>/dev/null; then
  if [[ -n "${LOADGEN_SRC:-}" ]]; then
    echo "==> Installing MLPerf LoadGen from LOADGEN_SRC=$LOADGEN_SRC"
    "${UV_PIP[@]}" install --python "$PYTHON" "$LOADGEN_SRC"
  else
    echo "==> Installing MLPerf LoadGen (sparse checkout: only loadgen/, not the whole inference repo)..."
    LG_TMP="$(mktemp -d)"
    # Shallow + blob-less + sparse checkout: avoids downloading other benchmarks' trees/history.
    git clone --depth 1 --filter=blob:none --no-checkout \
      https://github.com/mlcommons/inference.git "$LG_TMP/inference"
    git -C "$LG_TMP/inference" sparse-checkout init --cone
    git -C "$LG_TMP/inference" sparse-checkout set loadgen
    git -C "$LG_TMP/inference" checkout
    "${UV_PIP[@]}" install --python "$PYTHON" "$LG_TMP/inference/loadgen"
    rm -rf "$LG_TMP"
  fi
fi

if [[ "${SKIP_DATA_GEN:-0}" == "1" ]] && [[ -f "$DATA_DIR/sampled_data/0.csv" ]]; then
  echo "==> SKIP_DATA_GEN=1 and sampled_data present, skipping streaming_synthetic_data.py"
else
  "$PYTHON" streaming_synthetic_data.py --preset "$DATA_PRESET" --output-folder "$DATA_DIR"
fi

DATASET="sampled-streaming-smoke"
if [[ "$DATA_PRESET" == "sampled" ]]; then
  DATASET="sampled-streaming-100b"
fi
if [[ "$DATA_PRESET" == "full" ]]; then
  echo "Refusing to run benchmark with preset=full (use smoke or sampled)." >&2
  exit 1
fi

export WORLD_SIZE="${WORLD_SIZE:-1}"
echo "==> Running benchmark: dataset=$DATASET WORLD_SIZE=$WORLD_SIZE"
cd "$DLRM"
# Empty checkpoint uses randomly initialized weights (see checkpoint.py).
"$PYTHON" main.py \
  --dataset "$DATASET" \
  --dataset-path-prefix "$DATA_DIR/" \
  --num-queries "${NUM_QUERIES:-256}" \
  --batchsize "${BATCHSIZE:-4}" \
  --model-path "" \
  --dataset-percentage "${DATASET_PERCENTAGE:-1.0}"
