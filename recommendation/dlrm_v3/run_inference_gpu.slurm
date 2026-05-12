#!/bin/bash
#
# DLRMv3 推理基准测试 SLURM 脚本 — gpu24 节点
#
# 优化汇总：
#   1. batchsize=8 (Server) / 8 (Offline)
#   2. non_blocking CPU→GPU 传输 + 专用 CUDA stream (inference_modules.py)
#   3. Dist 路径去掉 deepcopy，CPU tensor 走 shared memory (model_family.py)
#   NOTE: torch.compile 已禁用 — 模型已用手写 Triton 内核，compile 只加 overhead
#
#SBATCH --job-name=DLRMv3_Infer
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -w gpu24
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/infer_gpu24_%j.out
#SBATCH --error=logs/infer_gpu24_%j.err

# ── conda 初始化 ────────────────────────────────────────────────
__conda_setup="$('/home/comp/cswjyu/anaconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/comp/cswjyu/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/comp/cswjyu/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/comp/cswjyu/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "激活 Conda 环境: dlrmv3"
conda activate dlrmv3

# ── 执行目录 ────────────────────────────────────────────────────
EXEC_DIR="/home/comp/cswjyu/orion-yuwenjun/inference/recommendation/dlrm_v3"
echo "导航到执行目录: $EXEC_DIR"
cd "$EXEC_DIR" || { echo "无法进入目录，退出"; exit 1; }

mkdir -p logs

echo "当前 Python: $(which python)"
echo "当前 Conda 环境: $CONDA_DEFAULT_ENV"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 流式数据目录：须含 sampled_data/offset.csv 等（仓库内 smoke 演示数据）
DATA_DIR="/home/comp/cswjyu/orion-yuwenjun/inference/recommendation/dlrm_v3/_demo_streaming_data"

export WORLD_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# 精简 Triton autotune 配置：30 个 → 6 个（V100/A100 推理最优候选）
# 从 30 分钟降低到 ~2 分钟。首次跑完后结果缓存到 ~/.triton/cache/。
export HSTU_FAST_AUTOTUNE=1

# ── Triton 编译缓存（使用 Triton 默认路径，所有 Job 共享）───────────
# 首次推理时 Triton JIT 编译 HSTU attention 算子（V100 约 10-30 分钟）
# 之后的运行直接复用 ~/.triton/cache/，无需重新编译。
# !! 升级 Triton 版本后如遇问题可 rm -rf ~/.triton/cache/ 清空 !!
echo "Triton 缓存目录: ${HOME}/.triton/cache/"
ls ~/.triton/cache/ 2>/dev/null | wc -l | xargs -I{} echo "  已缓存内核数: {}"

echo "========================================================"
echo "  [1/2] Server 场景  batchsize=8"
echo "  首次运行：Triton 内核编译（约 10-30 min），后续 <1 min"
echo "========================================================"

# --use-compile False：模型使用手写 Triton 内核，torch.compile 反而增加 overhead
torchrun --nproc_per_node="${WORLD_SIZE}" main.py \
    --dataset sampled-streaming-smoke \
    --dataset-path-prefix "${DATA_DIR}/" \
    --scenario-name Server \
    --batchsize 8 \
    --target-qps 3000 \
    --dataset-percentage 0.001 \
    --data-producer-threads 8 \
    2>&1 | tee logs/infer_server_gpu24_${SLURM_JOB_ID}.log

echo ""
echo "========================================================"
echo "  [2/2] Offline 场景  batchsize=8"
echo "  第二次运行：复用 Triton 缓存，应快速完成"
echo "========================================================"

torchrun --nproc_per_node="${WORLD_SIZE}" main.py \
    --dataset sampled-streaming-smoke \
    --dataset-path-prefix "${DATA_DIR}/" \
    --scenario-name Offline \
    --batchsize 8 \
    --target-qps 10000 \
    --dataset-percentage 0.001 \
    --data-producer-threads 8 \
    2>&1 | tee logs/infer_offline_gpu24_${SLURM_JOB_ID}.log

echo "脚本执行完成。"
