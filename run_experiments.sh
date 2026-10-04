#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh —— C2G 实验矩阵一键运行脚本
#
# 用法（在 8xH100 远程机器上，先完成数据下载）：
#   bash run_experiments.sh e001   # 复现 baseline
#   bash run_experiments.sh e002   # SP4096 单点改进
#   bash run_experiments.sh all    # 依次跑完整个矩阵
#
# 每个实验产出 logs/<RUN_ID>.txt，可用 analyze_logs.py 汇总。
# 说明：数据与 tokenizer 需先由 data/cached_challenge_fineweb.py 下载缓存。
# =============================================================================
set -euo pipefail

# ---- 路径配置（按需修改）----
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/WangChunYan_C2G_train_gpt.py}"
NPROC="${NPROC:-8}"

# 数据根目录（官方缓存目录，由 cached_challenge_fineweb.py 生成）
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/parameter-golf/data/datasets}"

run() {
  local run_id="$1" data_dir="$2" tok_path="$3" vocab="$4"
  shift 4
  echo "====> 启动 $run_id"
  NCCL_IB_DISABLE=1 \
  RUN_ID="$run_id" \
  DATA_PATH="$data_dir" \
  TOKENIZER_PATH="$tok_path" \
  VOCAB_SIZE="$vocab" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}" \
  "$@" \
  torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN_SCRIPT"
}

case "${1:-}" in
  e001) # baseline 复现（SP1024）
    run e001_baseline "$DATA_ROOT/fineweb10B_sp1024" \
        "$REPO_DIR/parameter-golf/data/tokenizers/fineweb_1024_bpe.model" 1024 \
        SEED=1337 ;;
  e002) # SP4096 单点改进
    run e002_sp4096 "$DATA_ROOT/fineweb10B_sp4096" \
        "$REPO_DIR/parameter-golf/data/tokenizers/fineweb_4096_bpe.model" 4096 \
        SEED=1337 ;;
  e003) # SP8192 单点改进
    run e003_sp8192 "$DATA_ROOT/fineweb10B_sp8192" \
        "$REPO_DIR/parameter-golf/data/tokenizers/fineweb_8192_bpe.model" 8192 \
        SEED=1337 ;;
  e004) # SP4096 + MLP3x
    run e004_sp4096_mlp3 "$DATA_ROOT/fineweb10B_sp4096" \
        "$REPO_DIR/parameter-golf/data/tokenizers/fineweb_4096_bpe.model" 4096 \
        MLP_MULT=3 SEED=1337 ;;
  e005) # SP4096 + MLP3x + QK-gain
    run e005_sp4096_mlp3_qk "$DATA_ROOT/fineweb10B_sp4096" \
        "$REPO_DIR/parameter-golf/data/tokenizers/fineweb_4096_bpe.model" 4096 \
        MLP_MULT=3 QK_GAIN_INIT=4.5 SEED=1337 ;;
  all)
    for s in 1337 42 2025; do
      # 关键实验三 seed
      run e002_sp4096_s$s "$DATA_ROOT/fineweb10B_sp4096" \
          "$REPO_DIR/parameter-golf/data/tokenizers/fineweb_4096_bpe.model" 4096 SEED=$s
      run e004_sp4096_mlp3_s$s "$DATA_ROOT/fineweb10B_sp4096" \
          "$REPO_DIR/parameter-golf/data/tokenizers/fineweb_4096_bpe.model" 4096 \
          MLP_MULT=3 SEED=$s
    done
    ;;
  *)
    echo "用法: bash run_experiments.sh {e001|e002|e003|e004|e005|all}"
    exit 1 ;;
esac
