#!/usr/bin/env bash
# Re-shard one model's remaining users across several GPUs.
#
# Extraction is resumable (finished users are skipped both when the work list is
# built and again inside the loop), so stopping a single-GPU worker and restarting
# it as N shards only loses the user currently in flight.
#
# Usage: scripts/rebalance_represent.sh <model_name> <gpu,gpu,...>
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
CFG=configs/real_full.yaml
MODEL=$1
IFS=',' read -r -a GPUS <<<"$2"
N=${#GPUS[@]}

# The model name lives in the environment, not the command line, so match on
# /proc/<pid>/environ rather than pgrep patterns.
echo "[rebalance] stopping existing workers for $MODEL"
for pid in $(pgrep -f "ssr.cli --config $CFG represent"); do
  if tr '\0' '\n' </proc/"$pid"/environ 2>/dev/null | grep -qx "SSR_ONLY_MODELS=$MODEL"; then
    echo "[rebalance] kill $pid ($MODEL)"
    kill "$pid"
  fi
done
sleep 20

done_now=$(find "artifacts/full/reps/$MODEL" -name '*.npz' 2>/dev/null | wc -l)
echo "[rebalance] $MODEL has $done_now users cached; launching $N shards on GPUs ${GPUS[*]}"
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g SSR_ONLY_MODELS=$MODEL SSR_USER_SHARD="$i/$N" \
    nohup python -m ssr.cli --config $CFG represent \
    >>"artifacts/full_represent_${MODEL}_gpu${g}.log" 2>&1 &
  echo "[rebalance]   GPU$g shard $i/$N pid $!"
done
