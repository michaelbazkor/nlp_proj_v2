#!/usr/bin/env bash
# Full-cohort run: captions -> corpus -> represent (1 model per GPU) -> train -> report.
# Assumes the four caption shards were already launched; waits for them, then drives
# the remaining stages. Every stage is resumable, so re-running skips finished work.
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
CFG=configs/real_full.yaml
MODELS=(qwen3_32b deepseek_r1_distill_qwen_32b gemma4_26b_moe llama_3_3_70b)
T0=$(date +%s)

log() { echo "[pipeline +$(( ($(date +%s) - T0) / 60 ))m $(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 1. captions
for attempt in 1 2 3; do
  while pgrep -f "caption_shard.py" >/dev/null; do sleep 60; done
  missing=()
  for i in 0 1 2 3; do
    grep -q "DONE captioned=" "artifacts/full_captions_shard$i.log" || missing+=("$i")
  done
  [ ${#missing[@]} -eq 0 ] && break
  log "caption shards incomplete: ${missing[*]} (attempt $attempt) — relaunching"
  for i in "${missing[@]}"; do
    CUDA_VISIBLE_DEVICES=$i nohup python scripts/caption_shard.py --config $CFG \
      --shard "$i" --num-shards 4 --batch-size 48 --flush-every 480 \
      >>"artifacts/full_captions_shard$i.log" 2>&1 &
  done
  sleep 120
done
log "captions complete; merging shards"
python scripts/merge_caption_shards.py --config $CFG | tee artifacts/full_captions_merge.log

# ---------------------------------------------------------------- 2. corpus
log "building corpora"
python -m ssr.cli --config $CFG corpus --force >artifacts/full_corpus.log 2>&1
tail -4 artifacts/full_corpus.log

# ---------------------------------------------------------------- 3. represent
N_USERS=$(python -c "import pandas as pd;print(len(pd.read_parquet('artifacts/full/corpora.parquet')))")
log "represent target: $N_USERS users x ${#MODELS[@]} models"

count_done() { find "artifacts/full/reps/$1" -name '*.npz' 2>/dev/null | wc -l; }
incomplete() {
  local out=()
  for m in "${MODELS[@]}"; do
    [ "$(count_done "$m")" -lt "$N_USERS" ] && out+=("$m")
  done
  echo "${out[@]}"
}

for attempt in 1 2 3 4 5; do
  # shellcheck disable=SC2207
  todo=($(incomplete))
  [ ${#todo[@]} -eq 0 ] && break
  pids=()
  if [ ${#todo[@]} -eq 1 ]; then
    # Last model left (expected: the 8-bit 70B): split its users across all GPUs.
    m=${todo[0]}
    log "represent attempt $attempt: $m sharded over 4 GPUs"
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i SSR_ONLY_MODELS=$m SSR_USER_SHARD="$i/4" \
        nohup python -m ssr.cli --config $CFG represent \
        >>"artifacts/full_represent_${m}_gpu$i.log" 2>&1 &
      pids+=($!)
    done
  else
    log "represent attempt $attempt: one GPU per model -> ${todo[*]}"
    i=0
    for m in "${todo[@]}"; do
      CUDA_VISIBLE_DEVICES=$((i % 4)) SSR_ONLY_MODELS=$m \
        nohup python -m ssr.cli --config $CFG represent \
        >>"artifacts/full_represent_$m.log" 2>&1 &
      pids+=($!)
      log "  GPU$((i % 4)) -> $m (pid $!)"
      i=$((i + 1))
    done
  fi
  for p in "${pids[@]}"; do wait "$p"; done
  for m in "${MODELS[@]}"; do log "  $m: $(count_done "$m")/$N_USERS user files"; done
done

# shellcheck disable=SC2207
left=($(incomplete))
if [ ${#left[@]} -ne 0 ]; then
  log "ABORT: representations incomplete for ${left[*]}; training needs every user"
  exit 1
fi
log "represent finished for all models"

# ---------------------------------------------------------------- 4. train
log "training STM/MTM (parallel grid search)"
python scripts/train_parallel.py --config $CFG >artifacts/full_train.log 2>&1
tail -12 artifacts/full_train.log

# ---------------------------------------------------------------- 5. report
log "compiling report"
python scripts/make_report.py --config $CFG
log "PIPELINE DONE"
