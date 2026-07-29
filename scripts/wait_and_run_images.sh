#!/usr/bin/env bash
# Wait until pics.zip is a valid zip archive, then run the image-inclusive POC.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="$ROOT/pics.zip"
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "[wait] watching $ZIP ..."
while true; do
  if [[ -f "$ZIP" ]] && python -c "import zipfile; zipfile.ZipFile('$ZIP').namelist()" 2>/dev/null; then
    echo "[wait] zip ready: $(du -h "$ZIP" | cut -f1)"
    break
  fi
  sz=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  echo "[wait] not ready yet (size=${sz}); sleeping 60s"
  sleep 60
done

cd "$ROOT"
python -m ssr.cli --config configs/poc_standard.yaml captions
python -m ssr.cli --config configs/poc_standard.yaml all
python -m ssr.cli --config configs/poc_standard.yaml report
echo "[wait] image-inclusive POC complete"
