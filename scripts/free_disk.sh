#!/usr/bin/env bash
# Free disk for the POC / H100 staging VM.
set -euo pipefail
rm -rf \
  "${HOME}/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct" \
  "${HOME}/.ollama" \
  "${HOME}/nlp_proj" \
  "${HOME}/notebooks" \
  "${HOME}/llama.cpp"
df -h /
echo "done"
