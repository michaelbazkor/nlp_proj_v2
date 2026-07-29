#!/usr/bin/env bash
# Push experiment artifacts via git LFS (weights, logs, reps). Never push LLMs or pics.zip.
set -euo pipefail
cd "$(dirname "$0")/.."

git lfs install
git add .gitattributes
git add artifacts/ reports/ configs/ src/ scripts/ pyproject.toml README.md || true
git status
echo "Review the staged files, then: git commit -m 'artifacts: <experiment>' && git push"
