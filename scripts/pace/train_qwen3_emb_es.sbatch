#!/bin/bash
#SBATCH -J qwen3-emb-es
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/joker-2025}"
ENV_NAME="${ENV_NAME:-joker}"

cd "$REPO_DIR"
mkdir -p logs artifacts/embeddings

module purge
module load anaconda3
source activate "$ENV_NAME"

python -m pip install -U pip
python -m pip install -U transformers datasets peft accelerate scikit-learn sentencepiece faiss-cpu

python scripts/training/train_qwen3_embeddings.py \
  --language es \
  --train-path data/processed/embedding_pairs.csv \
  --train-file-type csv \
  --corpus-path data/processed/retrieval_corpus_es.csv \
  --corpus-file-type csv \
  --corpus-text-col text \
  --output-root artifacts/embeddings \
  --epochs 3 \
  --batch-size 16 \
  --max-length 256
