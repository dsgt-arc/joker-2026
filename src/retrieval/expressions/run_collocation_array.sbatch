#!/bin/bash
#SBATCH --job-name=colloc
#SBATCH --account=paceship-clef2026_joker
#SBATCH --qos=inferno
#SBATCH --array=0-63
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/scratch1/0/%u/joker_retrieval/logs/colloc_%A_%a.out
#SBATCH --error=/storage/scratch1/0/%u/joker_retrieval/logs/colloc_%A_%a.err

set -euo pipefail

BASE=/storage/scratch1/0/$USER/joker_retrieval
PY=$BASE/conda_envs/punretrieval/bin/python

cd "$BASE"

"$PY" scripts/mine_collocation_shard.py
