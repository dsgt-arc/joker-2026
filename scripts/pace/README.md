# PACE Jobs: Qwen3 Embedding Training

This directory contains Slurm jobs to train language-specific LoRA adapters for `Qwen/Qwen3-Embedding-8B` and optionally build FAISS indexes.

## Files

- `train_qwen3_emb_fr.sbatch`: French training + FAISS build.
- `train_qwen3_emb_es.sbatch`: Spanish training + FAISS build.
- `../training/train_qwen3_embeddings.py`: shared training entrypoint.

## Expected data schema

Training file (`--train-path`) columns:
- `language` (`fr` or `es`)
- `anchor`
- `positive`
- optional `hard_negative`

Corpus file (`--corpus-path`) columns:
- text column (default `text`)

## Submit jobs on PACE

```bash
cd ~/joker-2025
mkdir -p logs
sbatch scripts/pace/train_qwen3_emb_fr.sbatch
sbatch scripts/pace/train_qwen3_emb_es.sbatch
```

If your repo or conda env paths differ:

```bash
REPO_DIR=/path/to/joker-2025 ENV_NAME=myenv sbatch scripts/pace/train_qwen3_emb_fr.sbatch
REPO_DIR=/path/to/joker-2025 ENV_NAME=myenv sbatch scripts/pace/train_qwen3_emb_es.sbatch
```

## Outputs

- Adapters:
  - `artifacts/embeddings/qwen3-emb-8b-fr-lora`
  - `artifacts/embeddings/qwen3-emb-8b-es-lora`
- FAISS artifacts:
  - `artifacts/embeddings/faiss/fr_qwen3_emb8b.faiss`
  - `artifacts/embeddings/faiss/es_qwen3_emb8b.faiss`
  - matching `*_embeddings.npy`

## Notes

- Default partition is `a100`; change `#SBATCH -p` to your available GPU partition on PACE.
- You should update `--train-path` and `--corpus-path` in each `.sbatch` file to your real datasets.
