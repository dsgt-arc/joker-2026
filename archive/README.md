# archive

Superseded code kept for reference. Nothing here is imported by the active 2026 pipeline in `src/`.

- `2025-previous/` — the 2023/2025 CLEF Joker pipeline (contrastive learning, cosine-similarity retrieval, 2025 generator/discriminator) plus a couple of unreferenced local draft scripts. Superseded by the 2026 system described in the [README](../README.md).
- `scripts/` — earlier iterations of standalone reporting/eval scripts, each superseded by a later version still living in `src/`:
  - `retrieval_stats.py` → `retrieval_stats_v2.py` → `retrieval_stats_v3.py` → `retrieval_stats_v4.py` → current: [`src/retrieval_stats_v5.py`](../src/retrieval_stats_v5.py)
  - `eval_phonetic_retrieval.py` → current: [`src/eval_phonetic_retrieval.py`](../src/eval_phonetic_retrieval.py) (was `eval_phonetic_retrieval_fixed.py`)
