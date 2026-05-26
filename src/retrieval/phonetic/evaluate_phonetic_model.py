
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

MODEL = Path("/storage/scratch1/0/rtaylor351/joker_retrieval/phonetic/models/bge-m3-ipa-prod-v2-short")
ITEMS = Path("/storage/scratch1/0/rtaylor351/joker_retrieval/phonetic/data/phonetic_items.tsv")

TOP_K = 15
SAMPLE_N = 5000
BATCH_SIZE = 128

def log(msg):
    print(msg, flush=True)

def main():
    log("[1/5] Loading phonetic_items.tsv")
    start = time.time()

    df = pd.read_csv(ITEMS, sep="\t")

    log(f"      loaded rows: {len(df):,} in {time.time() - start:.1f}s")

    log("[2/5] Cleaning/sample rows")

    df = df.dropna(subset=["ipa", "word"]).reset_index(drop=True)

    log(f"      valid rows: {len(df):,}")

    if len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=13).reset_index(drop=True)
        log(f"      sampled rows: {len(df):,}")

    texts = df["ipa"].astype(str).tolist()

    log("[3/5] Loading trained SentenceTransformer model")
    log(f"      model path: {MODEL}")

    start = time.time()

    model = SentenceTransformer(str(MODEL), device='cuda')

    log(f"      model loaded in {time.time() - start:.1f}s")

    log("[4/5] Encoding item embeddings")
    log(f"      items: {len(texts):,}")
    log(f"      batch size: {BATCH_SIZE}")

    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    log(f"      embeddings shape: {emb.shape}")

    queries = [
        "aokœ̃titʁ",
        "abɔ̃kɔ̃t",
        "apjedœvʁ",
        "abudə",
        "kɔ̃t",
        "ʃapitʁ",
    ]

    log("[5/5] Retrieving neighbors")
    log("\\nNEAREST NEIGHBORS\\n")

    for q in tqdm(queries, desc="queries"):
        q_emb = model.encode(
            [q],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = emb @ q_emb
        idx = np.argsort(-scores)[:TOP_K]

        print("=" * 80, flush=True)
        print(f"QUERY: {q}", flush=True)
        print("-" * 80, flush=True)

        for rank, i in enumerate(idx, 1):
            row = df.iloc[i]

            print(
                f"{rank:02d}  "
                f"{scores[i]:.4f}  "
                f"{row['ipa']}  "
                f"{row['word']}",
                flush=True,
            )

        print(flush=True)

if __name__ == "__main__":
    main()
