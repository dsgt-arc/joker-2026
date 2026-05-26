from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

ITEMS = "../data/retrieval/phonetic/phonetic_items.tsv"
MODEL = "../data/retrieval/phonetic/bge-m3-ipa-rebuilt-v1"
INDEX = "../data/retrieval/phonetic/phonetic_index.faiss"

TOP_K = 10

print("Loading phonetic items...")
items = pd.read_csv(ITEMS, sep="\t")

print("Loading model...")
model = SentenceTransformer(MODEL)

print("Loading FAISS index...")
index = faiss.read_index(INDEX)

queries = [
    "kɔ̃t",
    "ʃapitʁ",
    "abɔ̃kɔ̃t",
    "apjedœvʁ",
]

for q in queries:

    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("=" * 80)

    q_emb = model.encode(
        [q],
        normalize_embeddings=True,
    ).astype("float32")

    scores, idx = index.search(q_emb, TOP_K)

    rows = items.iloc[idx[0]]

    for rank, (score, (_, row)) in enumerate(
        zip(scores[0], rows.iterrows()),
        start=1,
    ):
        print(
            f"{rank:02d}  "
            f"{score:.4f}  "
            f"{row['ipa']}  "
            f"{row['word']}"
        )
