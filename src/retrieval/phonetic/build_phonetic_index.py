from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    phonetic_items_path,
    phonetic_embeddings_path,
    phonetic_index_path,
    phonetic_model_path,
)
from data import load


def main() -> None:
    print("Loading phonetic items...")
    items = load(phonetic_items_path)

    if "ipa" not in items.columns:
        raise ValueError(f"phonetic_items.tsv must contain an 'ipa' column. Found: {items.columns}")

    ipa_strings = items["ipa"].astype(str).fillna("").tolist()

    print(f"Loading model: {phonetic_model_path}")
    model = SentenceTransformer(phonetic_model_path)

    print(f"Encoding {len(ipa_strings):,} IPA strings...")
    embeddings = model.encode(
        ipa_strings,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    out_emb = Path(phonetic_embeddings_path)
    out_idx = Path(phonetic_index_path)
    out_emb.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving embeddings: {out_emb}")
    np.save(out_emb, embeddings)

    print("Building FAISS IndexFlatIP...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Saving FAISS index: {out_idx}")
    faiss.write_index(index, str(out_idx))

    print("Done.")
    print(f"items: {len(items):,}")
    print(f"embedding shape: {embeddings.shape}")
    print(f"index vectors: {index.ntotal:,}")


if __name__ == "__main__":
    main()
