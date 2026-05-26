import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("expression_bank/processed/expression_bank.parquet")
texts = df["surface"].fillna("").astype(str).tolist()

model = SentenceTransformer("BAAI/bge-m3", device="cuda")

embeddings = model.encode(
    texts,
    batch_size=1024,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

np.save("expression_bank/faiss/expression_embeddings.npy", embeddings)

print(embeddings.shape)
print("saved -> expression_bank/faiss/expression_embeddings.npy")
