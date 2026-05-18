import faiss
import numpy as np

EMB = "expression_bank/faiss/expression_embeddings.npy"
OUT = "expression_bank/faiss/expression_index.faiss"

x = np.load(EMB).astype("float32")

index = faiss.IndexFlatIP(x.shape[1])
index.add(x)

faiss.write_index(index, OUT)

print(index.ntotal)
print(f"saved -> {OUT}")
