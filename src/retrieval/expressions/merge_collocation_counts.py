import math
import os
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE = (
    Path("/storage/scratch1/0")
    / os.environ["USER"]
    / "joker_retrieval"
)

COUNT_DIR = (
    BASE
    / "expression_bank"
    / "counts"
)

OUTPUT = (
    BASE
    / "expression_bank"
    / "raw"
    / "collocations.parquet"
)

MIN_COUNT = 30
MAX_ROWS = 1000000

unigrams = Counter()
bigrams = Counter()
trigrams = Counter()

files = sorted(
    COUNT_DIR.glob("counts_*.pkl")
)

print(f"count files: {len(files)}")

for path in tqdm(files):

    with open(path, "rb") as f:
        data = pickle.load(f)

    unigrams.update(
        data["unigrams"]
    )

    bigrams.update(
        data["bigrams"]
    )

    trigrams.update(
        data["trigrams"]
    )

total = sum(
    unigrams.values()
)

rows = []

def add_rows(counter, source):

    for phrase, count in counter.items():

        if count < MIN_COUNT:
            continue

        words = phrase.split()

        denom = 1.0

        for w in words:

            denom *= max(
                unigrams[w] / total,
                1e-12,
            )

        p_phrase = count / total

        pmi = math.log2(
            max(
                p_phrase / denom,
                1e-12,
            )
        )

        if pmi < 2.5:
            continue

        score = (
            pmi
            * math.log1p(count)
        )

        rows.append(
            {
                "surface": phrase,
                "frequency": count,
                "pmi": pmi,
                "score": score,
                "source": source,
            }
        )

add_rows(
    bigrams,
    "opensubtitles_bigram",
)

add_rows(
    trigrams,
    "opensubtitles_trigram",
)

df = pd.DataFrame(rows)

df = df.sort_values(
    "score",
    ascending=False,
).head(MAX_ROWS)

df.to_parquet(OUTPUT)

print(df.shape)
print(df.head(100))
print(f"saved -> {OUTPUT}")
