import pandas as pd

paths = [
    "expression_bank/raw/wiktionary.parquet",
    "expression_bank/raw/parseme.parquet",
    "expression_bank/raw/collocations.parquet",
]

dfs = []

for path in paths:

    df = pd.read_parquet(path)

    cols = []

    for c in [
        "surface",
        "content",
        "source",
        "frequency",
        "pmi",
        "score",
    ]:
        if c in df.columns:
            cols.append(c)

    df = df[cols].copy()

    if "content" not in df.columns:
        df["content"] = ""

    if "frequency" not in df.columns:
        df["frequency"] = 1

    if "pmi" not in df.columns:
        df["pmi"] = 0.0

    if "score" not in df.columns:
        df["score"] = 0.0

    df["surface"] = (
        df["surface"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["surface"].str.len() >= 3
    ]

    dfs.append(df)

df = pd.concat(
    dfs,
    ignore_index=True,
)

df = df.drop_duplicates(
    subset=["surface"]
)

df = df.reset_index(drop=True)

print(df.shape)
print(df["source"].value_counts())

df.to_parquet(
    "expression_bank/processed/expression_bank.parquet"
)

print(
    "saved -> expression_bank/processed/expression_bank.parquet"
)
