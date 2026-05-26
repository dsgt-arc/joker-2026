from pathlib import Path
import pandas as pd

rows = []

files = [
    p for p in Path("corpora/sharedtask-data").rglob("*.cupt")
    if "/FR/" in str(p) or "/fr/" in str(p).lower() or "french" in str(p).lower()
]

print(f"French-looking files: {len(files)}")
for p in files[:20]:
    print(p)

for path in files:
    sent_tokens = []
    sent_has_mwe = False

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                if sent_has_mwe and sent_tokens:
                    rows.append({
                        "surface": " ".join(sent_tokens),
                        "content": "",
                        "source": "parseme",
                        "file": str(path),
                    })
                sent_tokens = []
                sent_has_mwe = False
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 11:
                continue

            token = cols[1]
            mwe = cols[10]
            sent_tokens.append(token)

            if mwe != "*":
                sent_has_mwe = True

df = pd.DataFrame(rows)
if len(df):
    df.drop_duplicates(subset=["surface"], inplace=True)

df.to_parquet("expression_bank/raw/parseme.parquet")
print(df.shape)
print(df.head())
