import bz2
import html
import re
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm

xml_path = "corpora/frwiktionary-latest-pages-articles.xml"
out_path = "expression_bank/raw/wiktionary.parquet"

KEYWORDS = ("locution", "proverbe", "expression")

rows = []
count = 0

context = ET.iterparse(xml_path, events=("end",))

for event, elem in tqdm(context):
    tag = elem.tag.split("}")[-1]

    if tag != "page":
        continue

    title = None
    text = None

    for child in elem:
        ctag = child.tag.split("}")[-1]
        if ctag == "title":
            title = child.text or ""
        elif ctag == "revision":
            for rchild in child:
                rtag = rchild.tag.split("}")[-1]
                if rtag == "text":
                    text = rchild.text or ""
                    break

    if title and text:
        lc = text.lower()
        if any(k in lc for k in KEYWORDS):
            rows.append({
                "surface": html.unescape(title),
                "content": text[:2000],
                "source": "wiktionary",
            })

    elem.clear()
    count += 1

    if len(rows) % 5000 == 0 and rows:
        print(f"pages={count:,} rows={len(rows):,}")

df = pd.DataFrame(rows)
df.drop_duplicates(subset=["surface"], inplace=True)
df.to_parquet(out_path)
print(df.shape)
print(f"saved {out_path}")
