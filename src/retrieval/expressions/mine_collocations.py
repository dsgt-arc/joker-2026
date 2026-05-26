import math
import re
from collections import Counter

import pandas as pd
import spacy
from tqdm import tqdm

INPUT = "corpora/fr.txt"
OUTPUT = "expression_bank/raw/collocations.parquet"

MIN_COUNT = 30
MAX_ROWS = 750000
BATCH_SIZE = 10000

TOKEN_RE = re.compile(r"^[a-zàâçéèêëîïôûùüÿñæœ'-]+$", re.I)

ALLOWED_PATTERNS = {
    ("ADJ", "NOUN"),
    ("NOUN", "ADJ"),
    ("NOUN", "NOUN"),
    ("VERB", "NOUN"),
    ("VERB", "ADV"),
}

STOP_EDGE = {
    "je","tu","il","elle","nous","vous","ils","elles","on",
    "le","la","les","un","une","des","du","de","d","à","au","aux",
    "et","ou","mais","donc","or","ni","car","que","qui","quoi",
    "ne","pas","plus","tout","tous","très","ce","cet","cette","ces",
}

nlp = spacy.load(
    "fr_core_news_sm",
    disable=["ner", "parser"],
)

unigrams = Counter()
ngrams = Counter()

def good_token(tok):
    txt = tok.text.strip().lower()

    if not TOKEN_RE.match(txt):
        return False

    if len(txt) <= 1:
        return False

    if tok.pos_ in {"PUNCT", "SPACE", "SYM", "NUM"}:
        return False

    return True

def add_doc(doc):

    toks = []

    for tok in doc:

        if not good_token(tok):
            continue

        lemma = tok.lemma_.lower().strip("'’-")

        if not lemma:
            continue

        toks.append((lemma, tok.pos_))

    for lemma, _ in toks:
        unigrams[lemma] += 1

    for i in range(len(toks) - 1):

        w1, p1 = toks[i]
        w2, p2 = toks[i + 1]

        if w1 in STOP_EDGE or w2 in STOP_EDGE:
            continue

        if (p1, p2) in ALLOWED_PATTERNS:
            ngrams[f"{w1} {w2}"] += 1

def main():

    batch = []

    with open(INPUT, encoding="utf-8", errors="ignore") as f:

        for i, line in enumerate(tqdm(f), start=1):

            line = line.strip().lower()

            if not line:
                continue

            batch.append(line)

            if i % 10000 == 0:
                print(f"processed {i:,} lines")

            if len(batch) >= BATCH_SIZE:

                for doc in nlp.pipe(
                    batch,
                    batch_size=512,
                ):
                    add_doc(doc)

                batch = []

        if batch:

            for doc in nlp.pipe(
                batch,
                batch_size=512,
            ):
                add_doc(doc)

    total = sum(unigrams.values())

    rows = []

    for phrase, count in ngrams.items():

        if count < MIN_COUNT:
            continue

        words = phrase.split()

        denom = 1.0

        for w in words:
            denom *= max(unigrams[w] / total, 1e-12)

        p_phrase = count / total

        pmi = math.log2(
            max(p_phrase / denom, 1e-12)
        )

        if pmi < 2.5:
            continue

        score = pmi * math.log1p(count)

        rows.append({
            "surface": phrase,
            "frequency": count,
            "pmi": pmi,
            "score": score,
            "source": "opensubtitles_collocation",
        })

    df = pd.DataFrame(rows)

    if len(df):
        df = df.sort_values(
            "score",
            ascending=False,
        ).head(MAX_ROWS)

    df.to_parquet(OUTPUT)

    print(df.shape)
    print(df.head(50))
    print(f"saved -> {OUTPUT}")

if __name__ == "__main__":
    main()
