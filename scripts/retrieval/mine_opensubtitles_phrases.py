import re
from collections import Counter

import pandas as pd
from tqdm import tqdm

INPUT = "corpora/fr.txt"
OUTPUT = "expression_bank/raw/opensubtitles_mined.parquet"

NGRAM_MIN = 2
NGRAM_MAX = 6
MIN_COUNT = 30
MAX_ROWS = 500_000

TOKEN_RE = re.compile(r"[a-zàâçéèêëîïôûùüÿñæœ'-]+", re.I)

BAD_START_END = {
    "le", "la", "les", "un", "une", "des", "de", "du", "à", "au", "aux",
    "et", "ou", "mais", "donc", "or", "ni", "car", "que", "qui", "quoi",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
}

def tokenize(line):
    return [t.lower().strip("-'") for t in TOKEN_RE.findall(line.lower()) if len(t.strip("-'")) > 1]

def keep_phrase(tokens):
    if len(tokens) < NGRAM_MIN:
        return False
    if tokens[0] in BAD_START_END or tokens[-1] in BAD_START_END:
        return False
    if all(len(t) <= 2 for t in tokens):
        return False
    return True

def main():
    counts = Counter()
    line_count = 0

    with open(INPUT, encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f):
            toks = tokenize(line)
            line_count += 1

            if len(toks) < NGRAM_MIN:
                continue

            for n in range(NGRAM_MIN, NGRAM_MAX + 1):
                if len(toks) < n:
                    continue

                for i in range(len(toks) - n + 1):
                    gram = toks[i:i+n]
                    if keep_phrase(gram):
                        counts[" ".join(gram)] += 1

    rows = [
        {
            "surface": phrase,
            "content": "",
            "source": "opensubtitles",
            "frequency": count,
        }
        for phrase, count in counts.items()
        if count >= MIN_COUNT
    ]

    rows.sort(key=lambda x: -x["frequency"])
    rows = rows[:MAX_ROWS]

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT)

    print("lines", line_count)
    print(df.shape)
    print(df.head(20))
    print(f"saved -> {OUTPUT}")

if __name__ == "__main__":
    main()
