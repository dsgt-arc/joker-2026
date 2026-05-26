import os
import pickle
import re
from collections import Counter
from pathlib import Path

import spacy
from tqdm import tqdm

BASE = Path("/storage/scratch1/0") / os.environ["USER"] / "joker_retrieval"
TASK_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])
SHARD = f"{TASK_ID:03d}"

INPUT = BASE / "corpora" / "shards" / f"fr_{SHARD}"
OUTPUT = BASE / "expression_bank" / "counts" / f"counts_{SHARD}.pkl"

TOKEN_RE = re.compile(r"^[a-zàâçéèêëîïôûùüÿñæœ'-]+$", re.I)

ALLOWED_PATTERNS = {
    ("ADJ", "NOUN"),
    ("NOUN", "ADJ"),
    ("NOUN", "NOUN"),
    ("VERB", "NOUN"),
    ("VERB", "ADV"),
    ("ADV", "ADJ"),
}

STOP_EDGE = {
    "je","tu","il","elle","nous","vous","ils","elles","on",
    "le","la","les","un","une","des","du","de","d","à","au","aux",
    "et","ou","mais","donc","or","ni","car","que","qui","quoi",
    "ne","pas","plus","tout","tous","très","ce","cet","cette","ces",
    "me","te","se","moi","toi","lui","leur","leurs",
}

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

unigrams = Counter()
bigrams = Counter()
trigrams = Counter()

def keep_token(tok):
    text = tok.text.strip().lower()
    return (
        len(text) > 1
        and TOKEN_RE.match(text)
        and tok.pos_ not in {"PUNCT", "SPACE", "SYM", "NUM", "X"}
    )

def process(doc):
    toks = []

    for tok in doc:
        if keep_token(tok):
            lemma = tok.lemma_.lower().strip("'’-")
            if lemma:
                toks.append((lemma, tok.pos_))

    for lemma, _ in toks:
        unigrams[lemma] += 1

    for i in range(len(toks) - 1):
        w1, p1 = toks[i]
        w2, p2 = toks[i + 1]

        if w1 in STOP_EDGE or w2 in STOP_EDGE:
            continue

        if (p1, p2) in ALLOWED_PATTERNS:
            bigrams[f"{w1} {w2}"] += 1

    for i in range(len(toks) - 2):
        w1, p1 = toks[i]
        w2, p2 = toks[i + 1]
        w3, p3 = toks[i + 2]

        if w1 in STOP_EDGE or w3 in STOP_EDGE:
            continue

        content = sum(p in {"NOUN", "VERB", "ADJ", "ADV"} for p in (p1, p2, p3))

        if content >= 2:
            trigrams[f"{w1} {w2} {w3}"] += 1

def line_iter():
    with open(INPUT, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().lower()
            if line:
                yield line

for doc in tqdm(nlp.pipe(line_iter(), batch_size=2048), desc=f"shard {SHARD}"):
    process(doc)

with open(OUTPUT, "wb") as f:
    pickle.dump(
        {
            "unigrams": unigrams,
            "bigrams": bigrams,
            "trigrams": trigrams,
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print(f"saved {OUTPUT}")
