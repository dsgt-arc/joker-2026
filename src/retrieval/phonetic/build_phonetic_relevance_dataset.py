import hashlib
import os
import random
import re
import unicodedata
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm



# PACE defaults. Override with environment variables if needed.
PHONETIC_INPUT_PATH = Path(os.environ.get(
    "PHONETIC_INPUT_PATH",
    "/storage/scratch1/0/rtaylor351/joker_retrieval/phonetic/data/phonetic_items.tsv",
))
PHONETIC_OUTPUT_DIR = Path(os.environ.get(
    "PHONETIC_OUTPUT_DIR",
    "/storage/scratch1/0/rtaylor351/joker_retrieval/phonetic/data_rebuilt",
))


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)

    # Fallback: try TSV first, then CSV.
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)

SEED = 13
random.seed(SEED)

NUM_WORKERS = int(os.environ.get("PHONETIC_NUM_WORKERS", "16"))
NEAR_SCAN = int(os.environ.get("PHONETIC_NEAR_SCAN", "300"))
HARD_SCAN = int(os.environ.get("PHONETIC_HARD_SCAN", "300"))

# Main fix: keep exact / near-homophone signal strong, but stop rhyme/skeleton
# examples from dominating the embedding geometry.
MAX_EXACT_PER_ANCHOR = 4
MAX_NEAR_PER_ANCHOR = 8
MAX_STRONG_RHYME_PER_ANCHOR = 2
MAX_WEAK_RHYME_PER_ANCHOR = 2
MAX_SKELETON_PER_ANCHOR = 2
MAX_SYNTHETIC_PER_ANCHOR = 4
MAX_HARD_NEGATIVES_PER_ANCHOR = 12

TRAIN_FRAC = 0.90
DEV_FRAC = 0.05
TRAIN_PAIR_MIN_SCORE = float(os.environ.get("PHONETIC_TRAIN_PAIR_MIN_SCORE", "0.35"))

VOWELS = set("aeɛəiœøouyɑɔæɶẽɛ̃ɑ̃ɔ̃œ̃")
FRENCH_CONSONANTS = set("pbtdkgfvszʃʒmnɲŋlʁrjwɥxɡç")

ITEMS = []
BY_IPA = {}
BY_LEN = {}
BY_ONSET = {}
BY_SUFFIX2 = {}
BY_SUFFIX3 = {}
BY_RHYME = {}
BY_CONS = {}
BY_VOWELS = {}


def stable_float(key: str) -> float:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_split(key: str) -> str:
    x = stable_float(key)
    if x < TRAIN_FRAC:
        return "train"
    if x < TRAIN_FRAC + DEV_FRAC:
        return "dev"
    return "test"


def normalize_ipa(x: str) -> str:
    x = unicodedata.normalize("NFC", str(x).strip())
    x = x.replace("[", "").replace("]", "").replace("/", "")
    x = x.replace("ˈ", "").replace("ˌ", "")
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[|‖]", " ", x)
    return x.strip()


def normalize_surface(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def ipa_tokens(ipa: str) -> Tuple[str, ...]:
    ipa = normalize_ipa(ipa)
    out = []
    i = 0

    while i < len(ipa):
        ch = ipa[i]

        if ch.isspace() or ch in ".-_":
            out.append("|")
            i += 1
            continue

        token = ch
        i += 1

        while i < len(ipa) and unicodedata.combining(ipa[i]):
            token += ipa[i]
            i += 1

        if i < len(ipa) and ipa[i] in {"ː", "̆"}:
            token += ipa[i]
            i += 1

        out.append(token)

    compact = []
    for t in out:
        if t == "|" and (not compact or compact[-1] == "|"):
            continue
        compact.append(t)

    if compact and compact[-1] == "|":
        compact.pop()

    return tuple(compact)


def strip_boundaries(tokens: Sequence[str]) -> Tuple[str, ...]:
    return tuple(t for t in tokens if t != "|")


def is_vowel_token(t: str) -> bool:
    if not t:
        return False
    base = unicodedata.normalize("NFD", t)[0]
    return base in VOWELS


def is_consonant_token(t: str) -> bool:
    if not t:
        return False
    base = unicodedata.normalize("NFD", t)[0]
    return base in FRENCH_CONSONANTS


def consonant_skeleton(tokens: Sequence[str]) -> str:
    return "".join(t for t in tokens if is_consonant_token(t))


def vowel_skeleton(tokens: Sequence[str]) -> str:
    return "".join(t for t in tokens if is_vowel_token(t))


def onset_key(tokens: Sequence[str], n: int = 2) -> str:
    clean = strip_boundaries(tokens)
    return " ".join(clean[:n])


def suffix_key(tokens: Sequence[str], n: int) -> str:
    clean = strip_boundaries(tokens)
    return " ".join(clean[-n:]) if len(clean) >= n else ""


def rhyme_key(tokens: Sequence[str]) -> str:
    clean = list(strip_boundaries(tokens))
    last_vowel = None

    for i in range(len(clean) - 1, -1, -1):
        if is_vowel_token(clean[i]):
            last_vowel = i
            break

    if last_vowel is None:
        return ""

    return " ".join(clean[last_vowel:])


def edit_distance(a: Sequence[str], b: Sequence[str], max_dist: Optional[int] = None) -> int:
    a = list(strip_boundaries(a))
    b = list(strip_boundaries(b))

    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        curr = [i]
        row_min = i

        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            curr.append(val)
            row_min = min(row_min, val)

        prev = curr

        if max_dist is not None and row_min > max_dist:
            return max_dist + 1

    return prev[-1]


def jaccard_char_ngrams(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> set:
        s = re.sub(r"\s+", "", s)
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0

    return len(ga & gb) / len(ga | gb)


def relation_score(relation_type: str) -> float:
    scores = {
        "identity": 1.00,
        "exact_homophone": 1.00,
        "synthetic_schwa_drop": 0.86,
        "synthetic_boundary_drop": 0.84,
        "synthetic_r_variant": 0.78,
        "synthetic_nasal_variant": 0.78,
        "near_homophone_edit1": 0.82,
        "near_homophone_edit2": 0.70,
        # Rhyme is useful but should not dominate the retrieval encoder.
        "strong_rhyme": 0.38,
        "weak_rhyme": 0.18,
        # Skeletons are weak outer-neighborhood hints, not strong positives.
        "consonant_skeleton": 0.20,
        "vowel_skeleton": 0.12,
        "hard_negative": 0.00,
        "rhyme_hard_negative": 0.00,
    }
    return scores[relation_type]


def deterministic_sample(items: Sequence[int], k: int, key: str) -> list[int]:
    items = list(dict.fromkeys(items))
    if len(items) <= k:
        return items

    rng_seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(rng_seed)
    return rng.sample(items, k)


def add_identity_relation(rows: list, aid: int) -> None:
    a = ITEMS[aid]
    rows.append({
        "anchor_id": aid,
        "candidate_id": aid,
        "anchor_word": a["word"],
        "anchor_ipa": a["ipa"],
        "candidate_word": a["word"],
        "candidate_ipa": a["ipa"],
        "relation_type": "identity",
        "target_score": relation_score("identity"),
        "anchor_split": a["split"],
        "candidate_split": a["split"],
    })


def add_relation(rows: list, anchor_id: int, candidate_id: int, relation_type: str) -> None:
    if anchor_id == candidate_id:
        return

    a = ITEMS[anchor_id]
    c = ITEMS[candidate_id]

    rows.append({
        "anchor_id": anchor_id,
        "candidate_id": candidate_id,
        "anchor_word": a["word"],
        "anchor_ipa": a["ipa"],
        "candidate_word": c["word"],
        "candidate_ipa": c["ipa"],
        "relation_type": relation_type,
        "target_score": relation_score(relation_type),
        "anchor_split": a["split"],
        "candidate_split": c["split"],
    })


def synthetic_variants(tokens: Sequence[str]) -> list[tuple[str, str]]:
    clean = list(tokens)
    variants = []

    if "ə" in clean:
        v = [t for t in clean if t != "ə"]
        if v != clean:
            variants.append(("synthetic_schwa_drop", " ".join(v)))

    if "|" in clean:
        v = [t for t in clean if t != "|"]
        variants.append(("synthetic_boundary_drop", " ".join(v)))

    for src, tgt in [("ʁ", "r"), ("r", "ʁ")]:
        if src in clean:
            v = [tgt if t == src else t for t in clean]
            variants.append(("synthetic_r_variant", " ".join(v)))

    nasal_map = {
        "ɑ̃": "ɑ n",
        "ɔ̃": "ɔ n",
        "ɛ̃": "ɛ n",
        "œ̃": "œ n",
    }

    for nasal, repl in nasal_map.items():
        if nasal in clean:
            v = []
            for t in clean:
                if t == nasal:
                    v.extend(repl.split())
                else:
                    v.append(t)
            variants.append(("synthetic_nasal_variant", " ".join(v)))

    dedup = []
    seen = set()

    for label, ipa in variants:
        if ipa and ipa not in seen:
            seen.add(ipa)
            dedup.append((label, ipa))

    return dedup


def add_synthetic_relation(rows: list, anchor: dict, variant_ipa: str, relation_type: str) -> None:
    rows.append({
        "anchor_id": anchor["item_id"],
        "candidate_id": -1,
        "anchor_word": anchor["word"],
        "anchor_ipa": anchor["ipa"],
        "candidate_word": f"{anchor['word']}__{relation_type}",
        "candidate_ipa": variant_ipa,
        "relation_type": relation_type,
        "target_score": relation_score(relation_type),
        "anchor_split": anchor["split"],
        "candidate_split": anchor["split"],
    })


def is_false_negative(anchor: dict, cand: dict) -> bool:
    if anchor["ipa"] == cand["ipa"]:
        return True
    # edit distance <= 2 is already positive near-homophone territory.
    if edit_distance(anchor["tokens"], cand["tokens"], max_dist=2) <= 2:
        return True
    if anchor["cons"] and anchor["cons"] == cand["cons"]:
        return True
    if anchor["vowels"] and anchor["vowels"] == cand["vowels"]:
        return True
    return False


def init_worker(
    items,
    by_ipa,
    by_len,
    by_onset,
    by_suffix2,
    by_suffix3,
    by_rhyme,
    by_cons,
    by_vowels,
):
    global ITEMS, BY_IPA, BY_LEN, BY_ONSET, BY_SUFFIX2, BY_SUFFIX3, BY_RHYME, BY_CONS, BY_VOWELS

    ITEMS = items
    BY_IPA = by_ipa
    BY_LEN = by_len
    BY_ONSET = by_onset
    BY_SUFFIX2 = by_suffix2
    BY_SUFFIX3 = by_suffix3
    BY_RHYME = by_rhyme
    BY_CONS = by_cons
    BY_VOWELS = by_vowels


def build_rows_for_anchor(aid: int) -> list[dict]:
    a = ITEMS[aid]
    rows = []

    # Stabilizes geometry: exact identity is the center of the neighborhood.
    add_identity_relation(rows, aid)

    is_short = a["length"] <= 4

    exact = [x for x in BY_IPA.get(a["ipa"], []) if x != aid]
    for cid in deterministic_sample(exact, MAX_EXACT_PER_ANCHOR, f"{aid}:exact"):
        add_relation(rows, aid, cid, "exact_homophone")

    near_pool = set()
    for length in range(a["length"] - 2, a["length"] + 3):
        near_pool.update(BY_LEN.get(length, []))
    if a["onset"]:
        near_pool.update(BY_ONSET.get(a["onset"], []))

    near_pool = [x for x in near_pool if x != aid and ITEMS[x]["ipa"] != a["ipa"]]

    edit1 = []
    edit2 = []

    for cid in deterministic_sample(near_pool, NEAR_SCAN, f"{aid}:near_scan"):
        d = edit_distance(a["tokens"], ITEMS[cid]["tokens"], max_dist=2)
        if d == 1:
            edit1.append(cid)
        elif d == 2:
            edit2.append(cid)

    for cid in deterministic_sample(edit1, MAX_NEAR_PER_ANCHOR // 2, f"{aid}:edit1"):
        add_relation(rows, aid, cid, "near_homophone_edit1")

    for cid in deterministic_sample(edit2, MAX_NEAR_PER_ANCHOR // 2, f"{aid}:edit2"):
        add_relation(rows, aid, cid, "near_homophone_edit2")

    strong_rhyme = [
        x for x in BY_RHYME.get(a["rhyme"], [])
        if x != aid and ITEMS[x]["ipa"] != a["ipa"]
    ]
    strong_k = 1 if is_short else MAX_STRONG_RHYME_PER_ANCHOR

    for cid in deterministic_sample(strong_rhyme, strong_k, f"{aid}:strong_rhyme"):
        add_relation(rows, aid, cid, "strong_rhyme")

    weak_pool = set()
    if a["suffix2"]:
        weak_pool.update(BY_SUFFIX2.get(a["suffix2"], []))
    if a["suffix3"]:
        weak_pool.update(BY_SUFFIX3.get(a["suffix3"], []))

    weak_pool = [
        x for x in weak_pool
        if x != aid
        and ITEMS[x]["ipa"] != a["ipa"]
        and ITEMS[x]["rhyme"] != a["rhyme"]
    ]
    weak_k = 1 if is_short else MAX_WEAK_RHYME_PER_ANCHOR

    for cid in deterministic_sample(weak_pool, weak_k, f"{aid}:weak_rhyme"):
        add_relation(rows, aid, cid, "weak_rhyme")

    cons_pool = [
        x for x in BY_CONS.get(a["cons"], [])
        if x != aid and ITEMS[x]["ipa"] != a["ipa"] and ITEMS[x]["rhyme"] != a["rhyme"]
    ]

    for cid in deterministic_sample(cons_pool, MAX_SKELETON_PER_ANCHOR, f"{aid}:cons"):
        add_relation(rows, aid, cid, "consonant_skeleton")

    vowel_pool = [
        x for x in BY_VOWELS.get(a["vowels"], [])
        if x != aid and ITEMS[x]["ipa"] != a["ipa"] and ITEMS[x]["rhyme"] != a["rhyme"]
    ]

    for cid in deterministic_sample(vowel_pool, MAX_SKELETON_PER_ANCHOR, f"{aid}:vowels"):
        add_relation(rows, aid, cid, "vowel_skeleton")

    for relation_type, variant_ipa in synthetic_variants(a["tokens"])[:MAX_SYNTHETIC_PER_ANCHOR]:
        if variant_ipa != a["tokenized_ipa"]:
            add_synthetic_relation(rows, a, variant_ipa, relation_type)

    hard_pool = set()
    if a["onset"]:
        hard_pool.update(BY_ONSET.get(a["onset"], []))
    if a["rhyme"]:
        hard_pool.update(BY_RHYME.get(a["rhyme"], []))
    for length in range(a["length"] - 1, a["length"] + 2):
        hard_pool.update(BY_LEN.get(length, []))

    scored_hard = []

    for cid in deterministic_sample(list(hard_pool), HARD_SCAN, f"{aid}:hard_scan"):
        if cid == aid:
            continue

        cand = ITEMS[cid]

        if cand["split"] != a["split"]:
            continue

        if cand["ipa"] == a["ipa"]:
            continue

        d = edit_distance(a["tokens"], cand["tokens"], max_dist=2)
        if d <= 2:
            continue

        overlap = jaccard_char_ngrams(a["ipa"], cand["ipa"], 3)

        # Critical new negative: same rhyme ending does NOT imply equivalence.
        if a["rhyme"] and cand["rhyme"] == a["rhyme"]:
            scored_hard.append((0.95 + overlap, cid, "rhyme_hard_negative"))
            continue

        if is_false_negative(a, cand):
            continue

        if overlap >= 0.18 or cand["onset"] == a["onset"]:
            scored_hard.append((overlap, cid, "hard_negative"))

    scored_hard.sort(reverse=True)

    for _, cid, relation_type in scored_hard[:MAX_HARD_NEGATIVES_PER_ANCHOR]:
        add_relation(rows, aid, cid, relation_type)

    return rows


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    lower = {str(c).lower(): c for c in df.columns}

    word_col = (
        lower.get("word")
        or lower.get("surface")
        or lower.get("phrase")
        or lower.get("text")
        or lower.get("converted_phrase")
    )

    ipa_col = lower.get("ipa") or lower.get("phon") or lower.get("phonetic")

    if not word_col or not ipa_col:
        raise ValueError(f"Could not detect word/IPA columns. Columns are: {list(df.columns)}")

    return word_col, ipa_col


def write_tsv(df: pd.DataFrame, path: Path, desc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 250_000
    first = True

    with tqdm(total=len(df), desc=desc, unit="rows") as pbar:
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            df.iloc[start:end].to_csv(
                path,
                sep="\t",
                index=False,
                mode="w" if first else "a",
                header=first,
            )
            first = False
            pbar.update(end - start)


def clean_pairs(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print(f"[{name}] rows before: {len(df):,}", flush=True)

    with tqdm(total=6, desc=f"cleaning {name}", unit="step") as pbar:
        anchor_len = df["anchor_ipa"].astype(str).str.replace(" ", "", regex=False).str.len()
        pbar.update(1)

        candidate_len = df["candidate_ipa"].astype(str).str.replace(" ", "", regex=False).str.len()
        pbar.update(1)

        min_len = pd.concat([anchor_len, candidate_len], axis=1).min(axis=1)
        max_len = pd.concat([anchor_len, candidate_len], axis=1).max(axis=1)
        ratio = max_len / min_len.replace(0, pd.NA)
        pbar.update(1)

        rhyme = df["relation_type"].isin(["strong_rhyme", "weak_rhyme"])
        keep = pd.Series(True, index=df.index)
        keep &= ~(rhyme & (anchor_len < 3))
        keep &= ~(rhyme & (ratio > 2.5))
        pbar.update(1)

        out = df[keep].copy()
        pbar.update(1)

        # Keep these here so old raw files and new inline-clean outputs are consistent.
        score_updates = {
            "strong_rhyme": 0.38,
            "weak_rhyme": 0.18,
            "vowel_skeleton": 0.12,
            "consonant_skeleton": 0.20,
        }
        for relation_type, score in score_updates.items():
            out.loc[out["relation_type"] == relation_type, "target_score"] = score
        pbar.update(1)

    print(f"[{name}] rows after:  {len(out):,}", flush=True)
    print(f"[{name}] removed:     {len(df) - len(out):,}", flush=True)
    return out.reset_index(drop=True)


def main() -> None:
    out_dir = PHONETIC_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"phonetic input: {PHONETIC_INPUT_PATH}", flush=True)
    print(f"output dir:     {out_dir}", flush=True)
    print(f"workers:        {NUM_WORKERS}", flush=True)
    print(f"near scan:      {NEAR_SCAN}", flush=True)
    print(f"hard scan:      {HARD_SCAN}", flush=True)

    df = read_table(PHONETIC_INPUT_PATH).dropna()
    word_col, ipa_col = detect_columns(df)

    df = df[[word_col, ipa_col]].copy()
    df.columns = ["word", "ipa"]
    df["word"] = df["word"].map(normalize_surface)
    df["ipa"] = df["ipa"].map(normalize_ipa)
    df = df[(df["word"] != "") & (df["ipa"] != "")]
    df = df.drop_duplicates(["word", "ipa"]).reset_index(drop=True)

    print(f"loaded rows: {len(df):,}", flush=True)

    items = []
    by_ipa = defaultdict(list)
    by_len = defaultdict(list)
    by_onset = defaultdict(list)
    by_suffix2 = defaultdict(list)
    by_suffix3 = defaultdict(list)
    by_rhyme = defaultdict(list)
    by_cons = defaultdict(list)
    by_vowels = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="indexing IPA"):
        toks = ipa_tokens(row["ipa"])
        clean_toks = strip_boundaries(toks)

        if len(clean_toks) < 1:
            continue

        item = {
            "item_id": len(items),
            "word": row["word"],
            "ipa": row["ipa"],
            "tokens": toks,
            "tokenized_ipa": " ".join(toks),
            "length": len(clean_toks),
            "onset": onset_key(toks, 2),
            "suffix2": suffix_key(toks, 2),
            "suffix3": suffix_key(toks, 3),
            "rhyme": rhyme_key(toks),
            "cons": consonant_skeleton(toks),
            "vowels": vowel_skeleton(toks),
            "split": assign_split(row["ipa"]),
        }

        items.append(item)
        i = item["item_id"]

        by_ipa[item["ipa"]].append(i)
        by_len[item["length"]].append(i)

        if item["onset"]:
            by_onset[item["onset"]].append(i)
        if item["suffix2"]:
            by_suffix2[item["suffix2"]].append(i)
        if item["suffix3"]:
            by_suffix3[item["suffix3"]].append(i)
        if item["rhyme"]:
            by_rhyme[item["rhyme"]].append(i)
        if len(item["cons"]) >= 2:
            by_cons[item["cons"]].append(i)
        if len(item["vowels"]) >= 2:
            by_vowels[item["vowels"]].append(i)

    by_ipa_d = dict(by_ipa)
    by_len_d = dict(by_len)
    by_onset_d = dict(by_onset)
    by_suffix2_d = dict(by_suffix2)
    by_suffix3_d = dict(by_suffix3)
    by_rhyme_d = dict(by_rhyme)
    by_cons_d = dict(by_cons)
    by_vowels_d = dict(by_vowels)

    init_worker(
        items,
        by_ipa_d,
        by_len_d,
        by_onset_d,
        by_suffix2_d,
        by_suffix3_d,
        by_rhyme_d,
        by_cons_d,
        by_vowels_d,
    )

    print(f"indexed items: {len(items):,}", flush=True)

    rows = []
    anchor_ids = list(range(len(items)))

    if NUM_WORKERS <= 1:
        for aid in tqdm(anchor_ids, desc="building phonetic relevance"):
            rows.extend(build_rows_for_anchor(aid))
    else:
        # Use forked worker processes so the large ITEMS/index dictionaries are inherited
        # copy-on-write instead of pickled and sent to every worker. This is much faster
        # on PACE/RHEL Linux nodes than ProcessPoolExecutor with large initargs.
        mp_context_name = os.environ.get("PHONETIC_MP_CONTEXT", "fork")
        ctx = get_context(mp_context_name)
        chunksize = int(os.environ.get("PHONETIC_CHUNKSIZE", "50"))
        print(
            f"multiprocessing context: {mp_context_name}; chunksize: {chunksize}",
            flush=True,
        )
        with ctx.Pool(processes=NUM_WORKERS) as pool:
            for anchor_rows in tqdm(
                pool.imap_unordered(build_rows_for_anchor, anchor_ids, chunksize=chunksize),
                total=len(anchor_ids),
                desc=f"building phonetic relevance ({NUM_WORKERS} forked workers)",
            ):
                rows.extend(anchor_rows)

    relevance = pd.DataFrame(rows)

    if relevance.empty:
        raise RuntimeError("No relevance rows were created.")

    relevance = relevance.drop_duplicates(
        ["anchor_ipa", "candidate_ipa", "relation_type"]
    ).reset_index(drop=True)

    train_relevance = relevance[
        (relevance["anchor_split"] == "train")
        & (relevance["candidate_split"] == "train")
    ].copy()

    dev_relevance = relevance[relevance["anchor_split"] == "dev"].copy()
    test_relevance = relevance[relevance["anchor_split"] == "test"].copy()

    train_pairs = train_relevance[
        train_relevance["target_score"] >= TRAIN_PAIR_MIN_SCORE
    ][[
        "anchor_ipa",
        "candidate_ipa",
        "anchor_word",
        "candidate_word",
        "relation_type",
        "target_score",
    ]].copy()

    positives = train_relevance[train_relevance["target_score"] >= TRAIN_PAIR_MIN_SCORE].copy()
    negatives = train_relevance[train_relevance["target_score"] == 0.0].copy()

    neg_by_anchor = defaultdict(list)
    for r in negatives.itertuples(index=False):
        neg_by_anchor[r.anchor_ipa].append((r.candidate_ipa, r.candidate_word))

    triplets = []
    for r in positives.itertuples(index=False):
        negs = neg_by_anchor.get(r.anchor_ipa, [])
        if not negs:
            continue

        rng_seed = int(hashlib.md5(str(r.anchor_ipa).encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(rng_seed)
        neg_ipa, neg_word = rng.choice(negs)

        triplets.append({
            "anchor_ipa": r.anchor_ipa,
            "positive_ipa": r.candidate_ipa,
            "negative_ipa": neg_ipa,
            "anchor_word": r.anchor_word,
            "positive_word": r.candidate_word,
            "negative_word": neg_word,
            "relation_type": r.relation_type,
            "target_score": r.target_score,
        })

    triplets_df = pd.DataFrame(triplets)

    items_df = pd.DataFrame([
        {
            "item_id": x["item_id"],
            "word": x["word"],
            "ipa": x["ipa"],
            "tokenized_ipa": x["tokenized_ipa"],
            "length": x["length"],
            "onset": x["onset"],
            "suffix2": x["suffix2"],
            "suffix3": x["suffix3"],
            "rhyme": x["rhyme"],
            "consonant_skeleton": x["cons"],
            "vowel_skeleton": x["vowels"],
            "split": x["split"],
        }
        for x in items
    ])

    eval_relevance = pd.concat([dev_relevance, test_relevance], ignore_index=True)
    # Identity is a training stabilizer, not an eval target.
    eval_relevance = eval_relevance[eval_relevance["relation_type"] != "identity"].copy()

    eval_qrels = eval_relevance[[
        "anchor_id",
        "candidate_id",
        "anchor_ipa",
        "candidate_ipa",
        "relation_type",
        "target_score",
        "anchor_split",
    ]].copy()

    clean_train = clean_pairs(train_pairs, "train_pairs")

    trip_for_clean = triplets_df.rename(
        columns={
            "positive_ipa": "candidate_ipa",
            "positive_word": "candidate_word",
        }
    )
    clean_trip = clean_pairs(trip_for_clean, "train_triplets")
    clean_trip = clean_trip.rename(
        columns={
            "candidate_ipa": "positive_ipa",
            "candidate_word": "positive_word",
        }
    )

    clean_eval = clean_pairs(eval_qrels, "eval_qrels")

    report = pd.DataFrame([
        {"metric": "input_rows", "value": len(df)},
        {"metric": "items", "value": len(items_df)},
        {"metric": "all_relevance_rows", "value": len(relevance)},
        {"metric": "train_pairs_raw", "value": len(train_pairs)},
        {"metric": "train_pairs_clean", "value": len(clean_train)},
        {"metric": "train_triplets_raw", "value": len(triplets_df)},
        {"metric": "train_triplets_clean", "value": len(clean_trip)},
        {"metric": "eval_qrels_raw", "value": len(eval_qrels)},
        {"metric": "eval_qrels_clean", "value": len(clean_eval)},
        {"metric": "num_workers", "value": NUM_WORKERS},
        {"metric": "near_scan", "value": NEAR_SCAN},
        {"metric": "hard_scan", "value": HARD_SCAN},
        {"metric": "train_pair_min_score", "value": TRAIN_PAIR_MIN_SCORE},
    ])

    cleaning_report = pd.DataFrame([
        {
            "file": "train_pairs",
            "before": len(train_pairs),
            "after": len(clean_train),
            "removed": len(train_pairs) - len(clean_train),
        },
        {
            "file": "train_triplets",
            "before": len(triplets_df),
            "after": len(clean_trip),
            "removed": len(triplets_df) - len(clean_trip),
        },
        {
            "file": "eval_qrels",
            "before": len(eval_qrels),
            "after": len(clean_eval),
            "removed": len(eval_qrels) - len(clean_eval),
        },
    ])

    relation_counts = relevance["relation_type"].value_counts().reset_index()
    relation_counts.columns = ["relation_type", "count"]

    clean_counts = (
        clean_train["relation_type"]
        .value_counts()
        .rename_axis("relation_type")
        .reset_index(name="count")
    )

    print("writing outputs...", flush=True)
    write_tsv(items_df, out_dir / "phonetic_items.tsv", "writing phonetic_items.tsv")
    write_tsv(relevance, out_dir / "phonetic_relevance.tsv", "writing phonetic_relevance.tsv")
    write_tsv(train_pairs, out_dir / "train_pairs.tsv", "writing train_pairs.tsv")
    write_tsv(triplets_df, out_dir / "train_triplets.tsv", "writing train_triplets.tsv")
    write_tsv(eval_qrels, out_dir / "eval_qrels.tsv", "writing eval_qrels.tsv")

    write_tsv(clean_train, out_dir / "train_pairs_clean.tsv", "writing train_pairs_clean.tsv")
    write_tsv(clean_trip, out_dir / "train_triplets_clean.tsv", "writing train_triplets_clean.tsv")
    write_tsv(clean_eval, out_dir / "eval_qrels_clean.tsv", "writing eval_qrels_clean.tsv")

    write_tsv(report, out_dir / "dataset_report.tsv", "writing dataset_report.tsv")
    write_tsv(cleaning_report, out_dir / "cleaning_report.tsv", "writing cleaning_report.tsv")
    write_tsv(relation_counts, out_dir / "relation_counts.tsv", "writing relation_counts.tsv")
    write_tsv(clean_counts, out_dir / "clean_relation_counts.tsv", "writing clean_relation_counts.tsv")

    print()
    print(report.to_string(index=False))
    print()
    print("raw relation counts")
    print(relation_counts.to_string(index=False))
    print()
    print("clean train relation counts")
    print(clean_counts.to_string(index=False))
    print()
    print(f"wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
