from __future__ import annotations

import ast
import json
import os
import math
import re
import sys
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from data import load, load_all, save
from config import (
    translate_dir,
    phonetic_items_path,
    phonetic_index_path,
    phonetic_model_path,
    fasttext_fr_path,
)

try:
    from config import phonetic_embeddings_path  # optional fast path
except Exception:
    phonetic_embeddings_path = None

pd.options.mode.chained_assignment = None

DETACHED_DIAGNOSTICS_VERSION = "v18_CLEAN_OUTPUT_FLAG"

# ─────────────────────────────────────────────────────────────────────────────
# Runtime configuration
# ─────────────────────────────────────────────────────────────────────────────

VERBOSE = os.environ.get("RETRIEVAL_VERBOSE", "0") == "1"
RETRIEVAL_STAGE_TIMINGS = os.environ.get("RETRIEVAL_STAGE_TIMINGS", "1") == "1"
TRANSLATE_MODEL = os.environ.get("RETRIEVAL_TRANSLATE_MODEL", "gemini-3.1-pro-preview")
CHUNK_SIZE = int(os.environ.get("RETRIEVAL_CHUNK_SIZE", "100"))
# Safety rail: avoid accidentally launching many long chunk jobs while iterating.
# Set RETRIEVAL_MAX_CHUNKS_PER_CALL=0 for no limit.
MAX_CHUNKS_PER_CALL = int(os.environ.get("RETRIEVAL_MAX_CHUNKS_PER_CALL", "1"))

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DATA = ROOT / "data" / "retrieval"
EXPR_DIR = RETRIEVAL_DATA / "expressions"
OUTPUT_DIR = ROOT / "data" / "processed" / "retrieval"

EXPR_BANK_PATH = EXPR_DIR / "expression_bank.parquet"
EXPR_INDEX_PATH = EXPR_DIR / "expression_index.faiss"
MODEL_NAME = os.environ.get("RETRIEVAL_EMBED_MODEL", "BAAI/bge-m3")

# Candidate breadth. These defaults are intentionally modest for local Mac runs.
SEMANTIC_K = int(os.environ.get("RETRIEVAL_SEMANTIC_K", "10"))
LEXICAL_K = int(os.environ.get("RETRIEVAL_LEXICAL_K", "5"))
PHONETIC_K = int(os.environ.get("RETRIEVAL_PHONETIC_K", "10"))
PHONETIC_PROBE_BEAM = int(os.environ.get("RETRIEVAL_PHONETIC_PROBE_BEAM", "8"))
PHONETIC_NEIGHBORS_PER_PROBE = int(os.environ.get("RETRIEVAL_PHONETIC_NEIGHBORS_PER_PROBE", "10"))
MIN_EXPANSION_PHONETIC = float(os.environ.get("RETRIEVAL_MIN_EXPANSION_PHONETIC", "0.60"))
MIN_OPPOSITE_SEMANTIC = float(os.environ.get("RETRIEVAL_MIN_OPPOSITE_SEMANTIC", "0.38"))
MAX_EXPANSION_BRIDGES = int(os.environ.get("RETRIEVAL_MAX_EXPANSION_BRIDGES", "24"))
SIDE_SEMANTIC_K = int(os.environ.get("RETRIEVAL_SIDE_SEMANTIC_K", "28"))
SIDE_LEVEL2_K = int(os.environ.get("RETRIEVAL_SIDE_LEVEL2_K", "8"))
MAX_IPA_CANDIDATES_PER_SIDE = int(os.environ.get("RETRIEVAL_MAX_IPA_CANDIDATES_PER_SIDE", "48"))
MAX_BRIDGES = int(os.environ.get("RETRIEVAL_MAX_BRIDGES", "8"))
MAX_IDENTITY_BRIDGE_FRACTION = float(os.environ.get("RETRIEVAL_MAX_IDENTITY_BRIDGE_FRACTION", "0.35"))
MAX_BRIDGES_PER_SURFACE = int(os.environ.get("RETRIEVAL_MAX_BRIDGES_PER_SURFACE", "2"))
MIN_PAIR_PHONETIC = float(os.environ.get("RETRIEVAL_MIN_PAIR_PHONETIC", "0.45"))
STRONG_BRIDGE_THRESHOLD = float(os.environ.get("RETRIEVAL_STRONG_BRIDGE_THRESHOLD", "0.74"))

# LLM-ready optimization: generate broadly, but run expensive semantic scoring
# only after cheap phonetic/structural/diversity pruning.  This keeps recall
# while avoiding O(candidate explosion) bridge scoring.
MAX_SEMANTIC_SCORED_EXPANSION_CANDIDATES = int(os.environ.get("RETRIEVAL_MAX_SEMANTIC_SCORED_EXPANSION_CANDIDATES", "40"))
MAX_CHEAP_EXPANSION_CANDIDATES = int(os.environ.get("RETRIEVAL_MAX_CHEAP_EXPANSION_CANDIDATES", "80"))
MAX_DIRECT_PAIR_CANDIDATES = int(os.environ.get("RETRIEVAL_MAX_DIRECT_PAIR_CANDIDATES", "40"))
A1B1_PER_SEED_SEMANTIC_ENABLED = os.environ.get("RETRIEVAL_A1B1_PER_SEED_SEMANTIC", "1") == "1"
A1B1_PER_SEED_SEMANTIC_K = int(os.environ.get("RETRIEVAL_A1B1_PER_SEED_SEMANTIC_K", "8"))
A1B1_PER_SEED_LIMIT = int(os.environ.get("RETRIEVAL_A1B1_PER_SEED_LIMIT", "20"))
LLM_JUDGE_CANDIDATE_LIMIT = int(os.environ.get("RETRIEVAL_LLM_JUDGE_CANDIDATE_LIMIT", "15"))

# Bucketized retrieval regimes. These are stable machine IDs used for
# ownership, diagnostics, ranking, and export. Display wording must not drive
# logic. The four regimes are executed independently; no merged A* x B* matrix
# is used for primary bucketized bridge mining.
RETRIEVAL_BUCKET_ORDER = [
    "A0_B0",
    "A1_B0",
    "B1_A0",
    "A1_B1",
    "A2_B01_DETACHED",
    "B2_A01_DETACHED",
]
RETRIEVAL_BUCKET_REGIMES = [
    {"bucket_id": "A0_B0", "source_pool": "A0", "target_pool": "B0"},
    {"bucket_id": "A1_B0", "source_pool": "A1", "target_pool": "B0"},
    {"bucket_id": "B1_A0", "source_pool": "B1", "target_pool": "A0"},
    {"bucket_id": "A1_B1", "source_pool": "A1", "target_pool": "B1"},
]
RETRIEVAL_BUCKET_LABELS = {
    "A0_B0": "phonetic_matches(A0, B0)",
    "A1_B0": "phonetic_matches(A1, B0)",
    "B1_A0": "phonetic_matches(B1, A0)",
    "A1_B1": "phonetic_matches(A1, B1)",
    "A2_B01_DETACHED": "detached_sound_neighbors(B0+B1) as A2",
    "B2_A01_DETACHED": "detached_sound_neighbors(A0+A1) as B2",
}
RETRIEVAL_BUCKET_INDEX = {bucket_id: i for i, bucket_id in enumerate(RETRIEVAL_BUCKET_ORDER)}


# JOKER-style affordance ranking: retrieval is a pre-judge, not the final judge.
# Prioritize phonetic collision + natural French + surprise/recoverability; treat
# semantic resemblance to the French semantic domain meanings as a soft bonus, not the objective.
MIN_LLM_CANDIDATE_PHONETIC_STRICT = float(os.environ.get("RETRIEVAL_MIN_LLM_CANDIDATE_PHONETIC_STRICT", "0.72"))
MIN_LLM_CANDIDATE_PHONETIC_BROAD = float(os.environ.get("RETRIEVAL_MIN_LLM_CANDIDATE_PHONETIC_BROAD", "0.80"))
MIN_LLM_CANDIDATE_NATURALNESS = float(os.environ.get("RETRIEVAL_MIN_LLM_CANDIDATE_NATURALNESS", "0.18"))
MIN_LLM_CANDIDATE_PIVOTABILITY = float(os.environ.get("RETRIEVAL_MIN_LLM_CANDIDATE_PIVOTABILITY", "0.32"))
MAX_GENERATOR_AFFORDANCES = int(os.environ.get("RETRIEVAL_MAX_GENERATOR_AFFORDANCES", "8"))
MAX_AFFORDANCES_PER_BUCKET = int(os.environ.get("RETRIEVAL_MAX_AFFORDANCES_PER_BUCKET", "4"))

# Detached recovery buckets. These always run after the existing bucketized and
# expansion routes. They keep exactly one semantic anchor side and generate the
# other side from broad phonetic neighbors, with no semantic check to the missing
# original domain. The goal is recall through new collision territory, not looser
# junk filtering.
DETACHED_BUCKETS_ENABLED = os.environ.get("RETRIEVAL_DETACHED_BUCKETS", "1") == "1"
DETACHED_NEIGHBORS_PER_ANCHOR = int(os.environ.get("RETRIEVAL_DETACHED_NEIGHBORS_PER_ANCHOR", "32"))
DETACHED_MIN_PHONETIC = float(os.environ.get("RETRIEVAL_DETACHED_MIN_PHONETIC", "0.80"))
DETACHED_MAX_ANCHOR_SEMANTIC_SIM = float(os.environ.get("RETRIEVAL_DETACHED_MAX_ANCHOR_SEMANTIC_SIM", "0.75"))
DETACHED_MIN_NATURALNESS = float(os.environ.get("RETRIEVAL_DETACHED_MIN_NATURALNESS", str(MIN_LLM_CANDIDATE_NATURALNESS)))
DETACHED_MIN_FREE_RECOGNIZABILITY = float(os.environ.get("RETRIEVAL_DETACHED_MIN_FREE_RECOGNIZABILITY", "0.32"))
DETACHED_MAX_PER_ANCHOR = int(os.environ.get("RETRIEVAL_DETACHED_MAX_PER_ANCHOR", "3"))
DETACHED_MAX_PER_BUCKET = int(os.environ.get("RETRIEVAL_DETACHED_MAX_PER_BUCKET", "24"))
PREFINAL_BRIDGE_CANDIDATE_LIMIT = int(os.environ.get("RETRIEVAL_PREFINAL_BRIDGE_CANDIDATE_LIMIT", "80"))


# Fast bridge mining mode for LLM-judge pipeline.  Retrieval should produce
# broad clean affordances quickly; the LLM will do expensive quality judgment.
FAST_BRIDGE_MINING = os.environ.get("RETRIEVAL_FAST_BRIDGE_MINING", "1") == "1"
SKIP_BRIDGE_OPPOSITE_SEMANTIC = os.environ.get("RETRIEVAL_SKIP_BRIDGE_OPPOSITE_SEMANTIC", "1") == "1"
BRIDGE_USE_LEVEL2 = os.environ.get("RETRIEVAL_BRIDGE_USE_LEVEL2", "0") == "1"
RETRIEVAL_REUSE_ROW_SEMANTIC_FOR_BRIDGES = os.environ.get("RETRIEVAL_REUSE_ROW_SEMANTIC_FOR_BRIDGES", "1") == "1"


# Quality tuning: reduce boring morphology echoes and reward novel lexical collisions.
SAME_ROOT_PENALTY = float(os.environ.get("RETRIEVAL_SAME_ROOT_PENALTY", "0.18"))
DIFFERENT_ROOT_BONUS = float(os.environ.get("RETRIEVAL_DIFFERENT_ROOT_BONUS", "0.08"))
COMMON_EXPRESSION_BONUS = float(os.environ.get("RETRIEVAL_COMMON_EXPRESSION_BONUS", "0.04"))
CROSS_SIDE_COLLISION_BONUS = float(os.environ.get("RETRIEVAL_CROSS_SIDE_COLLISION_BONUS", "0.06"))
MAX_BRIDGES_PER_ROOT = int(os.environ.get("RETRIEVAL_MAX_BRIDGES_PER_ROOT", "2"))

RETRIEVAL_DEBUG_PACKS = os.environ.get("RETRIEVAL_DEBUG_PACKS", "0") == "1"
RETRIEVAL_OUTPUT_DEBUG = os.environ.get("RETRIEVAL_OUTPUT_DEBUG", "0") == "1"
RETRIEVAL_SAVE_TRACES = os.environ.get("RETRIEVAL_SAVE_TRACES", "0") == "1"
USE_SPACY_LEMMAS = os.environ.get("RETRIEVAL_USE_SPACY_LEMMAS", "1") == "1"
SPACY_MODEL = os.environ.get("RETRIEVAL_SPACY_MODEL", "fr_core_news_md")
_NLP_FR = None
_SPACY_WARNED = False

# Optional FastText branch. This does not replace BGE; it adds a bounded,
# deterministic word-level semantic drift beam for Low-style bridge mining.
# Important: FastText is intentionally NOT recursive. It expands selected seed
# terms once, with strict budgets, so runtime is predictable.
USE_FASTTEXT = os.environ.get("RETRIEVAL_USE_FASTTEXT", "0") == "1"
FASTTEXT_MODEL_PATH = os.environ.get("RETRIEVAL_FASTTEXT_MODEL", fasttext_fr_path)
FASTTEXT_K = int(os.environ.get("RETRIEVAL_FASTTEXT_K", "12"))
FASTTEXT_SEED_LIMIT = int(os.environ.get("RETRIEVAL_FASTTEXT_SEED_LIMIT", "10"))
FASTTEXT_MIN_SIM = float(os.environ.get("RETRIEVAL_FASTTEXT_MIN_SIM", "0.45"))
FASTTEXT_LEVEL1_PENALTY = float(os.environ.get("RETRIEVAL_FASTTEXT_LEVEL1_PENALTY", "0.86"))
FASTTEXT_MAX_CANDIDATES_PER_SIDE = int(os.environ.get("RETRIEVAL_FASTTEXT_MAX_CANDIDATES_PER_SIDE", "48"))
FASTTEXT_MAX_TOKENS_PER_SEED = int(os.environ.get("RETRIEVAL_FASTTEXT_MAX_TOKENS_PER_SEED", "2"))
_FASTTEXT_WARNED = False

REQUIRED_COLUMNS = [
    "first_meaning_fr",
    "second_meaning_fr",
]


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized semantic embedding cache
# ─────────────────────────────────────────────────────────────────────────────

_SURFACE_EMBED_CACHE: dict[str, np.ndarray] = {}

def cached_batch_embed(model, texts):
    texts = [(t or "").strip() for t in texts]

    missing = [t for t in texts if t not in _SURFACE_EMBED_CACHE]

    if missing:
        vecs = model.encode(
            missing,
            batch_size=64,
            normalize_embeddings=True,
        )

        for t, v in zip(missing, vecs):
            _SURFACE_EMBED_CACHE[t] = v

    return np.stack([
        _SURFACE_EMBED_CACHE[t]
        for t in texts
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────


def log(*args: Any) -> None:
    if VERBOSE:
        print(*args)


def clean(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except TypeError:
        pass
    return str(x).strip()


def short(x: Any, n: int = 500) -> str:
    return clean(x)[:n]


@lru_cache(maxsize=500_000)
def _norm_text_cached(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'").replace("ʼ", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def norm_text(x: Any) -> str:
    return _norm_text_cached(clean(x))


@lru_cache(maxsize=500_000)
def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(ch) != "Mn"
    )


def surface_key(x: Any) -> str:
    """Accent/punctuation-insensitive surface key for duplicate control."""
    text = strip_accents(norm_text(x))
    text = re.sub(r"^(un|une|le|la|les|l'|des|du|de la|de l')\s+", "", text)
    text = re.sub(r"[-’'\s]+", "", text)
    return text


def _get_spacy_fr():
    """Lazy-load spaCy only if available. Retrieval should still run without it."""
    global _NLP_FR, _SPACY_WARNED
    if not USE_SPACY_LEMMAS:
        return None
    if _NLP_FR is not None:
        return _NLP_FR
    try:
        import spacy
        _NLP_FR = spacy.load(SPACY_MODEL, disable=["parser", "ner", "textcat"])
        return _NLP_FR
    except Exception as e:
        if not _SPACY_WARNED:
            log(
                f"WARNING: spaCy French lemmatizer unavailable ({e}). "
                "Falling back to rough suffix normalization. "
                "Install with: python -m pip install spacy && "
                "python -m spacy download fr_core_news_md"
            )
            _SPACY_WARNED = True
        return None


@lru_cache(maxsize=200_000)
def _spacy_lemma_key_cached(text: str) -> str:
    nlp = _get_spacy_fr()
    if nlp is None:
        return ""
    doc = nlp(text)
    lemmas: list[str] = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if tok.is_stop and tok.text.lower() in {"un", "une", "le", "la", "les", "l", "des", "du", "de"}:
            continue
        lemma = strip_accents((tok.lemma_ or tok.text).lower().strip())
        lemma = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ]+", "", lemma)
        if lemma:
            lemmas.append(lemma)
    return "".join(lemmas)


def crude_fr_root_key(x: Any) -> str:
    """Very conservative French-ish root key for quality filtering.

    This is not used for semantic retrieval. It only prevents top bridge lists
    from being dominated by boring morphology variants such as zéro/zéros,
    acte/actes, courbé/courbés, brûlaient/brûlait, or accent-only spellings.
    """
    text = surface_key(x)
    if not text:
        return ""

    # Normalize common orthographic variants after accent stripping.
    text = text.replace("oe", "œ")

    # Remove obvious silent plural before participle/adjective/conjugation endings.
    root = text
    if len(root) > 4 and root.endswith(("s", "x")):
        root = root[:-1]

    # Ordered from longest/specific to shortest/general. One pass is enough for
    # broad same-root penalties; structural rejection uses structurally_trivial_variant().
    suffixes = (
        "issements", "issement", "issantes", "issante", "issants", "issant",
        "eraient", "erait", "erais", "erons", "erez", "eront",
        "aient", "ait", "ais", "antes", "ante", "ants", "ant", "ent",
        "ions", "iez", "ons", "ez",
        "euses", "euse", "eaux", "aux",
        "ives", "ive", "ifs", "if",
        "ees", "ee", "es", "e",
    )
    for suffix in suffixes:
        if len(root) > len(suffix) + 3 and root.endswith(suffix):
            root = root[: -len(suffix)]
            break

    return root


def rough_lemma_key(x: Any) -> str:
    """Primary lemma key for display/debugging.

    Prefer spaCy when available, but do not trust it as the only signal: spaCy
    can leave rare forms, accent variants, and generated fragments uncollapsed.
    same_root() below combines spaCy with crude_fr_root_key().
    """
    text = norm_text(x)
    if not text:
        return ""
    lemma_key = _spacy_lemma_key_cached(text)
    return lemma_key or crude_fr_root_key(text)


def same_root(a: Any, b: Any) -> bool:
    """Return true when two surfaces are probably the same lexical item.

    This deliberately catches trivial variants so retrieval does not rank them
    as creative bridges. It still allows true identity/polysemy candidates, but
    they are labeled and capped elsewhere instead of filling the top-k list.
    """
    if not clean(a) or not clean(b):
        return False
    if surface_key(a) == surface_key(b):
        return True

    spa = _spacy_lemma_key_cached(norm_text(a))
    spb = _spacy_lemma_key_cached(norm_text(b))
    if spa and spb and spa == spb:
        return True

    ca = crude_fr_root_key(a)
    cb = crude_fr_root_key(b)
    if ca and cb and ca == cb:
        return True

    # Last conservative guard: one root is a short inflectional extension of the other.
    if ca and cb and min(len(ca), len(cb)) >= 4:
        if ca.startswith(cb) or cb.startswith(ca):
            if abs(len(ca) - len(cb)) <= 2:
                return True
    return False


def plural_surface_key(x: Any) -> str:
    """Surface key with only obvious plural/gender/accent noise removed.

    This is intentionally less aggressive than crude_fr_root_key(). It catches
    boring variants such as séduisant/séduisants and brûlant/brulant, while
    preserving useful homophones like couvant/couvent or suie/suit.
    """
    key = surface_key(x)
    if not key:
        return ""
    # common silent plural markers
    if len(key) > 4 and key.endswith(("s", "x")):
        key = key[:-1]
    # feminine/plural adjective endings, conservative
    for suf in ("ees", "ee", "es"):
        if len(key) > len(suf) + 4 and key.endswith(suf):
            key = key[: -len(suf)]
            break
    return key


def same_surface_family(a: Any, b: Any) -> bool:
    """True for accent/plural/gender-only variants of the same surface."""
    if not clean(a) or not clean(b):
        return False
    if surface_key(a) == surface_key(b):
        return True
    if plural_surface_key(a) and plural_surface_key(a) == plural_surface_key(b):
        return True
    na, nb = surface_key(a), surface_key(b)
    if na and nb:
        # one is the other plus a short silent inflectional suffix
        for suf in ("s", "x", "es", "e"):
            if len(na) > 4 and na == nb + suf:
                return True
            if len(nb) > 4 and nb == na + suf:
                return True
    return False


def structurally_trivial_variant(a: Any, b: Any) -> bool:
    """Reject-only predicate for non-creative bridge candidates.

    This is stricter and safer than same_root(): it rejects exact/plural/accent
    variants and same spaCy lemma, but does NOT reject every shared crude root.
    That preserves potentially good puns like couvant/couvent.
    """
    if not clean(a) or not clean(b):
        return False
    if same_surface_family(a, b):
        return True

    spa_a = _spacy_lemma_key_cached(norm_text(a))
    spa_b = _spacy_lemma_key_cached(norm_text(b))
    if spa_a and spa_b and spa_a == spa_b:
        return True

    # Catch cases where crude roots match only because one surface is a trivial
    # extension of the other. Avoid rejecting equal-length alternations such as
    # couvant/couvent.
    ca, cb = crude_fr_root_key(a), crude_fr_root_key(b)
    sa, sb = surface_key(a), surface_key(b)
    if ca and cb and ca == cb and min(len(sa), len(sb)) >= 4:
        if sa.startswith(sb) or sb.startswith(sa):
            if abs(len(sa) - len(sb)) <= 3:
                return True
    return False


def trivial_inflection_related(a: Any, b: Any) -> bool:
    return bool(structurally_trivial_variant(a, b) and norm_text(a) != norm_text(b))


def lexical_novelty_bonus(a: Any, b: Any) -> float:
    """Reward genuine different-root collisions; penalize trivial variants."""
    if not clean(a) or not clean(b):
        return 0.0
    if norm_text(a) == norm_text(b):
        return -0.04
    if same_root(a, b):
        return -SAME_ROOT_PENALTY
    return DIFFERENT_ROOT_BONUS


def commonness_bonus_from_item(item: dict[str, Any]) -> float:
    """Tiny prior for candidates likely to be recognizable French words/phrases."""
    source = norm_text(item.get("source", item.get("candidate_source", "")))
    word = norm_text(item.get("surface", item.get("candidate", item.get("word", ""))))
    bonus = 0.0
    if "wiktionary" in source or "parseme" in source:
        bonus += COMMON_EXPRESSION_BONUS
    if "opensubtitles" in source:
        bonus += COMMON_EXPRESSION_BONUS * 0.5
    wc = len(word.split())
    if 1 <= wc <= 5:
        bonus += 0.015
    return min(bonus, 0.08)


def parse_listish(x: Any) -> list[str]:
    """Parse columns that may contain Python-list strings, JSON lists, or plain text."""
    if x is None:
        return []
    if isinstance(x, list):
        return [clean(v) for v in x if clean(v)]
    if isinstance(x, tuple):
        return [clean(v) for v in x if clean(v)]

    text = clean(x)
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        for parser in (ast.literal_eval, json.loads):
            try:
                value = parser(text)
                if isinstance(value, list):
                    return [clean(v) for v in value if clean(v)]
            except Exception:
                pass

    # Conservative fallback: treat semicolon/comma separated strings as lists.
    parts = re.split(r"\s*[;,]\s*", text)
    return [p for p in (clean(p) for p in parts) if p]


def unique_keep_order(values: Iterable[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = norm_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(clean(value))
        if limit is not None and len(out) >= limit:
            break
    return out


def bridge_surface_pair(b: dict[str, Any]) -> tuple[str, str]:
    """Stable surface pair for direct and expansion bridges."""
    left = clean(
        b.get("left_text")
        or b.get("a_surface")
        or b.get("sound_source")
        or b.get("source_surface")
        or ""
    )
    right = clean(
        b.get("right_text")
        or b.get("b_surface")
        or b.get("candidate")
        or b.get("candidate_surface")
        or ""
    )
    return left, right


def bridge_ipa_pair(b: dict[str, Any]) -> tuple[str, str]:
    """Stable IPA pair for direct and expansion bridges."""
    left_ipa = clean(
        b.get("left_ipa")
        or b.get("a_ipa")
        or b.get("sound_source_ipa")
        or b.get("source_ipa")
        or ""
    )
    right_ipa = clean(
        b.get("right_ipa")
        or b.get("b_ipa")
        or b.get("candidate_ipa")
        or ""
    )
    return left_ipa, right_ipa


def retrieval_pair_key_for_bridge(b: dict[str, Any]) -> str:
    """Order-insensitive bridge identity for first-bucket ownership."""
    left, right = bridge_surface_pair(b)
    left_ipa, right_ipa = bridge_ipa_pair(b)
    side1 = (surface_key(left), left_ipa)
    side2 = (surface_key(right), right_ipa)
    return json.dumps(sorted([side1, side2]), ensure_ascii=False, separators=(",", ":"))


def retrieval_pool_member_key(surface: Any, ipa: Any) -> tuple[str, str]:
    """Exact-ish pool membership key for bucketizing expansion bridges.

    Expansion bucket ownership must be based on whether the generated phonetic
    neighbor is actually present in one of the explicit A0/B0/A1/B1 pools.
    Surface alone can be ambiguous, and IPA alone is too broad, so use both.
    """
    return (surface_key(surface), clean(ipa))


def retrieval_pool_index(pool_map: dict[str, list[dict[str, Any]]]) -> dict[str, set[tuple[str, str]]]:
    index: dict[str, set[tuple[str, str]]] = {}
    for pool_name, items in (pool_map or {}).items():
        keys: set[tuple[str, str]] = set()
        for item in items or []:
            surface = clean(item.get("surface", item.get("word", item.get("text", ""))))
            ipa = clean(item.get("ipa", item.get("candidate_ipa", "")))
            key = retrieval_pool_member_key(surface, ipa)
            if key[0] and key[1]:
                keys.add(key)
        index[pool_name] = keys
    return index


def retrieval_bucket_for_pool_pair(pool_a: str, pool_b: str) -> str:
    """Return the bucket ID for an unordered explicit pool pair, or ''."""
    pair = frozenset([clean(pool_a), clean(pool_b)])
    if pair == frozenset(["A0", "B0"]):
        return "A0_B0"
    if pair == frozenset(["A1", "B0"]):
        return "A1_B0"
    if pair == frozenset(["B1", "A0"]):
        return "B1_A0"
    if pair == frozenset(["A1", "B1"]):
        return "A1_B1"
    return ""


def bucket_rank_sort_key(b: dict[str, Any]) -> tuple:
    return (
        affordance_stage_rank(b.get("affordance_stage", ""), b.get("bridge_type", "")),
        -float(b.get("llm_priority_score", b.get("bridge_score", 0.0)) or 0.0),
        -float(b.get("phonetic_score", 0.0) or 0.0),
        surface_key(bridge_surface_pair(b)[0]),
        surface_key(bridge_surface_pair(b)[1]),
    )


def retrieval_bucket_counts(bridges: list[dict[str, Any]]) -> dict[str, int]:
    counts = {bucket_id: 0 for bucket_id in RETRIEVAL_BUCKET_ORDER}
    for b in bridges or []:
        bucket_id = clean(b.get("retrieval_bucket", ""))
        if bucket_id in counts:
            counts[bucket_id] += 1
    return counts


def order_bridges_by_retrieval_bucket(bridges: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {bucket_id: [] for bucket_id in RETRIEVAL_BUCKET_ORDER}
    extras: list[dict[str, Any]] = []
    for b in bridges or []:
        bucket_id = clean(b.get("retrieval_bucket", ""))
        if bucket_id in grouped:
            grouped[bucket_id].append(b)
        else:
            extras.append(b)

    out: list[dict[str, Any]] = []
    for bucket_id in RETRIEVAL_BUCKET_ORDER:
        bucket_items = sorted(grouped[bucket_id], key=bucket_rank_sort_key)
        for rank, item in enumerate(bucket_items, 1):
            nb = dict(item)
            nb["retrieval_bucket"] = bucket_id
            nb["retrieval_bucket_rank"] = int(nb.get("retrieval_bucket_rank", rank) or rank)
            nb["retrieval_pair_key"] = nb.get("retrieval_pair_key") or retrieval_pair_key_for_bridge(nb)
            out.append(nb)
            if limit is not None and len(out) >= limit:
                return out
    for item in sorted(extras, key=bucket_rank_sort_key):
        out.append(item)
        if limit is not None and len(out) >= limit:
            return out
    return out


def group_exported_affordances_by_retrieval_bucket(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {bucket_id: [] for bucket_id in RETRIEVAL_BUCKET_ORDER}
    for item in items or []:
        bucket_id = clean(item.get("retrieval_bucket", ""))
        if bucket_id in grouped:
            grouped[bucket_id].append(item)
    return grouped


def collapse_dicts_by_root(
    items: list[dict[str, Any]],
    surface_key_name: str = "surface",
    score_keys: tuple[str, ...] = ("semantic_score", "final_score", "phonetic_score", "quality_score"),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Collapse trivial lemma/root variants before expensive phonetic ranking.

    This is deliberately upstream of bridge scoring. Penalizing after scoring was
    not enough: exact homophone/plural variants still consumed top-k budget.
    We keep one representative per root, preferring lower Low-level distance and
    higher available score fields.
    """
    best: dict[str, dict[str, Any]] = {}
    for item in items:
        surface = clean(item.get(surface_key_name, item.get("word", item.get("text", ""))))
        if not surface:
            continue
        root = rough_lemma_key(surface) or surface_key(surface) or norm_text(surface)
        if not root:
            continue
        current = best.get(root)
        if current is None:
            best[root] = item
            continue

        def rank(x: dict[str, Any]) -> tuple:
            level = int(x.get("level", x.get("source_level", 9)) or 9)
            scores = tuple(float(x.get(k, 0.0) or 0.0) for k in score_keys)
            # Prefer cleaner canonical surfaces over obvious plural/accent-only variants.
            text = clean(x.get(surface_key_name, x.get("word", x.get("text", ""))))
            sk = surface_key(text)
            plural_penalty = 1 if sk.endswith(("s", "x")) else 0
            accent_penalty = 1 if strip_accents(norm_text(text)) == norm_text(text) and norm_text(text) != text.lower().strip() else 0
            # Prefer shorter singular-looking forms, then higher score.
            return (-level, *scores, -plural_penalty, -accent_penalty, -len(text))

        if rank(item) > rank(current):
            best[root] = item

    out = list(best.values())
    out.sort(key=lambda x: (int(x.get("level", x.get("source_level", 9)) or 9), -max(float(x.get(k, 0.0) or 0.0) for k in score_keys)))
    return out[:limit] if limit is not None else out


def phonetic_family_key(word: Any, ipa: Any) -> tuple[str, str]:
    """Canonical key for phonetic candidate dedupe.

    This runs before candidates enter affordance pools or bridge ranking.  It
    collapses boring variants that share both a surface family/root and IPA
    (brûlant/brulant, fumant/fumants, séduisant/séduisants), while still
    allowing useful same-IPA different-surface homophones such as suie/suit or
    très/trait because their surface family/root differs.
    """
    w = clean(word)
    p = clean(ipa)
    if not w and not p:
        return ("", "")
    root = plural_surface_key(w) or crude_fr_root_key(w) or rough_lemma_key(w) or surface_key(w) or norm_text(w)
    return (root, p)


def collapse_phonetic_records_by_family(
    records: list[dict[str, Any]],
    surface_key_name: str = "word",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Keep one representative per (surface-family/root, IPA) phonetic family.

    The representative is chosen by final_score/phonetic_score, then by a
    small singular/accent/canonical preference.  This is stricter than generic
    semantic root collapse and is applied globally to phonetic candidate pools.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}

    def candidate_rank(item: dict[str, Any]) -> tuple:
        text = clean(item.get(surface_key_name, item.get("surface", item.get("candidate", ""))))
        sk = surface_key(text)
        # Prefer singular-looking, accented/canonical, shorter forms if scores tie.
        plural_penalty = 1 if sk.endswith(("s", "x")) else 0
        # If the original contains accents, prefer it over accentless spelling.
        accent_bonus = 1 if strip_accents(norm_text(text)) != norm_text(text) else 0
        return (
            float(item.get("final_score", 0.0) or 0.0),
            float(item.get("phonetic_score", 0.0) or 0.0),
            -plural_penalty,
            accent_bonus,
            -len(text),
        )

    for item in records:
        word = clean(item.get(surface_key_name, item.get("surface", item.get("candidate", ""))))
        ipa = clean(item.get("ipa", item.get("candidate_ipa", "")))
        key = phonetic_family_key(word, ipa)
        if not key[0] and not key[1]:
            continue
        current = best.get(key)
        if current is None or candidate_rank(item) > candidate_rank(current):
            best[key] = item

    out = list(best.values())
    out.sort(key=lambda x: (float(x.get("final_score", 0.0) or 0.0), float(x.get("phonetic_score", 0.0) or 0.0)), reverse=True)
    return out[:limit] if limit is not None else out


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def side_terms(row: pd.Series) -> tuple[list[str], list[str]]:
    return parse_listish(row.get("first_meaning_fr", [])), parse_listish(row.get("second_meaning_fr", []))


def build_semantic_query(row: pd.Series) -> str:
    a_terms, b_terms = side_terms(row)
    return " ".join(unique_keep_order(a_terms + b_terms))


def build_lexical_query(row: pd.Series) -> str:
    a_terms, b_terms = side_terms(row)
    return " ".join(unique_keep_order(a_terms + b_terms))


def semantic_side_channel(term: str, side: str) -> str:
    """Stable semantic-cache channel for one French launch term."""
    key = surface_key(term) or norm_text(term)
    return f"semantic_{side}_term:{key}"


def semantic_side_requests(terms: list[str], side: str, top_k: int = SEMANTIC_K) -> list[tuple[str, int, str]]:
    """Build one semantic request per French list item.

    Retrieval launches from the two French semantic-domain lists only.  The
    non-list source columns are not retrieval inputs.  Channels are term-stable
    so row-range prewarm and retrieve_row hit the same semantic-search cache.
    """
    requests: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for term in unique_keep_order(terms):
        t = clean(term)
        if not t:
            continue
        channel = semantic_side_channel(t, side)
        if channel in seen:
            continue
        seen.add(channel)
        requests.append((t, top_k, channel))
    return requests


def unique_keep_order_dicts_by_surface(items: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    """Keep first dict per normalized surface, preserving generator/export shape."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items or []:
        surface = clean(item.get("surface", item.get("text", "")))
        key = surface_key(surface) or norm_text(surface)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
        if limit is not None and len(out) >= limit:
            break
    return out


def merge_semantic_side_results(
    semantic_many: dict[str, list[dict[str, Any]]],
    channels: list[str],
) -> list[dict[str, Any]]:
    """Merge per-term semantic results by surface, keeping the best metadata."""
    best: dict[str, dict[str, Any]] = {}
    for channel in channels:
        for item in semantic_many.get(channel, []) or []:
            surface = clean(item.get("surface", item.get("text", "")))
            if not surface or lexically_bad_candidate_surface(surface):
                continue
            key = surface_key(surface) or norm_text(surface)
            current = best.get(key)
            candidate = dict(item)
            candidate["semantic_launch_channel"] = channel
            if current is None or float(candidate.get("score", 0.0) or 0.0) > float(current.get("score", 0.0) or 0.0):
                best[key] = candidate
    out = list(best.values())
    out.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return out



# A deliberately small recognizability prior for common French pun pivots that
# may come only from the phonetic index and therefore lack source/frequency
# metadata.  This is not a broad whitelist: it only rescues familiar, ordinary
# words that are plausible native-speaker joke pivots.
_COMMON_FRENCH_PUN_PIVOTS = {
    "air", "ère", "mère", "mer", "maire", "vert", "ver", "vers", "verre",
    "conte", "comte", "compte", "foi", "foie", "fois", "trait", "très",
    "paix", "pet", "pain", "pin", "peint", "sang", "sans", "cent", "seau",
    "sceau", "saut", "cher", "chair", "chère", "thé", "taie", "tes", "pansé",
    "pensée", "pense", "brise", "désert", "vocal", "poisson", "coupe-vent",
    "fumet", "presse", "récite", "contenir", "coiffe", "chérot",
}

# Exact homophones can look attractive but be useless if the second word is
# obscure, technical, rare, or an unnatural inflection.  These are not banned from
# the language; they are banned as pre-judge pivots unless later added back with
# evidence/frequency.
_BAD_LOW_VALUE_PUN_PIVOTS = {
    "axion", "axions", "saque", "sacque", "soufrai", "soufrait", "ceignît", "expiât", "frusques",
    "coursez", "coursé", "recourriez", "convulsionnons", "prenons", "nominalisation",
    "algol", "fusillons", "ramai", "nouant", "omette", "fugué", "fab",
    "totalisé", "localisé", "commenté", "au vrai", "en titre",
}

# Very common function/support forms often create perfect phonetic matches that
# are not usable joke pivots by themselves. They can still appear in semantic
# scaffolds, but should not dominate the generator-facing bridge list.
_LOW_PIVOT_FUNCTION_SURFACES = {
    "très", "être", "avoir", "auprès", "au vrai", "en titre", "voire", "entre",
    "avec", "sans", "pour", "dans", "donc", "mais", "ainsi",
}

_HIGH_PIVOT_NOUNISH_SUFFIXES = (
    "tion", "sion", "té", "eur", "euse", "age", "isme", "oir", "oire", "ure", "ise", "ance", "ence",
)


def surface_recognizability_prior(surface: Any) -> float:
    s = norm_text(surface)
    plain = strip_accents(s)
    if not s or s in _BAD_LOW_VALUE_PUN_PIVOTS or plain in {strip_accents(x) for x in _BAD_LOW_VALUE_PUN_PIVOTS}:
        return 0.0
    if s in _COMMON_FRENCH_PUN_PIVOTS or plain in {strip_accents(x) for x in _COMMON_FRENCH_PUN_PIVOTS}:
        return 0.42
    # Ordinary short alphabetic words are often usable; long/technical-looking
    # forms need corpus evidence before reaching the LLM judge.
    if re.fullmatch(r"[a-zàâçéèêëîïôûùüÿñæœ'-]{3,8}", s, flags=re.I):
        if not re.search(r"(tion|isation|isme|ions|iez|ions|aient|erent|assions)$", plain):
            return 0.16
    return 0.0








def expression_quality(item: dict[str, Any]) -> float:
    """Cheap naturalness prior for French pun pivots.

    This is deliberately conservative: one-word candidates should not look
    naturally usable unless they are attested by a source/frequency signal or
    get rescued by very strong phonetics.  This prevents rare conjugation junk
    from being promoted before the LLM judge.
    """
    source = norm_text(item.get("source", item.get("candidate_source", "")))
    surface = norm_text(item.get("surface", item.get("text", item.get("candidate", item.get("word", "")))))
    word_count = len(surface.split())

    if not surface or lexically_bad_candidate_surface(surface):
        return 0.0
    if surface in _BAD_LOW_VALUE_PUN_PIVOTS or strip_accents(surface) in {strip_accents(x) for x in _BAD_LOW_VALUE_PUN_PIVOTS}:
        return 0.0

    score = max(0.08, surface_recognizability_prior(surface))
    if "wiktionary" in source:
        score += 0.28
    if "parseme" in source:
        score += 0.24
    if "opensubtitles" in source:
        score += 0.16
    if "collocation" in source:
        score += 0.12
    if 2 <= word_count <= 5:
        score += 0.18
    elif word_count == 1:
        score += 0.02
    if clean(item.get("frequency")):
        score += 0.10
    if clean(item.get("pmi")):
        score += 0.04
    return float(min(score, 1.0))


def clamp01(x: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(x))))
    except Exception:
        return 0.0


def affordance_stage_rank(stage: Any, bridge_type: Any = "") -> int:
    """Lower is better. Used before scalar scores so strict matches beat broad coincidences."""
    st = clean(stage)
    bt = clean(bridge_type)
    if "direct" in st or bt in {"different_surface_homophone_bridge", "identity_polysemy_bridge"}:
        return 0
    if "level0" in st or "strict" in st:
        return 1
    if "level1" in st or "broad" in st:
        return 2
    if "fallback" in st:
        return 4
    return 3


def phonetic_relation_label(phon: float, same_ipa: bool = False) -> str:
    if same_ipa or phon >= 0.96:
        return "exact_or_near_homophone"
    if phon >= 0.82:
        return "strong_phonetic"
    if phon >= 0.68:
        return "near_phonetic"
    return "echo"


def humor_surprise_score(phon: float, source_sem: float, opposite_sem: float, same_root_flag: bool = False) -> float:
    """Approximate incongruity/recoverability for pre-judge ranking.

    Reward phonetic convergence, but avoid treating semantic similarity as the
    main virtue.  Moderate semantic anchorability is enough; excessive same-root
    or same-sense closeness should not dominate.
    """
    phon = clamp01(phon)
    source_sem = clamp01(source_sem)
    opposite_sem = clamp01(opposite_sem)
    anchorability = min(1.0, max(source_sem, opposite_sem) / 0.55)
    semantic_similarity_proxy = max(source_sem, opposite_sem)
    divergence = max(0.0, 1.0 - semantic_similarity_proxy)
    score = 0.55 * phon + 0.22 * divergence + 0.18 * anchorability
    if same_root_flag:
        score -= 0.25
    return clamp01(score)


def llm_priority_score_for_bridge(b: dict[str, Any]) -> float:
    phon = clamp01(b.get("phonetic_score", 0.0))
    source_sem = clamp01(b.get("source_semantic_score", b.get("semantic_A_score", 0.0)))
    opposite_sem = clamp01(b.get("opposite_semantic_score", b.get("semantic_B_score", 0.0)))
    left_tmp, right_tmp = bridge_surface_pair(b)
    naturalness = max(
        clamp01(b.get("naturalness_score", b.get("quality_score", 0.0))),
        surface_recognizability_prior(left_tmp),
        surface_recognizability_prior(right_tmp),
    )
    bridge_type = clean(b.get("bridge_type", ""))
    stage = clean(b.get("affordance_stage", b.get("stage", "")))
    same_root_flag = bool(b.get("same_root_penalty_applied", False))
    surprise = clamp01(b.get("surprise_score", humor_surprise_score(phon, source_sem, opposite_sem, same_root_flag)))
    pivotability = max(clamp01(b.get("pivotability_score", 0.0)), bridge_pivotability_score(b))

    type_bonus = 0.0
    if "homophone" in bridge_type:
        type_bonus += 0.10
    elif "strong_phonetic" in bridge_type:
        type_bonus += 0.06
    elif "near_phonetic" in bridge_type:
        type_bonus += 0.02
    if bool(b.get("semantic_verified", False)):
        type_bonus += 0.05
    if affordance_stage_rank(stage, bridge_type) >= 3:
        type_bonus -= 0.08
    if same_root_flag:
        type_bonus -= 0.18

    return clamp01(0.34 * phon + 0.20 * naturalness + 0.20 * surprise + 0.18 * pivotability + 0.08 * max(source_sem, opposite_sem) + type_bonus)



def bridge_type_for_pair(a: dict[str, Any], b: dict[str, Any], phonetic_score: float) -> str:
    """Classify why an A/B pair is useful for pun generation."""
    same_surface = norm_text(a.get("surface", "")) == norm_text(b.get("surface", ""))
    same_ipa = clean(a.get("ipa", "")) == clean(b.get("ipa", ""))

    if same_surface and same_ipa:
        return "identity_polysemy_bridge"
    if trivial_inflection_related(a.get("surface", ""), b.get("surface", "")):
        return "trivial_inflection_bridge"
    if same_ipa:
        return "different_surface_homophone_bridge"
    if phonetic_score >= 0.82:
        return "different_surface_strong_phonetic_bridge"
    if phonetic_score >= 0.68:
        return "different_surface_near_phonetic_bridge"
    return "near_rhyme_echo_bridge"


def bridge_type_bonus(bridge_type: str, semantic_A_score: float, semantic_B_score: float) -> float:
    """Prefer novel sound/meaning collisions, but keep strong true polysemy."""
    cross_semantic = min(semantic_A_score, semantic_B_score)

    if bridge_type == "different_surface_homophone_bridge":
        return 0.13
    if bridge_type == "different_surface_strong_phonetic_bridge":
        return 0.10
    if bridge_type == "different_surface_near_phonetic_bridge":
        return 0.07
    if bridge_type == "near_rhyme_echo_bridge":
        return 0.03
    if bridge_type == "identity_polysemy_bridge":
        # Good if genuinely shared by both meaning fields; bad if just a repeated seed.
        if cross_semantic >= 0.78:
            return 0.08
        if cross_semantic >= 0.62:
            return 0.02
        return -0.06
    if bridge_type == "trivial_inflection_bridge":
        return -0.10
    return 0.0


def select_diverse_bridges(bridges: list[dict[str, Any]], max_bridges: int = MAX_BRIDGES) -> list[dict[str, Any]]:
    """Keep top bridges while preventing identity/probe echoes from dominating."""
    if not bridges:
        return []

    bridges = sorted(bridges, key=lambda x: (affordance_stage_rank(x.get("affordance_stage", ""), x.get("bridge_type", "")), -float(x.get("llm_priority_score", x.get("bridge_score", 0.0)))), reverse=False)
    max_identity = max(1, int(math.ceil(max_bridges * MAX_IDENTITY_BRIDGE_FRACTION)))
    identity_count = 0
    surface_counts: dict[str, int] = {}
    seen_pairs: set[tuple[str, str, str, str]] = set()
    selected: list[dict[str, Any]] = []

    # First pass: enforce caps.
    for b in bridges:
        btype = clean(b.get("bridge_type", b.get("relation", "")))
        is_identity = "identity" in btype or "trivial_inflection" in btype
        if is_identity and identity_count >= max_identity:
            continue

        left_surface, right_surface = bridge_surface_pair(b)
        left = norm_text(left_surface)
        right = norm_text(right_surface)
        pair_key = (left, right, clean(b.get("left_ipa", b.get("sound_source_ipa", ""))), clean(b.get("right_ipa", b.get("candidate_ipa", ""))))
        if pair_key in seen_pairs:
            continue

        # Avoid many variants around the same surface consuming the whole pack.
        left_key = f"L:{left}"
        right_key = f"R:{right}"
        left_root = f"LR:{rough_lemma_key(left)}"
        right_root = f"RR:{rough_lemma_key(right)}"
        if surface_counts.get(left_key, 0) >= MAX_BRIDGES_PER_SURFACE:
            continue
        if surface_counts.get(right_key, 0) >= MAX_BRIDGES_PER_SURFACE:
            continue
        if surface_counts.get(left_root, 0) >= MAX_BRIDGES_PER_ROOT:
            continue
        if surface_counts.get(right_root, 0) >= MAX_BRIDGES_PER_ROOT:
            continue

        selected.append(b)
        seen_pairs.add(pair_key)
        surface_counts[left_key] = surface_counts.get(left_key, 0) + 1
        surface_counts[right_key] = surface_counts.get(right_key, 0) + 1
        surface_counts[left_root] = surface_counts.get(left_root, 0) + 1
        surface_counts[right_root] = surface_counts.get(right_root, 0) + 1
        if is_identity:
            identity_count += 1
        if len(selected) >= max_bridges:
            return selected

    # Second pass: fill remaining slots, but never reintroduce trivial
    # same-root bridges just to hit a count target. Quality beats quantity.
    if len(selected) < max_bridges:
        selected_ids = {id(x) for x in selected}
        for b in bridges:
            if id(b) in selected_ids:
                continue
            left_surface, right_surface = bridge_surface_pair(b)
            if trivial_inflection_related(left_surface, right_surface):
                continue
            selected.append(b)
            if len(selected) >= max_bridges:
                break

    return selected[:max_bridges]


def bridge_diagnostics(bridges: list[dict[str, Any]]) -> dict[str, Any]:
    types: dict[str, int] = {}
    for b in bridges:
        t = clean(b.get("bridge_type", b.get("relation", "unknown"))) or "unknown"
        types[t] = types.get(t, 0) + 1
    strong = [b for b in bridges if bool(b.get("semantic_verified", False)) and float(b.get("bridge_score", 0.0)) >= STRONG_BRIDGE_THRESHOLD]
    return {
        "bridge_count": len(bridges),
        "strong_bridge_count": len(strong),
        "identity_bridge_count": sum(v for k, v in types.items() if "identity" in k),
        "trivial_inflection_bridge_count": sum(v for k, v in types.items() if "trivial_inflection" in k),
        "different_surface_bridge_count": len(bridges) - sum(v for k, v in types.items() if "identity" in k or "trivial_inflection" in k),
        "best_bridge_score": float(bridges[0].get("bridge_score", 0.0)) if bridges else 0.0,
        "bridge_type_counts": types,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Expression retrieval: dense semantic + lexical TF-IDF
# ─────────────────────────────────────────────────────────────────────────────


class ExpressionRetriever:
    def __init__(self, semantic_k: int = SEMANTIC_K, lexical_k: int = LEXICAL_K) -> None:
        self.semantic_k = semantic_k
        self.lexical_k = lexical_k

        if not EXPR_BANK_PATH.exists():
            raise FileNotFoundError(EXPR_BANK_PATH)
        if not EXPR_INDEX_PATH.exists():
            raise FileNotFoundError(EXPR_INDEX_PATH)

        log("Loading expression bank:", EXPR_BANK_PATH)
        self.bank = pd.read_parquet(EXPR_BANK_PATH).reset_index(drop=True)

        log("Loading expression FAISS index:", EXPR_INDEX_PATH)
        self.index = faiss.read_index(str(EXPR_INDEX_PATH))

        log("Loading expression embedding model:", MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)

        self.bank["surface"] = self.bank["surface"].fillna("").astype(str)
        if "content" not in self.bank.columns:
            self.bank["content"] = ""
        if "source" not in self.bank.columns:
            self.bank["source"] = "unknown"

        lexical_docs = (
            self.bank["surface"].fillna("").astype(str)
            + " "
            + self.bank["content"].fillna("").astype(str).str[:1000]
        ).tolist()

        # This is fast enough locally, but can be precomputed later.
        log("Building lexical TF-IDF matrix")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=500_000,
        )
        self.lexical_matrix = self.vectorizer.fit_transform(lexical_docs)

    def item(self, idx: int, score: float, channel: str) -> dict[str, Any]:
        row = self.bank.iloc[int(idx)]
        out = {
            "surface": clean(row.get("surface", "")),
            "source": clean(row.get("source", "unknown")),
            "score": float(score),
            "channel": channel,
        }
        if "content" in row:
            out["content"] = short(row.get("content", ""), 500)
        if "frequency" in row:
            out["frequency"] = clean(row.get("frequency", ""))
        if "pmi" in row:
            out["pmi"] = clean(row.get("pmi", ""))
        if "score" in row:
            out["source_score"] = clean(row.get("score", ""))
        return out

    def _ensure_runtime_caches(self) -> None:
        # Existing pipeline instances survive hot reload, so new attrs must be added lazily.
        if not hasattr(self, "_semantic_search_cache"):
            self._semantic_search_cache: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        if not hasattr(self, "_lexical_search_cache"):
            self._lexical_search_cache: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        if not hasattr(self, "_embedding_cache"):
            self._embedding_cache: dict[str, np.ndarray] = {}
        if not hasattr(self, "_semantic_score_cache"):
            self._semantic_score_cache: dict[tuple[str, tuple[str, ...]], list[float]] = {}

    def _encode_texts_cached(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        self._ensure_runtime_caches()
        cleaned = [clean(t) for t in texts]
        missing = [t for t in dict.fromkeys(cleaned) if t and t not in self._embedding_cache]
        if missing:
            embs = self.model.encode(
                missing,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            for t, e in zip(missing, embs):
                self._embedding_cache[t] = e
        if not cleaned:
            return np.zeros((0, 1), dtype="float32")
        dim = next(iter(self._embedding_cache.values())).shape[0] if self._embedding_cache else 1
        return np.vstack([self._embedding_cache.get(t, np.zeros(dim, dtype="float32")) for t in cleaned]).astype("float32")

    def semantic_search(self, query: str, top_k: int | None = None, channel: str = "semantic") -> list[dict[str, Any]]:
        self._ensure_runtime_caches()
        query = clean(query)
        if not query:
            return []
        k = top_k or self.semantic_k
        key = (query, k, channel)
        if key in self._semantic_search_cache:
            return [dict(x) for x in self._semantic_search_cache[key]]
        q = self._encode_texts_cached([query])
        scores, indices = self.index.search(q, k)
        out = [self.item(idx, score, channel) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        self._semantic_search_cache[key] = [dict(x) for x in out]
        return out

    def semantic_search_many(self, requests: list[tuple[str, int, str]]) -> dict[str, list[dict[str, Any]]]:
        """Batch semantic FAISS searches for row-level queries.

        This preserves retrieval quality while avoiding three separate BGE encode
        calls per row for blended/A/B semantic searches. Cached queries are
        reused; uncached queries are encoded in one batch.
        """
        self._ensure_runtime_caches()
        out: dict[str, list[dict[str, Any]]] = {}
        missing: list[tuple[str, int, str]] = []
        for query, top_k, channel in requests:
            q = clean(query)
            if not q:
                out[channel] = []
                continue
            k = top_k or self.semantic_k
            key = (q, k, channel)
            cached = self._semantic_search_cache.get(key)
            if cached is not None:
                out[channel] = [dict(x) for x in cached]
            else:
                missing.append((q, k, channel))

        if missing:
            q_emb = self._encode_texts_cached([m[0] for m in missing])
            max_k = max(m[1] for m in missing)
            scores, indices = self.index.search(q_emb, max_k)
            for row_i, (q, k, channel) in enumerate(missing):
                items = [self.item(idx, score, channel) for idx, score in zip(indices[row_i][:k], scores[row_i][:k]) if idx >= 0]
                self._semantic_search_cache[(q, k, channel)] = [dict(x) for x in items]
                out[channel] = [dict(x) for x in items]
        return out

    def lexical_search(self, query: str, top_k: int | None = None, channel: str = "lexical") -> list[dict[str, Any]]:
        self._ensure_runtime_caches()
        query = clean(query)
        if not query:
            return []
        k = top_k or self.lexical_k
        key = (query, k, channel)
        if key in self._lexical_search_cache:
            return [dict(x) for x in self._lexical_search_cache[key]]
        q = self.vectorizer.transform([query])
        scores = linear_kernel(q, self.lexical_matrix).ravel()
        top = np.argsort(scores)[::-1][:k]
        out = [self.item(idx, scores[idx], channel) for idx in top if scores[idx] > 0]
        self._lexical_search_cache[key] = [dict(x) for x in out]
        return out

    def semantic_scores(self, query: str, texts: list[str], batch_size: int = 64) -> list[float]:
        """Score arbitrary French candidate texts against a semantic-field query.

        Cache is per (query, candidate), not per whole batch, so changing the
        candidate list/order still reuses prior work across rows.
        """
        self._ensure_runtime_caches()
        query = clean(query)
        clean_texts = [clean(t) for t in texts]
        if not query or not clean_texts:
            return [0.0 for _ in clean_texts]

        # Backward-compatible lazy cache: older live pipeline objects may have
        # _semantic_score_cache keyed by (query, tuple(texts)); add the per-item
        # cache without requiring server restart.
        if not hasattr(self, "_semantic_score_item_cache"):
            self._semantic_score_item_cache: dict[tuple[str, str], float] = {}

        out: list[float | None] = []
        missing: list[str] = []
        missing_seen: set[str] = set()
        for t in clean_texts:
            k = (query, t)
            if k in self._semantic_score_item_cache:
                out.append(float(self._semantic_score_item_cache[k]))
            else:
                out.append(None)
                if t and t not in missing_seen:
                    missing_seen.add(t)
                    missing.append(t)

        if missing:
            q = self._encode_texts_cached([query], batch_size=batch_size)
            e = self._encode_texts_cached(missing, batch_size=batch_size)
            vals = np.matmul(e, q[0]).astype(float).tolist()
            for t, val in zip(missing, vals):
                self._semantic_score_item_cache[(query, t)] = float(val)

        final: list[float] = []
        for current, t in zip(out, clean_texts):
            if current is None:
                final.append(float(self._semantic_score_item_cache.get((query, t), 0.0)))
            else:
                final.append(float(current))
        return final



# ─────────────────────────────────────────────────────────────────────────────
# Optional FastText word-level semantic expansion
# ─────────────────────────────────────────────────────────────────────────────


class FastTextExpansionBackend:
    """Bounded word-level semantic stepping for Low-style expansion.

    Best-practice constraints:
    - Use FastText as a controlled lexical broadener, not a recursive graph walk.
    - Expand only selected seed terms once.
    - Cache every expansion.
    - Keep per-row candidate counts bounded and visible in diagnostics.
    """

    def __init__(self) -> None:
        self.enabled = bool(USE_FASTTEXT)
        self.model = None
        self.cache: dict[str, list[tuple[str, float]]] = {}
        self.last_stats: dict[str, Any] = {
            "fasttext_enabled": False,
            "fasttext_seed_count": 0,
            "fasttext_expansion_count": 0,
            "fasttext_budget_filled": False,
        }
        if not self.enabled:
            log("FastText expansion disabled")
            return

        path = Path(FASTTEXT_MODEL_PATH)
        if not path.exists():
            log(f"WARNING: FastText model not found at {path}; expansion disabled")
            self.enabled = False
            return

        log("Loading FastText expansion model:", path)
        try:
            from gensim.models.fasttext import load_facebook_vectors
            self.model = load_facebook_vectors(str(path))
            self.enabled = True
            self.last_stats["fasttext_enabled"] = True
            log("FastText expansion model loaded")
        except Exception as e:
            global _FASTTEXT_WARNED
            if not _FASTTEXT_WARNED:
                log(
                    f"WARNING: could not load FastText model ({e}). "
                    "Install with: python -m pip install gensim scipy"
                )
                _FASTTEXT_WARNED = True
            self.enabled = False
            self.model = None

    def _tokens_for_seed(self, text: str) -> list[str]:
        text = norm_text(text)
        if not text:
            return []
        parts = [
            p for p in re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ'\-]+", text, flags=re.I)
            if len(p) >= 2
        ]
        # FastText is word-level. For multiword expressions, use only a few
        # salient content tokens to avoid exploding the search.
        if " " not in text and len(text) >= 2:
            parts = [text] + parts
        # Prefer longer/salient tokens, but preserve deterministic order.
        parts = unique_keep_order(parts, limit=8)
        parts = sorted(parts, key=lambda x: (-len(x), x))
        return unique_keep_order(parts, limit=FASTTEXT_MAX_TOKENS_PER_SEED)

    def _neighbors_for_token(self, token: str) -> list[tuple[str, float]]:
        if not self.enabled or self.model is None or not token:
            return []
        key = norm_text(token)
        if key in self.cache:
            return self.cache[key]
        try:
            raw = self.model.most_similar(token, topn=FASTTEXT_K)
        except Exception:
            self.cache[key] = []
            return []
        out: list[tuple[str, float]] = []
        for word, score in raw:
            word = clean(word).replace("_", " ")
            score = float(score)
            if not word or len(word) < 2 or score < FASTTEXT_MIN_SIM:
                continue
            # Keep one-token lexical expansions. Multiword FastText artifacts are
            # rarely useful for this backend and inflate downstream phonetic search.
            if len(word.split()) > 1:
                continue
            out.append((word, score))
        self.cache[key] = out
        return out

    def expand(
        self,
        seeds: list[str],
        side: str,
        level: int = 1,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Expand a bounded seed beam exactly once.

        Hitting the limit is not a fallback or failure; it means the planned
        FastText beam is full. This keeps runtime predictable while preserving
        the useful lexical-neighborhood signal.
        """
        if not self.enabled:
            self.last_stats = {
                "fasttext_enabled": False,
                "fasttext_seed_count": 0,
                "fasttext_expansion_count": 0,
                "fasttext_budget_filled": False,
            }
            return []

        limit = limit or FASTTEXT_MAX_CANDIDATES_PER_SIDE
        level_penalty = FASTTEXT_LEVEL1_PENALTY

        selected_seeds = unique_keep_order(seeds, limit=FASTTEXT_SEED_LIMIT)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        for seed in selected_seeds:
            for token in self._tokens_for_seed(seed):
                for word, sim in self._neighbors_for_token(token):
                    key = norm_text(word)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    # Same-root variants are not semantic drift; phonetic search
                    # already captures them and the bridge reranker penalizes them.
                    if same_root(seed, word):
                        continue
                    candidates.append({
                        "text": word,
                        "surface": word,
                        "side": side,
                        "level": level,
                        "source": "fasttext_cc_fr",
                        "semantic_score": float(sim) * level_penalty,
                        "fasttext_score": float(sim),
                        "channel": f"fasttext_{side}_L{level}",
                        "content": f"FastText neighbor of {seed}",
                        "parent": seed,
                    })
                    if len(candidates) >= limit:
                        self.last_stats = {
                            "fasttext_enabled": True,
                            "fasttext_seed_count": len(selected_seeds),
                            "fasttext_expansion_count": len(candidates),
                            "fasttext_budget_filled": True,
                        }
                        return candidates

        self.last_stats = {
            "fasttext_enabled": True,
            "fasttext_seed_count": len(selected_seeds),
            "fasttext_expansion_count": len(candidates),
            "fasttext_budget_filled": len(candidates) >= limit,
        }
        return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Phonetic retrieval: trained IPA encoder + corpus IPA lookup
# ─────────────────────────────────────────────────────────────────────────────


class PhoneticRetriever:
    def __init__(self, top_k: int = PHONETIC_K) -> None:
        self.top_k = top_k

        log("Loading phonetic items...")
        self.items = load(phonetic_items_path).reset_index(drop=True)
        self.items["word"] = self.items["word"].fillna("").astype(str)
        self.items["ipa"] = self.items["ipa"].fillna("").astype(str)
        self.items["_norm_word"] = self.items["word"].map(norm_text)

        log("Building exact surface→IPA lookup")
        self.surface_to_indices: dict[str, list[int]] = {}
        for i, key in enumerate(self.items["_norm_word"].tolist()):
            if key:
                self.surface_to_indices.setdefault(key, []).append(i)

        log("Building exact IPA→index lookup")
        self.ipa_to_indices: dict[str, list[int]] = {}
        for i, ipa_key in enumerate(self.items["ipa"].map(clean).tolist()):
            if ipa_key:
                self.ipa_to_indices.setdefault(ipa_key, []).append(i)

        log("Loading phonetic model...")
        self.model = SentenceTransformer(phonetic_model_path, local_files_only=True)

        log("Loading phonetic FAISS index...")
        self.index = faiss.read_index(phonetic_index_path)

        self.embedding_matrix = None
        emb_path = phonetic_embeddings_path
        try:
            if emb_path and Path(str(emb_path)).exists():
                log("Loading phonetic embeddings mmap:", emb_path)
                arr = np.load(str(emb_path), mmap_mode="r")
                if int(arr.shape[0]) == int(self.index.ntotal):
                    self.embedding_matrix = arr
                else:
                    log("Skipping phonetic embeddings mmap: row count mismatch", arr.shape, self.index.ntotal)
        except Exception as e:
            log("Skipping phonetic embeddings mmap:", e)

    def _ensure_runtime_caches(self) -> None:
        # Existing pipeline instances survive hot reload, so new attrs must be added lazily.
        if not hasattr(self, "_lookup_cache"):
            self._lookup_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        if not hasattr(self, "_ipa_embedding_cache"):
            self._ipa_embedding_cache: dict[str, np.ndarray] = {}
        if not hasattr(self, "_search_cache"):
            self._search_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        if not hasattr(self, "_search_from_text_cache"):
            self._search_from_text_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def lookup_records(self, text: str, limit: int = 5) -> list[dict[str, Any]]:
        """Exact corpus lookup for French surface → known IPA. No guessing/G2P."""
        self._ensure_runtime_caches()
        key = norm_text(text)
        ckey = (key, limit)
        if ckey in self._lookup_cache:
            return [dict(x) for x in self._lookup_cache[ckey]]
        idxs = self.surface_to_indices.get(key, [])[:limit]
        out: list[dict[str, Any]] = []
        for idx in idxs:
            row = self.items.iloc[int(idx)]
            out.append({
                "word": clean(row.get("word", "")),
                "ipa": clean(row.get("ipa", "")),
                "rhyme": clean(row.get("rhyme", "")),
                "suffix2": clean(row.get("suffix2", "")),
                "suffix3": clean(row.get("suffix3", "")),
                "consonant_skeleton": clean(row.get("consonant_skeleton", "")),
                "vowel_skeleton": clean(row.get("vowel_skeleton", "")),
                "split": clean(row.get("split", "")),
            })
        self._lookup_cache[ckey] = [dict(x) for x in out]
        return out

    def _vector_from_index_for_ipa(self, ipa: str) -> np.ndarray | None:
        """Return the precomputed FAISS/index vector for an IPA already in the corpus.

        This is the critical fast path: bridge mining queries are almost always
        IPA strings from phonetic_items.tsv. Re-encoding those strings with the
        SentenceTransformer inside every row is the source of the multi-minute
        slowdown. For known IPA, reconstruct the already-indexed vector instead.
        """
        self._ensure_runtime_caches()
        ipa = clean(ipa)
        if not ipa:
            return None
        if not hasattr(self, "ipa_to_indices"):
            self.ipa_to_indices = {}
            for i, ipa_key in enumerate(self.items["ipa"].map(clean).tolist()):
                if ipa_key:
                    self.ipa_to_indices.setdefault(ipa_key, []).append(i)
        idxs = self.ipa_to_indices.get(ipa, [])
        if not idxs:
            return None
        idx = int(idxs[0])
        try:
            emb = getattr(self, "embedding_matrix", None)
            if emb is not None:
                v = np.asarray(emb[idx], dtype="float32")
            else:
                v = self.index.reconstruct(idx).astype("float32")
            # The index was built over normalized embeddings; normalize defensively
            # in case future indexes differ.
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            return v.astype("float32")
        except Exception:
            return None

    def encode_ipa(self, ipa_strings: list[str]) -> np.ndarray:
        self._ensure_runtime_caches()
        if not ipa_strings:
            return np.zeros((0, 1), dtype="float32")
        cleaned = [clean(x) for x in ipa_strings]
        unique_missing_for_model: list[str] = []

        for ipa in dict.fromkeys(cleaned):
            if not ipa or ipa in self._ipa_embedding_cache:
                continue
            v = self._vector_from_index_for_ipa(ipa)
            if v is not None:
                self._ipa_embedding_cache[ipa] = v
            else:
                unique_missing_for_model.append(ipa)

        if unique_missing_for_model:
            embs = self.model.encode(
                unique_missing_for_model,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            for x, e in zip(unique_missing_for_model, embs):
                self._ipa_embedding_cache[x] = e

        dim = next(iter(self._ipa_embedding_cache.values())).shape[0] if self._ipa_embedding_cache else 1
        return np.vstack([self._ipa_embedding_cache.get(x, np.zeros(dim, dtype="float32")) for x in cleaned]).astype("float32")

    def rerank_neighbors(self, query_ipa: str, candidates: pd.DataFrame) -> pd.DataFrame:
        """Cheap vectorized reranker for a small FAISS oversample."""
        if candidates.empty:
            candidates["final_score"] = []
            return candidates
        q_suffix3 = query_ipa[-3:]
        q_suffix2 = query_ipa[-2:]
        ipa = candidates.get("ipa", pd.Series([""] * len(candidates))).fillna("").astype(str).map(clean)
        base = candidates.get("phonetic_score", pd.Series([0.0] * len(candidates))).astype(float).to_numpy()
        lens = ipa.str.len().to_numpy()
        qlen = max(len(query_ipa), 1)
        denom = np.maximum(np.maximum(lens, qlen), 1)
        edit = 1.0 - (np.abs(lens - qlen) / denom)
        edit = np.where(ipa.to_numpy() == query_ipa, 1.0, edit)
        suffix3 = (candidates.get("suffix3", pd.Series([""] * len(candidates))).fillna("").astype(str).map(clean).to_numpy() == q_suffix3).astype(float)
        suffix2 = (candidates.get("suffix2", pd.Series([""] * len(candidates))).fillna("").astype(str).map(clean).to_numpy() == q_suffix2).astype(float)
        consonant = candidates.get("consonant_skeleton", pd.Series([""] * len(candidates))).fillna("").astype(str).map(clean).ne("").astype(float).to_numpy()
        vowel = candidates.get("vowel_skeleton", pd.Series([""] * len(candidates))).fillna("").astype(str).map(clean).ne("").astype(float).to_numpy()
        candidates = candidates.copy()
        candidates["final_score"] = 0.65 * base + 0.15 * edit + 0.10 * suffix3 + 0.05 * suffix2 + 0.03 * consonant + 0.02 * vowel
        return candidates.sort_values("final_score", ascending=False)

    def search(self, query_ipa: str, top_k: int | None = None) -> list[dict[str, Any]]:
        self._ensure_runtime_caches()
        top_k = top_k or self.top_k
        query_ipa = clean(query_ipa)
        if not query_ipa:
            return []
        ckey = (query_ipa, top_k)
        if ckey in self._search_cache:
            return [dict(x) for x in self._search_cache[ckey]]

        q = self.encode_ipa([query_ipa])
        # Ask FAISS for an oversample, because family/lexical cleaning can drop rows.
        # This preserves recall while still returning a small final beam.
        search_k = max(top_k, min(top_k * 8, 96))
        scores, idxs = self.index.search(q, search_k)
        out = self.items.iloc[idxs[0]].copy()
        out["phonetic_score"] = scores[0]
        out = self.rerank_neighbors(query_ipa, out)
        records = out.to_dict("records")
        records = [r for r in records if not lexically_bad_candidate_surface(r.get("word", ""))]
        records = collapse_phonetic_records_by_family(records, surface_key_name="word", limit=top_k)
        out = pd.DataFrame(records)

        keep_cols = [
            "word",
            "ipa",
            "phonetic_score",
            "final_score",
            "rhyme",
            "suffix2",
            "suffix3",
            "consonant_skeleton",
            "vowel_skeleton",
        ]
        if out.empty:
            result: list[dict[str, Any]] = []
        else:
            result = out[[c for c in keep_cols if c in out.columns]].head(top_k).to_dict("records")
        self._search_cache[ckey] = [dict(x) for x in result]
        return result

    def search_many(self, query_ipas: list[str], top_k: int | None = None) -> dict[str, list[dict[str, Any]]]:
        """Batch phonetic-neighbor search for many IPA probes.

        This keeps the server/model contract unchanged but avoids doing one FAISS
        call per semantic probe.  Cached probes are reused; uncached probes are
        encoded and searched as one matrix.
        """
        self._ensure_runtime_caches()
        top_k = top_k or self.top_k
        cleaned = [clean(x) for x in query_ipas if clean(x)]
        unique_ipas = list(dict.fromkeys(cleaned))
        if not unique_ipas:
            return {}

        result: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []
        for ipa in unique_ipas:
            ckey = (ipa, top_k)
            cached = self._search_cache.get(ckey)
            if cached is not None:
                result[ipa] = [dict(x) for x in cached]
            else:
                missing.append(ipa)

        if missing:
            q = self.encode_ipa(missing)
            search_k = max(top_k, min(top_k * 8, 96))
            scores, idxs = self.index.search(q, search_k)

            for row_i, query_ipa in enumerate(missing):
                out = self.items.iloc[idxs[row_i]].copy()
                out["phonetic_score"] = scores[row_i]
                out = self.rerank_neighbors(query_ipa, out)
                records = out.to_dict("records")
                records = [r for r in records if not lexically_bad_candidate_surface(r.get("word", ""))]
                records = collapse_phonetic_records_by_family(records, surface_key_name="word", limit=top_k)
                df = pd.DataFrame(records)

                keep_cols = [
                    "word",
                    "ipa",
                    "phonetic_score",
                    "final_score",
                    "rhyme",
                    "suffix2",
                    "suffix3",
                    "consonant_skeleton",
                    "vowel_skeleton",
                ]
                if df.empty:
                    out_records: list[dict[str, Any]] = []
                else:
                    out_records = df[[c for c in keep_cols if c in df.columns]].head(top_k).to_dict("records")
                self._search_cache[(query_ipa, top_k)] = [dict(x) for x in out_records]
                result[query_ipa] = [dict(x) for x in out_records]

        return result

    def search_from_text(self, text: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Search phonetic neighbors for a French surface only if the IPA is known."""
        self._ensure_runtime_caches()
        k = top_k or self.top_k
        ckey = (norm_text(text), k)
        if ckey in self._search_from_text_cache:
            return [dict(x) for x in self._search_from_text_cache[ckey]]
        records = self.lookup_records(text, limit=1)
        if not records:
            self._search_from_text_cache[ckey] = []
            return []
        query_ipa = records[0]["ipa"]
        results = self.search(query_ipa, top_k=k)
        out = []
        for r in results:
            nr = dict(r)
            nr["probe_text"] = clean(text)
            nr["probe_ipa"] = query_ipa
            out.append(nr)
        self._search_from_text_cache[ckey] = [dict(x) for x in out]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Low-style bridge mining
# ─────────────────────────────────────────────────────────────────────────────


class BridgeMiner:
    """
    Low-style search over semantic neighborhoods.

    We are not asking retrieval to write the joke. We ask it to discover candidate
    sound/meaning bridges:
      side A semantic neighborhood × side B semantic neighborhood → phonetic collisions.

    If no direct collision is found, we step semantically outward and still return
    fallback affordances for the generator.
    """

    def __init__(self, expression: ExpressionRetriever, phonetic: PhoneticRetriever, fasttext: FastTextExpansionBackend | None = None) -> None:
        self.expression = expression
        self.phonetic = phonetic
        self.fasttext = fasttext

    def _seed_candidates(self, terms: list[str], side: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for term in unique_keep_order(terms):
            out.append({
                "text": term,
                "surface": term,
                "side": side,
                "level": 0,
                "source": "meaning_seed",
                "semantic_score": 1.0,
                "channel": f"seed_{side}",
                "content": "",
            })
        return out

    def _semantic_candidates(self, query: str, side: str, level: int, top_k: int) -> list[dict[str, Any]]:
        results = self.expression.semantic_search(query, top_k=top_k, channel=f"semantic_{side}_L{level}")
        out: list[dict[str, Any]] = []
        level_penalty = 0.90 if level == 1 else 0.72
        for r in results:
            out.append({
                "text": r["surface"],
                "surface": r["surface"],
                "side": side,
                "level": level,
                "source": r.get("source", "unknown"),
                "semantic_score": float(r.get("score", 0.0)) * level_penalty,
                "channel": r.get("channel", f"semantic_{side}_L{level}"),
                "content": r.get("content", ""),
                "frequency": r.get("frequency", ""),
                "pmi": r.get("pmi", ""),
            })
        return out

    def _semantic_result_nodes(self, results: list[dict[str, Any]], side: str, level: int = 1, limit: int | None = None) -> list[dict[str, Any]]:
        """Convert already-computed row semantic results into bridge-mining nodes.

        This avoids doing the same BGE semantic search again inside bridge mining.
        """
        out: list[dict[str, Any]] = []
        level_penalty = 0.90 if level == 1 else 0.72
        for r in (results or [])[: (limit or SIDE_SEMANTIC_K)]:
            surface = clean(r.get("surface", r.get("text", "")))
            if not surface or lexically_bad_candidate_surface(surface):
                continue
            out.append({
                "text": surface,
                "surface": surface,
                "side": side,
                "level": level,
                "source": r.get("source", "precomputed_semantic"),
                "semantic_score": float(r.get("score", r.get("semantic_score", 0.0)) or 0.0) * level_penalty,
                "channel": r.get("channel", f"semantic_{side}_L{level}_reused"),
                "content": r.get("content", ""),
                "frequency": r.get("frequency", ""),
                "pmi": r.get("pmi", ""),
            })
        return out

    def _finalize_side_candidates(self, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Deduplicate/collapse candidates and attach IPA records."""
        by_key: dict[str, dict[str, Any]] = {}
        for cand in candidates:
            key = norm_text(cand.get("surface", cand.get("text", "")))
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = cand
            else:
                if (cand["semantic_score"], -cand["level"]) > (existing["semantic_score"], -existing["level"]):
                    by_key[key] = cand

        all_candidates = list(by_key.values())
        all_candidates.sort(key=lambda x: (x["level"], -float(x.get("semantic_score", 0.0))))
        all_candidates = collapse_dicts_by_root(
            all_candidates,
            surface_key_name="surface",
            score_keys=("semantic_score", "quality_score"),
        )

        with_ipa: list[dict[str, Any]] = []
        for cand in all_candidates:
            if lexically_bad_candidate_surface(cand.get("surface", "")):
                continue
            for rec in self.phonetic.lookup_records(cand["surface"], limit=1):
                enriched = dict(cand)
                enriched.update(rec)
                enriched["quality_score"] = expression_quality({
                    "surface": cand.get("surface", ""),
                    "source": cand.get("source", ""),
                    "frequency": cand.get("frequency", ""),
                })
                with_ipa.append(enriched)
                break

        with_ipa.sort(key=lambda x: (x["level"], -float(x.get("semantic_score", 0.0)), -float(x.get("quality_score", 0.0))))
        with_ipa = collapse_dicts_by_root(
            with_ipa,
            surface_key_name="surface",
            score_keys=("semantic_score", "quality_score"),
            limit=MAX_IPA_CANDIDATES_PER_SIDE,
        )
        return all_candidates, with_ipa

    def expand_side_from_precomputed(
        self,
        terms: list[str],
        side: str,
        semantic_results: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fast bridge-mining side expansion using row-level semantic results.

        The old path called BGE semantic_search again inside bridge_mining for A and B.
        This path reuses semantic_A/semantic_B already computed in retrieve_row, keeping
        accuracy/recall while eliminating duplicate search work.
        """
        seeds = unique_keep_order(terms, limit=20)
        candidates = self._seed_candidates(seeds, side)
        level1 = self._semantic_result_nodes(semantic_results or [], side=side, level=1, limit=SIDE_SEMANTIC_K)
        candidates.extend(level1)

        if self.fasttext is not None and self.fasttext.enabled:
            bge_seed_terms = unique_keep_order([x["surface"] for x in level1[:8]], limit=8)
            ft_seed_pool = unique_keep_order(seeds + bge_seed_terms, limit=FASTTEXT_SEED_LIMIT)
            candidates.extend(self.fasttext.expand(
                ft_seed_pool,
                side=side,
                level=1,
                limit=FASTTEXT_MAX_CANDIDATES_PER_SIDE,
            ))

        return self._finalize_side_candidates(candidates)

    def expand_bucketized_side_from_precomputed(
        self,
        terms: list[str],
        side: str,
        semantic_results: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Build explicit original and expanded pools for bucketized mining.

        A0/B0 are only original French list seeds with known IPA. A1/B1 are
        semantic-near candidates, including the bounded FastText beam, with known
        IPA. The pools are finalized independently so primary bucket regimes can
        be executed as separate vectorized pairwise searches.
        """
        seeds = unique_keep_order(terms, limit=20)
        seed_nodes = self._seed_candidates(seeds, side)

        if semantic_results is not None:
            level1 = self._semantic_result_nodes(semantic_results or [], side=side, level=1, limit=SIDE_SEMANTIC_K)
        else:
            q1 = " ".join(seeds)
            level1 = self._semantic_candidates(q1, side=side, level=1, top_k=SIDE_SEMANTIC_K) if q1 else []

        # Widen A1/B1 at the source. Row-level semantic results are often
        # dominated by the blended domain query, so each original seed also gets
        # its own bounded semantic query. This broadens buckets 2-4 and also
        # gives detached buckets 5-6 more anchors, without loosening final junk
        # filters.
        per_seed_level1: list[dict[str, Any]] = []
        if A1B1_PER_SEED_SEMANTIC_ENABLED:
            for seed in unique_keep_order(seeds, limit=A1B1_PER_SEED_LIMIT):
                per_seed_level1.extend(
                    self._semantic_candidates(
                        seed,
                        side=side,
                        level=1,
                        top_k=A1B1_PER_SEED_SEMANTIC_K,
                    )
                )

        level1 = level1 + per_seed_level1

        ft_level1: list[dict[str, Any]] = []
        if self.fasttext is not None and self.fasttext.enabled:
            bge_seed_terms = unique_keep_order([x["surface"] for x in level1[:16]], limit=16)
            ft_seed_pool = unique_keep_order(seeds + bge_seed_terms, limit=FASTTEXT_SEED_LIMIT)
            ft_level1 = self.fasttext.expand(
                ft_seed_pool,
                side=side,
                level=1,
                limit=FASTTEXT_MAX_CANDIDATES_PER_SIDE,
            )

        expanded_nodes = level1 + ft_level1
        all_nodes = seed_nodes + expanded_nodes

        semantic_all, ipa_all = self._finalize_side_candidates(all_nodes)
        _, ipa_0 = self._finalize_side_candidates(seed_nodes)
        _, ipa_1 = self._finalize_side_candidates(expanded_nodes)

        pool0_name = f"{side}0"
        pool1_name = f"{side}1"
        for item in ipa_0:
            item["retrieval_pool"] = pool0_name
        for item in ipa_1:
            item["retrieval_pool"] = pool1_name

        return {
            "semantic_candidates": semantic_all,
            "with_ipa": ipa_all,
            pool0_name: ipa_0,
            pool1_name: ipa_1,
        }

    def _build_bucket_bridge(
        self,
        bucket_id: str,
        source_pool: str,
        target_pool: str,
        source_item: dict[str, Any],
        target_item: dict[str, Any],
        phon: float,
    ) -> dict[str, Any]:
        source_side = "A" if source_pool.startswith("A") else "B"
        target_side = "A" if target_pool.startswith("A") else "B"
        bridge_type = bridge_type_for_pair(source_item, target_item, phon)

        source_semantic_score = float(source_item.get("semantic_score", 0.0) or 0.0)
        target_semantic_score = float(target_item.get("semantic_score", 0.0) or 0.0)
        semantic_A_score = source_semantic_score if source_side == "A" else target_semantic_score
        semantic_B_score = source_semantic_score if source_side == "B" else target_semantic_score
        quality = (float(source_item.get("quality_score", 0.0) or 0.0) + float(target_item.get("quality_score", 0.0) or 0.0)) / 2.0
        level = max(int(source_item.get("level", 9) or 9), int(target_item.get("level", 9) or 9))
        level_bonus = {0: 1.0, 1: 0.88, 2: 0.74}.get(level, 0.60)
        type_bonus = bridge_type_bonus(bridge_type, semantic_A_score, semantic_B_score)
        cross_semantic_score = min(semantic_A_score, semantic_B_score)
        novelty = lexical_novelty_bonus(source_item["surface"], target_item["surface"])
        common_bonus = (commonness_bonus_from_item(source_item) + commonness_bonus_from_item(target_item)) / 2.0
        cross_side_bonus = CROSS_SIDE_COLLISION_BONUS if (not same_root(source_item["surface"], target_item["surface"]) and phon >= 0.72) else 0.0
        naturalness = quality
        same_root_flag = (
            same_root(source_item["surface"], target_item["surface"])
            or structurally_trivial_variant(source_item["surface"], target_item["surface"])
            or boring_morphophonetic_echo(source_item["surface"], target_item["surface"])
        )
        surprise = humor_surprise_score(phon, semantic_A_score, semantic_B_score, same_root_flag)
        bridge_score = (
            0.40 * phon
            + 0.13 * naturalness
            + 0.13 * surprise
            + 0.10 * min(semantic_A_score, semantic_B_score)
            + 0.04 * level_bonus
            + type_bonus
            + novelty
            + common_bonus
            + cross_side_bonus
        )

        bdict = {
            "left_side": source_side,
            "right_side": target_side,
            "left_text": source_item["surface"],
            "right_text": target_item["surface"],
            "left_ipa": source_item["ipa"],
            "right_ipa": target_item["ipa"],
            "left_source": source_item.get("source", ""),
            "right_source": target_item.get("source", ""),
            "left_level": int(source_item.get("level", -1) or -1),
            "right_level": int(target_item.get("level", -1) or -1),
            "phonetic_score": float(phon),
            "semantic_A_score": float(semantic_A_score),
            "semantic_B_score": float(semantic_B_score),
            "cross_semantic_score": float(cross_semantic_score),
            "quality_score": float(quality),
            "type_bonus": float(type_bonus),
            "bridge_score": float(bridge_score),
            "lexical_novelty_bonus": float(novelty),
            "commonness_bonus": float(common_bonus),
            "cross_side_collision_bonus": float(cross_side_bonus),
            "semantic_verified": True,
            "semantic_relation": "direct_bucketized_cross_side_semantic",
            "affordance_stage": f"bucket_{bucket_id}_{source_pool}_{target_pool}",
            "phonetic_relation": phonetic_relation_label(float(phon), same_ipa=clean(source_item.get("ipa", "")) == clean(target_item.get("ipa", ""))),
            "naturalness_score": float(naturalness),
            "surprise_score": float(surprise),
            "same_root_penalty_applied": bool(same_root_flag),
            "bridge_type": bridge_type,
            "relation": bridge_type,
            "retrieval_bucket": bucket_id,
            "retrieval_bucket_label": RETRIEVAL_BUCKET_LABELS.get(bucket_id, bucket_id),
            "retrieval_source_pool": source_pool,
            "retrieval_target_pool": target_pool,
        }
        bdict["llm_priority_score"] = llm_priority_score_for_bridge(bdict)
        bdict["retrieval_pair_key"] = retrieval_pair_key_for_bridge(bdict)
        return bdict

    def _run_bucket_pairwise(
        self,
        bucket_id: str,
        source_pool: str,
        target_pool: str,
        source_items: list[dict[str, Any]],
        target_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run one explicit bucket regime using vectorized phonetic scoring."""
        if not source_items or not target_items:
            return []
        sim = self._pairwise_phonetic(source_items, target_items)
        if not sim.size:
            return []

        source_bad = np.array([lexically_bad_candidate_surface(x.get("surface", "")) for x in source_items], dtype=bool)
        target_bad = np.array([lexically_bad_candidate_surface(x.get("surface", "")) for x in target_items], dtype=bool)
        valid_mask = sim >= MIN_PAIR_PHONETIC
        if source_bad.any():
            valid_mask[source_bad, :] = False
        if target_bad.any():
            valid_mask[:, target_bad] = False

        coords = np.argwhere(valid_mask)
        if not coords.size:
            return []

        phon_vals = sim[coords[:, 0], coords[:, 1]].astype(float)
        source_sem_vals = np.array([float(source_items[int(i)].get("semantic_score", 0.0) or 0.0) for i in coords[:, 0]], dtype=float)
        target_sem_vals = np.array([float(target_items[int(j)].get("semantic_score", 0.0) or 0.0) for j in coords[:, 1]], dtype=float)
        quality_vals = np.array([
            (float(source_items[int(i)].get("quality_score", 0.0) or 0.0) + float(target_items[int(j)].get("quality_score", 0.0) or 0.0)) / 2.0
            for i, j in coords
        ], dtype=float)
        cheap_vals = 0.42 * phon_vals + 0.20 * source_sem_vals + 0.20 * target_sem_vals + 0.08 * quality_vals
        if len(cheap_vals) > MAX_DIRECT_PAIR_CANDIDATES * 4:
            keep_idx = np.argpartition(-cheap_vals, MAX_DIRECT_PAIR_CANDIDATES * 4 - 1)[:MAX_DIRECT_PAIR_CANDIDATES * 4]
            coords = coords[keep_idx]
            phon_vals = phon_vals[keep_idx]
            cheap_vals = cheap_vals[keep_idx]

        candidates: list[tuple[float, int, int, float]] = []
        for (i, j), phon, cheap_base in zip(coords, phon_vals, cheap_vals):
            i = int(i)
            j = int(j)
            source_item = source_items[i]
            target_item = target_items[j]
            if structurally_trivial_variant(source_item.get("surface", ""), target_item.get("surface", "")) or boring_morphophonetic_echo(source_item.get("surface", ""), target_item.get("surface", "")):
                continue
            novelty = lexical_novelty_bonus(source_item["surface"], target_item["surface"])
            candidates.append((float(cheap_base) + novelty, i, j, float(phon)))

        candidates.sort(reverse=True, key=lambda x: x[0])
        bridges = [
            self._build_bucket_bridge(bucket_id, source_pool, target_pool, source_items[i], target_items[j], phon)
            for _, i, j, phon in candidates[:MAX_DIRECT_PAIR_CANDIDATES]
        ]
        bridges.sort(key=bucket_rank_sort_key)
        for rank, bridge in enumerate(bridges, 1):
            bridge["retrieval_bucket_rank"] = rank
        return bridges


    def _dedupe_anchor_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep one anchor per surface+IPA while preserving pool provenance.

        Detached buckets combine original and expanded anchors (B0+B1 or A0+A1).
        Original anchors should be retained before expanded duplicates because they
        are more recoverable for generation.
        """
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in items or []:
            surface = clean(item.get("surface", item.get("text", item.get("word", ""))))
            ipa = clean(item.get("ipa", item.get("candidate_ipa", "")))
            if not surface or not ipa:
                continue
            key = retrieval_pool_member_key(surface, ipa)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _detached_score_for_bridge(self, b: dict[str, Any]) -> float:
        """Rank detached candidates without requiring missing-side semantics."""
        phon = clamp01(b.get("phonetic_score", 0.0))
        natural = clamp01(b.get("naturalness_score", b.get("quality_score", 0.0)))
        recognizability = max(
            surface_recognizability_prior(b.get("detached_free_surface", "")),
            surface_recognizability_prior(b.get("detached_anchor_surface", "")),
        )
        semantic_distance = clamp01(b.get("detached_semantic_distance_from_anchor", 0.0))
        anchor_quality = clamp01(b.get("detached_anchor_quality", 0.0))
        # Small bonus for ordinary different-root lexical collisions. This should
        # never rescue a same-family candidate because those are hard-rejected.
        novelty = max(0.0, float(b.get("lexical_novelty_bonus", 0.0) or 0.0))
        return clamp01(
            0.45 * phon
            + 0.20 * natural
            + 0.15 * recognizability
            + 0.10 * semantic_distance
            + 0.05 * anchor_quality
            + 0.05 * novelty
        )

    def _detached_sound_collision_bridges(
        self,
        anchor_items: list[dict[str, Any]],
        anchor_pool_label: str,
        free_pool_label: str,
        bucket_id: str,
        anchor_side: str,
        free_side: str,
    ) -> list[dict[str, Any]]:
        """Generate one-anchor detached sound-collision bridges.

        This is intentionally not a normal pairwise bucket.  The anchor side is
        semantically meaningful (A0+A1 or B0+B1).  The free side is a clean French
        phonetic neighbor of that anchor.  There is no semantic check to the
        missing original domain; the LLM judge must decide whether the free word
        can be made useful.
        """
        anchors = self._dedupe_anchor_items(anchor_items)
        stats: dict[str, Any] = {
            "bucket_id": bucket_id,
            "anchor_pool_label": anchor_pool_label,
            "free_pool_label": free_pool_label,
            "anchors_input": int(len(anchor_items or [])),
            "anchors_after_dedupe": int(len(anchors)),
            "anchor_ipas_queried": 0,
            "anchors_missing_surface_or_ipa": 0,
            "anchors_with_neighbors": 0,
            "neighbors_seen": 0,
            "candidates_before_semantic_distance": 0,
            "semantic_distance_scored": 0,
            "accepted_before_anchor_cap": 0,
            "accepted_after_anchor_cap": 0,
            "returned_after_bucket_cap": 0,
            "rejections": {},
        }

        def reject(reason: str, n: int = 1) -> None:
            rejections = stats.setdefault("rejections", {})
            rejections[reason] = int(rejections.get(reason, 0)) + int(n)

        if not anchors:
            stats["rejections"]["no_anchors"] = 1
            self._detached_diagnostics.append(stats)
            return []

        anchor_ipas = [clean(x.get("ipa", "")) for x in anchors if clean(x.get("ipa", ""))]
        stats["anchor_ipas_queried"] = int(len(anchor_ipas))
        neighbors_by_ipa = self.phonetic.search_many(anchor_ipas, top_k=DETACHED_NEIGHBORS_PER_ANCHOR)

        raw: list[dict[str, Any]] = []
        # Batch semantic-distance requests per anchor.  This checks only that the
        # free word is not merely the same meaning/family as the anchor; it is not
        # a semantic check to the opposite original domain.
        for anchor in anchors:
            anchor_surface = clean(anchor.get("surface", anchor.get("text", anchor.get("word", ""))))
            anchor_ipa = clean(anchor.get("ipa", ""))
            if not anchor_surface or not anchor_ipa:
                stats["anchors_missing_surface_or_ipa"] = int(stats.get("anchors_missing_surface_or_ipa", 0)) + 1
                continue
            anchor_pool_actual = clean(anchor.get("retrieval_pool", "")) or anchor_pool_label
            neighbors = neighbors_by_ipa.get(anchor_ipa, []) or []
            if neighbors:
                stats["anchors_with_neighbors"] = int(stats.get("anchors_with_neighbors", 0)) + 1
            stats["neighbors_seen"] = int(stats.get("neighbors_seen", 0)) + int(len(neighbors))
            candidates_for_anchor: list[dict[str, Any]] = []
            seen_free_families: set[tuple[str, str]] = set()

            for n in neighbors:
                free_surface = clean(n.get("word", n.get("surface", "")))
                free_ipa = clean(n.get("ipa", ""))
                phon = float(n.get("phonetic_score", n.get("final_score", 0.0)) or 0.0)
                if not free_surface or not free_ipa:
                    reject("missing_free_surface_or_ipa")
                    continue
                if phon < DETACHED_MIN_PHONETIC:
                    reject("phonetic_below_detached_min")
                    continue
                if lexically_bad_candidate_surface(free_surface):
                    reject("lexically_bad_free_surface")
                    continue
                verb_reason = detached_low_value_verb_form_reason(free_surface)
                if verb_reason:
                    reject(verb_reason)
                    continue
                commonness_reason = detached_low_commonness_reason(free_surface)
                if commonness_reason:
                    reject(commonness_reason)
                    continue
                shorthand_reason = detached_clipped_or_shorthand_reason(free_surface)
                if shorthand_reason:
                    reject(shorthand_reason)
                    continue
                if surface_key(free_surface) == surface_key(anchor_surface):
                    reject("same_surface_as_anchor")
                    continue
                if same_root(anchor_surface, free_surface):
                    reject("same_root_as_anchor")
                    continue
                if structurally_trivial_variant(anchor_surface, free_surface):
                    reject("structurally_trivial_variant")
                    continue
                if boring_morphophonetic_echo(anchor_surface, free_surface):
                    reject("boring_morphophonetic_echo")
                    continue
                if universal_trivial_bridge(anchor_surface, free_surface):
                    reject("universal_trivial_bridge")
                    continue
                echo_reason = detached_orthographic_echo_reason(anchor_surface, free_surface)
                if echo_reason:
                    reject(echo_reason)
                    continue

                family_key = phonetic_family_key(free_surface, free_ipa)
                if family_key in seen_free_families:
                    reject("duplicate_free_phonetic_family_for_anchor")
                    continue
                seen_free_families.add(family_key)

                naturalness = max(
                    expression_quality({"surface": free_surface, "source": n.get("source", "")}),
                    surface_recognizability_prior(free_surface),
                )
                anchor_quality = max(
                    float(anchor.get("quality_score", 0.0) or 0.0),
                    expression_quality({"surface": anchor_surface, "source": anchor.get("source", "")}),
                    surface_recognizability_prior(anchor_surface),
                )
                if naturalness < DETACHED_MIN_NATURALNESS:
                    reject("naturalness_below_detached_min")
                    continue

                stats["candidates_before_semantic_distance"] = int(stats.get("candidates_before_semantic_distance", 0)) + 1
                candidates_for_anchor.append({
                    "anchor": anchor,
                    "neighbor": n,
                    "anchor_surface": anchor_surface,
                    "anchor_ipa": anchor_ipa,
                    "anchor_pool_actual": anchor_pool_actual,
                    "free_surface": free_surface,
                    "free_ipa": free_ipa,
                    "phonetic_score": phon,
                    "phonetic_final_score": float(n.get("final_score", phon) or phon),
                    "naturalness_score": float(naturalness),
                    "anchor_quality": float(anchor_quality),
                })

            if not candidates_for_anchor:
                reject("no_candidates_survived_pre_semantic_for_anchor")
                continue

            semantic_scores = self.expression.semantic_scores(
                anchor_surface,
                [x["free_surface"] for x in candidates_for_anchor],
            )

            stats["semantic_distance_scored"] = int(stats.get("semantic_distance_scored", 0)) + int(len(candidates_for_anchor))
            accepted_for_anchor: list[dict[str, Any]] = []
            for item, sim in zip(candidates_for_anchor, semantic_scores):
                sim = float(sim)
                if sim >= DETACHED_MAX_ANCHOR_SEMANTIC_SIM:
                    reject("semantic_too_close_to_anchor")
                    continue

                free_surface = item["free_surface"]
                anchor_surface = item["anchor_surface"]
                phon = float(item["phonetic_score"])
                naturalness = float(item["naturalness_score"])
                anchor_quality = float(item["anchor_quality"])
                semantic_distance = clamp01(1.0 - sim)
                same_ipa = clean(item["free_ipa"]) == clean(item["anchor_ipa"])
                novelty = lexical_novelty_bonus(free_surface, anchor_surface)
                surprise = humor_surprise_score(phon, 0.0, 0.0, False)

                if free_side == "A":
                    left_side, right_side = "A", "B"
                    left_text, right_text = free_surface, anchor_surface
                    left_ipa, right_ipa = item["free_ipa"], item["anchor_ipa"]
                    semantic_A_score, semantic_B_score = 0.0, float(item["anchor"].get("semantic_score", 1.0) or 1.0)
                    retrieval_source_pool, retrieval_target_pool = free_pool_label, anchor_pool_label
                else:
                    left_side, right_side = "B", "A"
                    left_text, right_text = free_surface, anchor_surface
                    left_ipa, right_ipa = item["free_ipa"], item["anchor_ipa"]
                    semantic_A_score, semantic_B_score = float(item["anchor"].get("semantic_score", 1.0) or 1.0), 0.0
                    retrieval_source_pool, retrieval_target_pool = free_pool_label, anchor_pool_label

                bdict = {
                    "left_side": left_side,
                    "right_side": right_side,
                    "left_text": left_text,
                    "right_text": right_text,
                    "left_ipa": left_ipa,
                    "right_ipa": right_ipa,
                    "sound_source": anchor_surface,
                    "sound_source_ipa": item["anchor_ipa"],
                    "candidate": free_surface,
                    "candidate_ipa": item["free_ipa"],
                    "source_side": anchor_side,
                    "opposite_side": free_side,
                    "source_level": int(item["anchor"].get("level", -1) or -1),
                    "phonetic_score": phon,
                    "semantic_A_score": float(semantic_A_score),
                    "semantic_B_score": float(semantic_B_score),
                    "cross_semantic_score": 0.0,
                    "source_semantic_score": float(item["anchor"].get("semantic_score", 1.0) or 1.0),
                    "opposite_semantic_score": 0.0,
                    "quality_score": naturalness,
                    "naturalness_score": naturalness,
                    "surprise_score": float(surprise),
                    "lexical_novelty_bonus": float(novelty),
                    "commonness_bonus": commonness_bonus_from_item({"surface": free_surface, "source": item["neighbor"].get("source", "")}),
                    "bridge_type": "detached_sound_collision_needs_llm_judge",
                    "relation": "detached_sound_collision_needs_llm_judge",
                    "semantic_verified": False,
                    "semantic_relation": "detached_sound_collision_needs_llm_judge",
                    "affordance_stage": f"detached_{bucket_id}",
                    "phonetic_relation": phonetic_relation_label(phon, same_ipa=same_ipa),
                    "retrieval_bucket": bucket_id,
                    "retrieval_bucket_label": RETRIEVAL_BUCKET_LABELS.get(bucket_id, bucket_id),
                    "retrieval_source_pool": retrieval_source_pool,
                    "retrieval_target_pool": retrieval_target_pool,
                    "retrieval_anchor_pool": anchor_pool_label,
                    "retrieval_anchor_pool_actual": item["anchor_pool_actual"],
                    "retrieval_free_pool": free_pool_label,
                    "detached_anchor_side": anchor_side,
                    "detached_free_side": free_side,
                    "detached_anchor_surface": anchor_surface,
                    "detached_free_surface": free_surface,
                    "detached_anchor_ipa": item["anchor_ipa"],
                    "detached_free_ipa": item["free_ipa"],
                    "detached_anchor_semantic_similarity": float(sim),
                    "detached_semantic_distance_from_anchor": float(semantic_distance),
                    "detached_anchor_quality": float(anchor_quality),
                    "detached_generation_warning": "one_semantic_anchor_only",
                    "same_root_penalty_applied": False,
                }
                bdict["bridge_score"] = self._detached_score_for_bridge(bdict)
                bdict["llm_priority_score"] = bdict["bridge_score"]
                bdict["retrieval_pair_key"] = retrieval_pair_key_for_bridge(bdict)
                accepted_for_anchor.append(bdict)

            stats["accepted_before_anchor_cap"] = int(stats.get("accepted_before_anchor_cap", 0)) + int(len(accepted_for_anchor))
            accepted_for_anchor.sort(key=lambda x: (
                -float(x.get("bridge_score", 0.0) or 0.0),
                -float(x.get("phonetic_score", 0.0) or 0.0),
                surface_key(x.get("detached_free_surface", "")),
            ))
            kept_for_anchor = accepted_for_anchor[:DETACHED_MAX_PER_ANCHOR]
            stats["accepted_after_anchor_cap"] = int(stats.get("accepted_after_anchor_cap", 0)) + int(len(kept_for_anchor))
            if len(accepted_for_anchor) > len(kept_for_anchor):
                reject("per_anchor_cap", len(accepted_for_anchor) - len(kept_for_anchor))
            raw.extend(kept_for_anchor)

        raw.sort(key=lambda x: bucket_rank_sort_key(x))
        out = raw[:DETACHED_MAX_PER_BUCKET]
        stats["returned_after_bucket_cap"] = int(len(out))
        if len(raw) > len(out):
            reject("per_bucket_cap", len(raw) - len(out))
        self._detached_diagnostics.append(stats)
        for rank, bridge in enumerate(out, 1):
            bridge["retrieval_bucket_rank"] = rank
        return out

    def _bucketize_expansion_bridge(
        self,
        bridge: dict[str, Any],
        pool_index: dict[str, set[tuple[str, str]]] | None = None,
    ) -> dict[str, Any]:
        """Assign phonetic-neighbor expansion output to a bucket without dropping recall.

        Direct bucket matching is membership-based.  Expansion matching is a
        different baseline route: it starts from a semantic node on one side,
        asks the phonetic index for sound-neighbors, and then checks/proxies the
        neighbor against the opposite semantic field.  When the generated
        neighbor is not already present in the explicit opposite pool, it still
        represents the opposite expanded semantic side for retrieval purposes.

        Therefore:
          A0 source + B semantic/proxy neighbor -> B1_A0
          A1 source + B semantic/proxy neighbor -> A1_B1
          B0 source + A semantic/proxy neighbor -> A1_B0
          B1 source + A semantic/proxy neighbor -> A1_B1

        If the neighbor is actually in an explicit original/expanded opposite
        pool, use that concrete pool.  Otherwise use the opposite expanded pool
        as an expansion-proxy and record that fact in diagnostics fields.
        """
        source_side = clean(bridge.get("source_side", ""))
        source_level = int(bridge.get("source_level", 9) or 9)
        source_pool = clean(bridge.get("source_pool", bridge.get("retrieval_source_pool", "")))
        if not source_pool:
            if source_side == "A":
                source_pool = "A0" if source_level <= 0 else "A1"
            elif source_side == "B":
                source_pool = "B0" if source_level <= 0 else "B1"

        candidate_surface = clean(bridge.get("candidate", bridge.get("candidate_surface", bridge.get("right_text", ""))))
        candidate_ipa = clean(bridge.get("candidate_ipa", bridge.get("right_ipa", "")))
        candidate_key = retrieval_pool_member_key(candidate_surface, candidate_ipa)

        pool_index = pool_index or {}
        if source_side == "A":
            candidate_pool_order = ["B0", "B1"]
            proxy_target_pool = "B1"
        elif source_side == "B":
            candidate_pool_order = ["A0", "A1"]
            proxy_target_pool = "A1"
        else:
            candidate_pool_order = ["A0", "B0", "A1", "B1"]
            proxy_target_pool = ""

        target_pool = ""
        target_membership = ""
        for pool_name in candidate_pool_order:
            if candidate_key in pool_index.get(pool_name, set()):
                target_pool = pool_name
                target_membership = "explicit_pool_member"
                break

        if not target_pool and proxy_target_pool:
            target_pool = proxy_target_pool
            target_membership = "expansion_opposite_semantic_proxy"

        bucket_id = retrieval_bucket_for_pool_pair(source_pool, target_pool) if target_pool else ""

        out = dict(bridge)
        out["retrieval_source_pool_actual"] = source_pool
        out["retrieval_target_pool_actual"] = target_pool
        out["retrieval_expansion_target_membership"] = target_membership
        if not bucket_id:
            out["retrieval_bucket"] = ""
            out["retrieval_bucket_label"] = "unbucketed_expansion_candidate"
            out["retrieval_source_pool"] = source_pool
            out["retrieval_target_pool"] = target_pool
            out["retrieval_pair_key"] = retrieval_pair_key_for_bridge(out)
            return out

        regime = next((r for r in RETRIEVAL_BUCKET_REGIMES if r["bucket_id"] == bucket_id), None) or {}
        out["retrieval_bucket"] = bucket_id
        out["retrieval_bucket_label"] = RETRIEVAL_BUCKET_LABELS.get(bucket_id, bucket_id)
        # Store canonical regime pools for bucket ordering/diagnostics.  The
        # actual/proxy membership fields above preserve expansion provenance.
        out["retrieval_source_pool"] = regime.get("source_pool", source_pool)
        out["retrieval_target_pool"] = regime.get("target_pool", target_pool)
        out["retrieval_pair_key"] = retrieval_pair_key_for_bridge(out)
        return out


    def expand_side(self, terms: list[str], side: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return all semantic candidates and the subset with known IPA.

        Expansion order:
        1. Direct meaning seeds.
        2. BGE phrase/expression neighborhood.
        3. One bounded FastText lexical beam from direct seeds + top BGE terms.
        4. A second BGE semantic step using the compact expanded query.

        FastText is not recursive and never runs over every semantic node.
        """
        seeds = unique_keep_order(terms, limit=20)
        candidates = self._seed_candidates(seeds, side)

        q1 = " ".join(seeds)
        level1 = self._semantic_candidates(q1, side=side, level=1, top_k=SIDE_SEMANTIC_K) if q1 else []
        candidates.extend(level1)

        # Bounded FastText beam: direct seeds first, then a few top BGE phrase terms.
        ft_level1: list[dict[str, Any]] = []
        if self.fasttext is not None and self.fasttext.enabled:
            bge_seed_terms = unique_keep_order([x["surface"] for x in level1[:8]], limit=8)
            ft_seed_pool = unique_keep_order(seeds + bge_seed_terms, limit=FASTTEXT_SEED_LIMIT)
            ft_level1 = self.fasttext.expand(
                ft_seed_pool,
                side=side,
                level=1,
                limit=FASTTEXT_MAX_CANDIDATES_PER_SIDE,
            )
            candidates.extend(ft_level1)

        # Low-style semantic step outward via BGE, using a compact query that includes
        # direct seeds, top BGE terms, and selected FastText lexical drift terms.
        l1_terms = unique_keep_order([x["surface"] for x in level1[:12]], limit=12)
        ft_terms = unique_keep_order([x["surface"] for x in ft_level1[:12]], limit=12)
        q2 = " ".join(unique_keep_order(seeds[:6] + l1_terms + ft_terms[:8], limit=26))
        if FAST_BRIDGE_MINING and not BRIDGE_USE_LEVEL2:
            # Level-2 BGE expansion was a major per-row cost and mostly feeds the
            # later LLM judge with low-quality long-tail items.  In fast candidate
            # generation mode, keep level-0 seeds + level-1 BGE + FastText only.
            level2 = []
        else:
            level2 = self._semantic_candidates(q2, side=side, level=2, top_k=SIDE_LEVEL2_K) if q2 else []
        candidates.extend(level2)

        # Deduplicate by surface; keep the best semantic score / lowest level.
        by_key: dict[str, dict[str, Any]] = {}
        for cand in candidates:
            key = norm_text(cand.get("surface", cand.get("text", "")))
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = cand
            else:
                if (cand["semantic_score"], -cand["level"]) > (existing["semantic_score"], -existing["level"]):
                    by_key[key] = cand

        all_candidates = list(by_key.values())
        all_candidates.sort(key=lambda x: (x["level"], -float(x.get("semantic_score", 0.0))))

        # Collapse semantic nodes by lemma/root before IPA lookup. This prevents
        # plural/conjugation/accent variants from wasting phonetic probe budget.
        all_candidates = collapse_dicts_by_root(
            all_candidates,
            surface_key_name="surface",
            score_keys=("semantic_score", "quality_score"),
        )

        with_ipa: list[dict[str, Any]] = []
        for cand in all_candidates:
            for rec in self.phonetic.lookup_records(cand["surface"], limit=1):
                enriched = dict(cand)
                enriched.update(rec)
                enriched["quality_score"] = expression_quality({
                    "surface": cand.get("surface", ""),
                    "source": cand.get("source", ""),
                    "frequency": cand.get("frequency", ""),
                })
                with_ipa.append(enriched)
                break

        with_ipa.sort(key=lambda x: (x["level"], -float(x.get("semantic_score", 0.0)), -float(x.get("quality_score", 0.0))))
        with_ipa = collapse_dicts_by_root(
            with_ipa,
            surface_key_name="surface",
            score_keys=("semantic_score", "quality_score"),
            limit=MAX_IPA_CANDIDATES_PER_SIDE,
        )
        return all_candidates, with_ipa

    def _pairwise_phonetic(self, a: list[dict[str, Any]], b: list[dict[str, Any]]) -> np.ndarray:
        if not a or not b:
            return np.zeros((0, 0), dtype="float32")
        a_emb = self.phonetic.encode_ipa([x["ipa"] for x in a])
        b_emb = self.phonetic.encode_ipa([x["ipa"] for x in b])
        return np.matmul(a_emb, b_emb.T)

    def _cheap_expansion_score(self, item: dict[str, Any]) -> float:
        """Cheap recall score used before expensive semantic scoring."""
        level = int(item.get("source_level", 9))
        level_bonus = {0: 1.0, 1: 0.86, 2: 0.70}.get(level, 0.55)
        phon = float(item.get("phonetic_score", 0.0))
        final = float(item.get("phonetic_final_score", phon))
        source_sem = float(item.get("source_semantic_score", 0.0))
        cand = clean(item.get("candidate", ""))
        src = clean(item.get("source_text", ""))
        novelty = lexical_novelty_bonus(src, cand)
        quality = expression_quality({"surface": cand})
        bad_penalty = -0.40 if lexically_bad_candidate_surface(cand) else 0.0
        root_penalty = -0.20 if (same_root(src, cand) or structurally_trivial_variant(src, cand) or boring_morphophonetic_echo(src, cand)) else 0.0
        return (
            0.42 * phon
            + 0.12 * final
            + 0.18 * source_sem
            + 0.09 * level_bonus
            + 0.08 * quality
            + novelty
            + bad_penalty
            + root_penalty
        )

    def _prefilter_expansion_candidates(self, raw: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Collapse and diversify before BGE semantic scoring."""
        if not raw:
            return []
        for x in raw:
            x["cheap_bridge_score"] = float(self._cheap_expansion_score(x))

        raw = collapse_dicts_by_root(
            raw,
            surface_key_name="candidate",
            score_keys=("cheap_bridge_score", "phonetic_score", "source_semantic_score", "phonetic_final_score"),
            limit=max(limit * 3, limit),
        )

        selected: list[dict[str, Any]] = []
        seen_candidate_roots: set[str] = set()
        seen_source_roots: dict[str, int] = {}
        for item in sorted(raw, key=lambda x: float(x.get("cheap_bridge_score", 0.0)), reverse=True):
            cand = clean(item.get("candidate", ""))
            src = clean(item.get("source_text", ""))
            if not cand or not src:
                continue
            if lexically_bad_candidate_surface(cand):
                continue
            if structurally_trivial_variant(src, cand) or boring_morphophonetic_echo(src, cand):
                continue
            croot = rough_lemma_key(cand) or surface_key(cand)
            sroot = rough_lemma_key(src) or surface_key(src)
            if croot in seen_candidate_roots:
                continue
            if sroot and seen_source_roots.get(sroot, 0) >= 4:
                continue
            selected.append(item)
            if croot:
                seen_candidate_roots.add(croot)
            if sroot:
                seen_source_roots[sroot] = seen_source_roots.get(sroot, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def _expansion_bridges(
        self,
        source_nodes: list[dict[str, Any]],
        source_side: str,
        opposite_terms: list[str],
        opposite_side: str,
    ) -> list[dict[str, Any]]:
        """Phonetic expansion first, semantic check second.

        For each semantic node on one side, retrieve words/phrases that sound like it.
        Then score those phonetic neighbors against the opposite meaning field. This
        discovers bridges that direct A×B comparison misses.
        """
        opposite_query = " ".join(unique_keep_order(opposite_terms, limit=16))
        if not source_nodes or not clean(opposite_query):
            return []

        probes = sorted(
            source_nodes,
            key=lambda x: (int(x.get("level", 9)), -float(x.get("semantic_score", 0.0))),
        )[:PHONETIC_PROBE_BEAM]

        raw: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        probe_ipas = [clean(node.get("ipa", "")) for node in probes]
        neighbors_by_ipa = self.phonetic.search_many(
            probe_ipas,
            top_k=PHONETIC_NEIGHBORS_PER_PROBE,
        )

        for node in probes:
            query_ipa = clean(node.get("ipa", ""))
            if not query_ipa:
                continue

            neighbors = neighbors_by_ipa.get(query_ipa, [])
            for n in neighbors:
                phon = float(n.get("phonetic_score", n.get("final_score", 0.0)))
                if phon < MIN_EXPANSION_PHONETIC:
                    continue

                candidate = clean(n.get("word", ""))
                candidate_ipa = clean(n.get("ipa", ""))
                if not candidate or not candidate_ipa:
                    continue
                if lexically_bad_candidate_surface(candidate):
                    continue

                source_surface = clean(node.get("surface", node.get("text", "")))
                # Expansion route is specifically for discovering new lexical
                # affordances. Same-root variants are not bridges and are
                # removed before semantic scoring so they cannot dominate.
                if structurally_trivial_variant(source_surface, candidate):
                    continue

                key = (plural_surface_key(source_surface), plural_surface_key(candidate), candidate_ipa)
                if key in seen:
                    continue
                seen.add(key)

                raw.append({
                    "source_side": source_side,
                    "opposite_side": opposite_side,
                    "source_text": source_surface,
                    "source_ipa": query_ipa,
                    "source_level": int(node.get("level", -1)),
                    "source_pool": clean(node.get("retrieval_pool", "")),
                    "source_semantic_score": float(node.get("semantic_score", 0.0)),
                    "source_origin": node.get("source", ""),
                    "candidate": candidate,
                    "candidate_ipa": candidate_ipa,
                    "candidate_rhyme": n.get("rhyme", ""),
                    "candidate_consonant_skeleton": n.get("consonant_skeleton", ""),
                    "candidate_vowel_skeleton": n.get("vowel_skeleton", ""),
                    "phonetic_score": phon,
                    "phonetic_final_score": float(n.get("final_score", phon)),
                })

        if not raw:
            return []

        # Cheap structural/diversity prefilter before expensive BGE semantic scoring.
        # This preserves broad recall while avoiding semantic scoring over the full
        # phonetic expansion cross-product.
        raw = self._prefilter_expansion_candidates(
            raw,
            limit=min(
                MAX_SEMANTIC_SCORED_EXPANSION_CANDIDATES,
                max(8, PHONETIC_PROBE_BEAM * PHONETIC_NEIGHBORS_PER_PROBE),
            ),
        )
        if not raw:
            return []

        # In the LLM-judge architecture, do not spend tens of seconds BGE-scoring
        # every phonetic expansion candidate against the opposite field.  Generate
        # clean, diverse candidates quickly and let the LLM judge usefulness later.
        # Set RETRIEVAL_SKIP_BRIDGE_OPPOSITE_SEMANTIC=0 to restore the old expensive
        # behavior for ablation runs.
        semantic_verified = not (FAST_BRIDGE_MINING and SKIP_BRIDGE_OPPOSITE_SEMANTIC)
        if not semantic_verified:
            semantic_scores = []
            for x in raw:
                # Cheap proxy used only for pre-judge ordering.  It is NOT a
                # verified semantic match, so downstream labels must say
                # needs_judge rather than strong/opposite_semantic.
                quality = expression_quality({"surface": x.get("candidate", ""), "source": x.get("candidate_source", "")})
                source_sem = float(x.get("source_semantic_score", 0.0))
                phon = float(x.get("phonetic_score", 0.0))
                proxy = 0.18 + 0.14 * source_sem + 0.16 * quality + 0.06 * phon
                semantic_scores.append(float(max(0.12, min(0.50, proxy))))
        else:
            semantic_scores = self.expression.semantic_scores(
                opposite_query,
                [x["candidate"] for x in raw],
            )

        bridges: list[dict[str, Any]] = []
        for item, opp_sem in zip(raw, semantic_scores):
            opp_sem = float(opp_sem)
            if semantic_verified and opp_sem < MIN_OPPOSITE_SEMANTIC:
                continue

            same_surface = norm_text(item["source_text"]) == norm_text(item["candidate"])
            same_ipa = clean(item["source_ipa"]) == clean(item["candidate_ipa"])

            suffix = "to_opposite_semantic" if semantic_verified else "needs_judge"
            if same_surface and same_ipa:
                bridge_type = f"identity_polysemy_expansion_{suffix}"
                type_bonus = 0.04 if opp_sem >= 0.45 else -0.06
            elif trivial_inflection_related(item["source_text"], item["candidate"]):
                bridge_type = f"trivial_inflection_expansion_{suffix}"
                type_bonus = -0.18
            elif item["phonetic_score"] >= 0.96:
                bridge_type = f"expansion_homophone_{suffix}"
                type_bonus = 0.13
            elif item["phonetic_score"] >= 0.72:
                bridge_type = f"expansion_strong_phonetic_{suffix}"
                type_bonus = 0.08
            elif item["phonetic_score"] >= 0.58:
                bridge_type = f"expansion_near_phonetic_{suffix}"
                type_bonus = 0.02
            else:
                bridge_type = f"expansion_echo_{suffix}"
                type_bonus = -0.03

            level = int(item.get("source_level", 9))
            level_bonus = {0: 1.0, 1: 0.88, 2: 0.74}.get(level, 0.60)
            source_sem = float(item.get("source_semantic_score", 0.0))

            novelty = lexical_novelty_bonus(item["source_text"], item["candidate"])
            common_bonus = commonness_bonus_from_item({"surface": item["candidate"], "source": item.get("candidate_source", "")})
            naturalness = expression_quality({"surface": item["candidate"], "source": item.get("candidate_source", ""), "frequency": item.get("candidate_frequency", "")})
            same_root_flag = same_root(item["source_text"], item["candidate"]) or structurally_trivial_variant(item["source_text"], item["candidate"]) or boring_morphophonetic_echo(item["source_text"], item["candidate"])
            surprise = humor_surprise_score(float(item["phonetic_score"]), source_sem, opp_sem, same_root_flag)

            # bridge_score is retrieval priority only.  It intentionally gives
            # semantic verification a bonus, but it does not pretend proxy
            # semantics prove a strong bridge.
            semantic_weight = 0.16 if semantic_verified else 0.04
            bridge_score = (
                0.42 * float(item["phonetic_score"])
                + semantic_weight * opp_sem
                + 0.08 * source_sem
                + 0.13 * naturalness
                + 0.10 * surprise
                + 0.03 * level_bonus
                + type_bonus
                + novelty
                + common_bonus
            )

            bdict = {
                "bridge_type": bridge_type,
                "relation": bridge_type,
                "source_side": source_side,
                "opposite_side": opposite_side,
                "sound_source": item["source_text"],
                "sound_source_ipa": item["source_ipa"],
                "candidate": item["candidate"],
                "candidate_ipa": item["candidate_ipa"],
                "candidate_rhyme": item.get("candidate_rhyme", ""),
                "source_level": level,
                "source_semantic_score": source_sem,
                "opposite_semantic_score": opp_sem,
                "phonetic_score": float(item["phonetic_score"]),
                "bridge_score": float(bridge_score),
                "lexical_novelty_bonus": float(novelty),
                "commonness_bonus": float(common_bonus),
                "semantic_verified": bool(semantic_verified),
                "semantic_relation": "verified_opposite_semantic" if semantic_verified else "proxy_needs_llm_judge",
                "affordance_stage": f"{source_side}{level}_phon_to_{opposite_side}_{'verified' if semantic_verified else 'proxy'}",
                "phonetic_relation": phonetic_relation_label(float(item["phonetic_score"]), same_ipa=same_ipa),
                "naturalness_score": float(naturalness),
                "surprise_score": float(surprise),
                "same_root_penalty_applied": bool(same_root_flag),
            }
            bdict["llm_priority_score"] = llm_priority_score_for_bridge(bdict)
            bridges.append(bdict)

        bridges.sort(key=lambda x: (affordance_stage_rank(x.get("affordance_stage", ""), x.get("bridge_type", "")), -float(x.get("llm_priority_score", x.get("bridge_score", 0.0)))), reverse=False)
        return bridges[:MAX_EXPANSION_BRIDGES]

    def mine_bridges(self, a_terms: list[str], b_terms: list[str], semantic_A_results: list[dict[str, Any]] | None = None, semantic_B_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        stage_times: dict[str, float] = {}

        def mark(name: str, t0: float) -> None:
            stage_times[name] = round(time.time() - t0, 3)

        t = time.time()
        if RETRIEVAL_REUSE_ROW_SEMANTIC_FOR_BRIDGES:
            side_A = self.expand_bucketized_side_from_precomputed(a_terms, "A", semantic_A_results)
        else:
            side_A = self.expand_bucketized_side_from_precomputed(a_terms, "A", None)
        semantic_A = side_A["semantic_candidates"]
        ipa_A = side_A["with_ipa"]
        ipa_A0 = side_A["A0"]
        ipa_A1 = side_A["A1"]
        mark("bridge_expand_side_A", t)

        t = time.time()
        if RETRIEVAL_REUSE_ROW_SEMANTIC_FOR_BRIDGES:
            side_B = self.expand_bucketized_side_from_precomputed(b_terms, "B", semantic_B_results)
        else:
            side_B = self.expand_bucketized_side_from_precomputed(b_terms, "B", None)
        semantic_B = side_B["semantic_candidates"]
        ipa_B = side_B["with_ipa"]
        ipa_B0 = side_B["B0"]
        ipa_B1 = side_B["B1"]
        mark("bridge_expand_side_B", t)

        pool_map = {
            "A0": ipa_A0,
            "B0": ipa_B0,
            "A1": ipa_A1,
            "B1": ipa_B1,
        }
        pool_index = retrieval_pool_index(pool_map)
        bucket_raw: dict[str, list[dict[str, Any]]] = {bucket_id: [] for bucket_id in RETRIEVAL_BUCKET_ORDER}
        unbucketed_expansion_count = 0
        # Per-row detached diagnostics.  Reset here so every retrieval row reports
        # exactly where detached candidates were rejected.
        self._detached_diagnostics: list[dict[str, Any]] = []

        t = time.time()
        for regime in RETRIEVAL_BUCKET_REGIMES:
            bucket_id = regime["bucket_id"]
            bucket_raw[bucket_id] = self._run_bucket_pairwise(
                bucket_id,
                regime["source_pool"],
                regime["target_pool"],
                pool_map.get(regime["source_pool"], []),
                pool_map.get(regime["target_pool"], []),
            )
        mark("bridge_bucket_pairwise", t)

        # Route 2 from the baseline: phonetic-neighbor expansion first, semantic
        # check/proxy second.  This is not Low-leap. It is part of the existing
        # bridge miner and must remain active; bucketization only assigns each
        # expansion bridge to the appropriate first-class regime.
        t = time.time()
        expansion_A_to_B = [self._bucketize_expansion_bridge(b, pool_index) for b in self._expansion_bridges(ipa_A, "A", b_terms, "B")]
        mark("bridge_expansion_A_to_B", t)

        t = time.time()
        expansion_B_to_A = [self._bucketize_expansion_bridge(b, pool_index) for b in self._expansion_bridges(ipa_B, "B", a_terms, "A")]
        mark("bridge_expansion_B_to_A", t)

        for br in expansion_A_to_B + expansion_B_to_A:
            bucket_id = clean(br.get("retrieval_bucket", ""))
            if bucket_id in bucket_raw:
                bucket_raw[bucket_id].append(br)
            else:
                unbucketed_expansion_count += 1

        detached_A2_from_B_count = 0
        detached_B2_from_A_count = 0
        if DETACHED_BUCKETS_ENABLED:
            t = time.time()
            b_anchors = self._dedupe_anchor_items(list(ipa_B0) + list(ipa_B1))
            detached_A2_from_B = self._detached_sound_collision_bridges(
                b_anchors,
                anchor_pool_label="B01",
                free_pool_label="A2",
                bucket_id="A2_B01_DETACHED",
                anchor_side="B",
                free_side="A",
            )
            bucket_raw["A2_B01_DETACHED"].extend(detached_A2_from_B)
            detached_A2_from_B_count = len(detached_A2_from_B)
            mark("bridge_detached_A2_from_B01", t)

            t = time.time()
            a_anchors = self._dedupe_anchor_items(list(ipa_A0) + list(ipa_A1))
            detached_B2_from_A = self._detached_sound_collision_bridges(
                a_anchors,
                anchor_pool_label="A01",
                free_pool_label="B2",
                bucket_id="B2_A01_DETACHED",
                anchor_side="A",
                free_side="B",
            )
            bucket_raw["B2_A01_DETACHED"].extend(detached_B2_from_A)
            detached_B2_from_A_count = len(detached_B2_from_A)
            mark("bridge_detached_B2_from_A01", t)

        t = time.time()
        filtered_by_bucket: dict[str, list[dict[str, Any]]] = {bucket_id: [] for bucket_id in RETRIEVAL_BUCKET_ORDER}
        for bucket_id in RETRIEVAL_BUCKET_ORDER:
            for br in bucket_raw.get(bucket_id, []):
                left, right = bridge_surface_pair(br)
                if lexically_bad_candidate_surface(left) or lexically_bad_candidate_surface(right):
                    continue
                if structurally_trivial_variant(left, right):
                    continue
                filtered_by_bucket[bucket_id].append(br)
        mark("bridge_bucket_filter", t)

        t = time.time()
        selected: list[dict[str, Any]] = []
        seen_pairs: dict[str, dict[str, Any]] = {}
        duplicate_events: list[dict[str, Any]] = []
        counts_after_dedupe = {bucket_id: 0 for bucket_id in RETRIEVAL_BUCKET_ORDER}

        for bucket_id in RETRIEVAL_BUCKET_ORDER:
            ranked = sorted(filtered_by_bucket[bucket_id], key=bucket_rank_sort_key)
            local_rank = 0
            for br in ranked:
                pair_key = retrieval_pair_key_for_bridge(br)
                if pair_key in seen_pairs:
                    duplicate_events.append({
                        "pair_key": pair_key,
                        "kept_bucket": seen_pairs[pair_key].get("retrieval_bucket", ""),
                        "discarded_bucket": bucket_id,
                    })
                    continue
                local_rank += 1
                nb = dict(br)
                nb["retrieval_bucket"] = bucket_id
                nb["retrieval_bucket_rank"] = local_rank
                nb["retrieval_pair_key"] = pair_key
                selected.append(nb)
                seen_pairs[pair_key] = nb
                counts_after_dedupe[bucket_id] += 1
        mark("bridge_bucket_dedupe", t)

        t = time.time()
        bridges = order_bridges_by_retrieval_bucket(selected, PREFINAL_BRIDGE_CANDIDATE_LIMIT)
        prefinal_bridge_candidate_count = len(bridges)
        mark("bridge_bucket_select", t)

        diagnostics = bridge_diagnostics(bridges)
        diagnostics["retrieval_bucket_order"] = list(RETRIEVAL_BUCKET_ORDER)
        diagnostics["retrieval_bucket_regimes"] = [dict(x) for x in RETRIEVAL_BUCKET_REGIMES]
        diagnostics["retrieval_bucket_pool_counts"] = {k: len(v) for k, v in pool_map.items()}
        diagnostics["a1b1_widening_enabled"] = bool(A1B1_PER_SEED_SEMANTIC_ENABLED)
        diagnostics["fasttext_default_disabled_v15"] = True
        diagnostics["a1b1_per_seed_semantic_k"] = int(A1B1_PER_SEED_SEMANTIC_K)
        diagnostics["a1b1_per_seed_limit"] = int(A1B1_PER_SEED_LIMIT)
        diagnostics["a1_pool_size"] = int(len(ipa_A1))
        diagnostics["b1_pool_size"] = int(len(ipa_B1))
        diagnostics["retrieval_bucket_pool_surfaces"] = {
            k: [clean(x.get("surface", x.get("text", ""))) for x in v]
            for k, v in pool_map.items()
        }
        diagnostics["retrieval_bucket_counts_before_dedupe"] = {bucket_id: len(bucket_raw.get(bucket_id, [])) for bucket_id in RETRIEVAL_BUCKET_ORDER}
        diagnostics["retrieval_bucket_counts_before_filter"] = {bucket_id: len(bucket_raw.get(bucket_id, [])) for bucket_id in RETRIEVAL_BUCKET_ORDER}
        diagnostics["retrieval_bucket_counts_after_filter"] = {bucket_id: len(filtered_by_bucket.get(bucket_id, [])) for bucket_id in RETRIEVAL_BUCKET_ORDER}
        diagnostics["retrieval_bucket_counts_after_dedupe"] = counts_after_dedupe
        diagnostics["retrieval_bucket_counts_after_final_sanitize"] = retrieval_bucket_counts(bridges)
        diagnostics["retrieval_exported_bucket_counts"] = retrieval_bucket_counts(bridges)
        diagnostics["prefinal_bridge_candidate_limit"] = int(PREFINAL_BRIDGE_CANDIDATE_LIMIT)
        diagnostics["prefinal_bridge_candidate_count"] = int(prefinal_bridge_candidate_count)
        diagnostics["retrieval_duplicate_pair_events"] = duplicate_events
        diagnostics["direct_pair_bridge_count"] = sum(len(v) for v in bucket_raw.values()) - len(expansion_A_to_B) - len(expansion_B_to_A)
        diagnostics["expansion_A_to_B_count"] = len(expansion_A_to_B)
        diagnostics["expansion_B_to_A_count"] = len(expansion_B_to_A)
        diagnostics["detached_buckets_enabled"] = bool(DETACHED_BUCKETS_ENABLED)
        diagnostics["detached_A2_from_B01_count"] = int(detached_A2_from_B_count)
        diagnostics["detached_B2_from_A01_count"] = int(detached_B2_from_A_count)
        diagnostics["detached_neighbors_per_anchor"] = int(DETACHED_NEIGHBORS_PER_ANCHOR)
        diagnostics["detached_min_phonetic"] = float(DETACHED_MIN_PHONETIC)
        diagnostics["detached_max_anchor_semantic_sim"] = float(DETACHED_MAX_ANCHOR_SEMANTIC_SIM)
        diagnostics["detached_anti_echo_filter_enabled"] = True
        diagnostics["detached_low_value_verb_filter_enabled"] = True
        diagnostics["detached_commonness_filter_enabled"] = True
        diagnostics["detached_strong_orthographic_overlap_filter_enabled"] = True
        diagnostics["detached_clipped_shorthand_filter_enabled"] = True
        diagnostics["detached_min_free_recognizability"] = float(DETACHED_MIN_FREE_RECOGNIZABILITY)
        diagnostics["detached_max_per_anchor"] = int(DETACHED_MAX_PER_ANCHOR)
        diagnostics["detached_max_per_bucket"] = int(DETACHED_MAX_PER_BUCKET)
        detached_stats = list(getattr(self, "_detached_diagnostics", []) or [])
        diagnostics["detached_rejection_stats"] = detached_stats
        detached_totals: dict[str, int] = {}
        for st in detached_stats:
            for reason, count in (st.get("rejections", {}) or {}).items():
                detached_totals[reason] = int(detached_totals.get(reason, 0)) + int(count)
        diagnostics["detached_rejection_totals"] = detached_totals
        diagnostics["detached_neighbors_seen_total"] = int(sum(int(st.get("neighbors_seen", 0) or 0) for st in detached_stats))
        diagnostics["detached_candidates_before_semantic_distance_total"] = int(sum(int(st.get("candidates_before_semantic_distance", 0) or 0) for st in detached_stats))
        diagnostics["detached_semantic_distance_scored_total"] = int(sum(int(st.get("semantic_distance_scored", 0) or 0) for st in detached_stats))
        diagnostics["detached_accepted_before_anchor_cap_total"] = int(sum(int(st.get("accepted_before_anchor_cap", 0) or 0) for st in detached_stats))
        diagnostics["detached_accepted_after_anchor_cap_total"] = int(sum(int(st.get("accepted_after_anchor_cap", 0) or 0) for st in detached_stats))
        diagnostics["unbucketed_expansion_count"] = int(unbucketed_expansion_count)
        diagnostics["phonetic_probe_beam"] = PHONETIC_PROBE_BEAM
        diagnostics["phonetic_neighbors_per_probe"] = PHONETIC_NEIGHBORS_PER_PROBE
        diagnostics["min_expansion_phonetic"] = MIN_EXPANSION_PHONETIC
        diagnostics["min_opposite_semantic"] = MIN_OPPOSITE_SEMANTIC
        diagnostics["stage_times_sec"] = stage_times
        diagnostics["fast_bridge_mining"] = bool(FAST_BRIDGE_MINING)
        diagnostics["skip_bridge_opposite_semantic"] = bool(SKIP_BRIDGE_OPPOSITE_SEMANTIC)
        diagnostics["bridge_use_level2"] = bool(BRIDGE_USE_LEVEL2)
        diagnostics["reuse_row_semantic_for_bridges"] = bool(RETRIEVAL_REUSE_ROW_SEMANTIC_FOR_BRIDGES)
        diagnostics["use_fasttext"] = bool(self.fasttext is not None and self.fasttext.enabled)
        diagnostics["low_leap_enabled_in_primary_output"] = False
        if self.fasttext is not None:
            diagnostics.update(getattr(self.fasttext, "last_stats", {}))

        if bridges and any(bool(b.get("semantic_verified", False)) for b in bridges):
            fallback_level = "verified_affordances"
        elif bridges:
            fallback_level = "judge_ready_affordances"
        elif ipa_A or ipa_B:
            fallback_level = "phonetic_affordances_only"
        elif semantic_A or semantic_B:
            fallback_level = "semantic_only"
        else:
            fallback_level = "generator_only"

        return {
            "semantic_A_candidates": semantic_A[:30],
            "semantic_B_candidates": semantic_B[:30],
            "semantic_A_with_ipa_count": len(ipa_A),
            "semantic_B_with_ipa_count": len(ipa_B),
            "bridge_candidates": bridges,
            "bridge_diagnostics": diagnostics,
            "fallback_level": fallback_level,
        }

    def phonetic_affordances_for_terms(self, terms: list[str], side: str, top_k_each: int = 8, max_terms: int = 8) -> list[dict[str, Any]]:
        """Batch phonetic affordance search for the row's French terms.

        Same result contract as the old term-by-term search_from_text loop, but
        it performs one batched phonetic FAISS search for all known IPA probes.
        """
        selected_terms = unique_keep_order(terms, limit=max_terms)
        probe_records: list[tuple[str, str]] = []
        seen_probe_ipas: set[str] = set()
        for term in selected_terms:
            records = self.phonetic.lookup_records(term, limit=1)
            if not records:
                continue
            ipa = clean(records[0].get("ipa", ""))
            if not ipa:
                continue
            probe_records.append((term, ipa))
            seen_probe_ipas.add(ipa)

        neighbors_by_ipa = self.phonetic.search_many(list(seen_probe_ipas), top_k=top_k_each) if seen_probe_ipas else {}

        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for term, probe_ipa in probe_records:
            for r in neighbors_by_ipa.get(probe_ipa, []):
                word = clean(r.get("word", ""))
                ipa = clean(r.get("ipa", ""))
                # Do not waste affordance slots on the same trivial family as the probe.
                if structurally_trivial_variant(term, word) and surface_key(term) != surface_key(word):
                    continue
                key = phonetic_family_key(word, ipa)
                if key in seen:
                    continue
                seen.add(key)
                nr = dict(r)
                nr["probe_text"] = clean(term)
                nr["probe_ipa"] = probe_ipa
                nr["probe_side"] = side
                out.append(nr)
        out = collapse_phonetic_records_by_family(out, surface_key_name="word", limit=PHONETIC_K)
        out.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
        return out[:PHONETIC_K]


# ─────────────────────────────────────────────────────────────────────────────
# Unified retrieval pipeline
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalPipeline:
    def __init__(self) -> None:
        self.expression = ExpressionRetriever(semantic_k=SEMANTIC_K, lexical_k=LEXICAL_K)
        self.phonetic = PhoneticRetriever(top_k=PHONETIC_K)
        self.fasttext = FastTextExpansionBackend()
        self.bridge_miner = BridgeMiner(self.expression, self.phonetic, self.fasttext)

    def retrieve_row(self, row: pd.Series) -> dict[str, Any]:
        stage_times: dict[str, float] = {}

        def mark(name: str, t0: float) -> None:
            stage_times[name] = round(time.time() - t0, 3)

        t = time.time()
        a_terms, b_terms = side_terms(row)
        semantic_query = build_semantic_query(row)
        lexical_query = build_lexical_query(row)
        lexical_A_query = " ".join(a_terms)
        lexical_B_query = " ".join(b_terms)
        semantic_A_requests = semantic_side_requests(a_terms, "A", SEMANTIC_K)
        semantic_B_requests = semantic_side_requests(b_terms, "B", SEMANTIC_K)
        semantic_A_channels = [channel for _, _, channel in semantic_A_requests]
        semantic_B_channels = [channel for _, _, channel in semantic_B_requests]
        diagnostics_input = {
            "semantic_launch_mode": "per_term",
            "retrieval_input_mode": "french_lists_only",
            "meaning_A_term_count": len(a_terms),
            "meaning_B_term_count": len(b_terms),
        }
        mark("setup", t)

        t = time.time()
        semantic_blended_query = " ".join(
            unique_keep_order(a_terms + b_terms, limit=24)
        )

        semantic_many = self.expression.semantic_search_many(
            (
                [(semantic_blended_query, SEMANTIC_K, "semantic_blended")]
                if clean(semantic_blended_query)
                else []
            )
            + semantic_A_requests
            + semantic_B_requests
        )

        semantic_blended = semantic_many.get("semantic_blended", [])

        semantic_A = merge_semantic_side_results(semantic_many, semantic_A_channels)
        semantic_B = merge_semantic_side_results(semantic_many, semantic_B_channels)

        semantic_blended = unique_keep_order_dicts_by_surface(
            semantic_blended + semantic_A[:5] + semantic_B[:5],
            limit=SEMANTIC_K,
        )
        diagnostics_input["semantic_A_result_count_after_merge"] = len(semantic_A)
        diagnostics_input["semantic_B_result_count_after_merge"] = len(semantic_B)
        mark("semantic_search", t)

        t = time.time()
        lexical_blended = self.expression.lexical_search(lexical_query, top_k=LEXICAL_K, channel="lexical_blended")
        lexical_A = self.expression.lexical_search(lexical_A_query, top_k=LEXICAL_K, channel="lexical_A")
        lexical_B = self.expression.lexical_search(lexical_B_query, top_k=LEXICAL_K, channel="lexical_B")
        mark("lexical_search", t)

        t = time.time()
        bridge_result = self.bridge_miner.mine_bridges(a_terms, b_terms, semantic_A, semantic_B)
        mark("bridge_mining", t)

        t = time.time()
        phonetic_A = self.bridge_miner.phonetic_affordances_for_terms(a_terms, "A", top_k_each=PHONETIC_NEIGHBORS_PER_PROBE, max_terms=min(len(a_terms), PHONETIC_PROBE_BEAM))
        phonetic_B = self.bridge_miner.phonetic_affordances_for_terms(b_terms, "B", top_k_each=PHONETIC_NEIGHBORS_PER_PROBE, max_terms=min(len(b_terms), PHONETIC_PROBE_BEAM))
        mark("phonetic_affordances", t)

        t = time.time()
        diagnostics = dict(bridge_result["bridge_diagnostics"])
        diagnostics.update(diagnostics_input)
        merged_stage_times = dict(diagnostics.get("stage_times_sec", {}) or {})
        merged_stage_times.update(stage_times)
        diagnostics["stage_times_sec"] = merged_stage_times
        generator_affordance_pack = {
            "meaning_A_terms": a_terms[:8],
            "meaning_B_terms": b_terms[:8],
            "fallback_level": bridge_result["fallback_level"],
            "bridge_diagnostics": diagnostics,
            "top_bridge_candidates": export_bridge_candidates(bridge_result["bridge_candidates"], MAX_GENERATOR_AFFORDANCES),
            "top_semantic_A": semantic_A[:5],
            "top_semantic_B": semantic_B[:5],
            "top_semantic_blended": semantic_blended[:5],
            "top_phonetic_A": phonetic_A[:5],
            "top_phonetic_B": phonetic_B[:5],
        }

        out = {
            "meaning_A_terms": a_terms,
            "meaning_B_terms": b_terms,
            "semantic_query": semantic_query,
            "semantic_A_query": " | ".join(a_terms),
            "semantic_B_query": " | ".join(b_terms),
            "lexical_query": lexical_query,
            "semantic_expressions": semantic_blended,
            "semantic_A_expressions": semantic_A,
            "semantic_B_expressions": semantic_B,
            "lexical_expressions": lexical_blended,
            "lexical_A_expressions": lexical_A,
            "lexical_B_expressions": lexical_B,
            "phonetic_A_candidates": phonetic_A,
            "phonetic_B_candidates": phonetic_B,
            "semantic_A_with_ipa_count": bridge_result["semantic_A_with_ipa_count"],
            "semantic_B_with_ipa_count": bridge_result["semantic_B_with_ipa_count"],
            "bridge_candidates": bridge_result["bridge_candidates"],
            "bridge_diagnostics": diagnostics,
            "generator_affordance_pack": generator_affordance_pack,
            "fallback_level": bridge_result["fallback_level"],
        }
        mark("pack_assembly", t)
        diagnostics["stage_times_sec"] = stage_times

        # Integrated Low-leap generator-affordance behavior.
        try:
            current_ideas = _additive_generator_ideas_from_bridges(
                out.get("bridge_candidates", []) or [],
                limit=MAX_GENERATOR_IDEAS,
            )
            # Mine Low-leap candidates when the row is thin, but never as a competing
            # backfill lane. They are appended to the internal candidate pool and
            # exported additively by lane.
            if False and len(current_ideas) < LOW_LEAP_TARGET_GENERATOR_IDEAS:
                extra = _mine_iterative_low_leap_bridges(self, out, current_ideas)
                if extra:
                    existing = out.get("bridge_candidates", []) or []
                    # Exact bridge dedupe only; no root collapse and no score top-N.
                    merged: list[dict[str, Any]] = []
                    seen: set[tuple[str, str, str]] = set()
                    for b in list(existing) + list(extra):
                        key = _bridge_exact_export_key(b)
                        if key in seen:
                            continue
                        merged.append(b)
                        seen.add(key)
                    out["bridge_candidates"] = merged
                    diag = dict(out.get("bridge_diagnostics", {}) or {})
                    diag["low_leap_candidate_count"] = int(len(extra))
                    diag["bridge_count"] = int(len(merged))
                    diag["selection_policy"] = "additive_lanes_exact_dedupe_only"
                    out["bridge_diagnostics"] = diag
                    if "generator_affordance_pack" in out and isinstance(out["generator_affordance_pack"], dict):
                        gen = dict(out["generator_affordance_pack"])
                        gen["bridge_diagnostics"] = diag
                        gen["top_bridge_candidates"] = export_bridge_candidates(merged, MAX_GENERATOR_AFFORDANCES)
                        out["generator_affordance_pack"] = gen
        except Exception as e:
            if RETRIEVAL_DEBUG_PACKS:
                diag = dict(out.get("bridge_diagnostics", {}) or {})
                diag["low_leap_error"] = str(e)
                out["bridge_diagnostics"] = diag

        return out



def export_bridge_candidate(b: dict[str, Any]) -> dict[str, Any]:
    """Normalize heterogeneous bridge schemas for generator/debug output.

    Direct A×B bridges use a_surface/b_surface. Expansion bridges use
    sound_source/candidate. Downstream code should not need to know which route
    produced the bridge, so expose stable aliases while preserving key scores.
    """
    bridge_type = clean(b.get("bridge_type") or b.get("relation") or "")
    source_surface = clean(b.get("sound_source") or b.get("a_surface") or b.get("left_surface") or b.get("left_text") or "")
    candidate_surface = clean(b.get("candidate") or b.get("b_surface") or b.get("right_surface") or b.get("right_text") or "")
    source_ipa = clean(b.get("sound_source_ipa") or b.get("a_ipa") or b.get("left_ipa") or "")
    candidate_ipa = clean(b.get("candidate_ipa") or b.get("b_ipa") or b.get("right_ipa") or "")

    # Stable aliases for old generator/debug code.
    a_surface = clean(b.get("a_surface") or source_surface)
    b_surface = clean(b.get("b_surface") or candidate_surface)

    return {
        "bridge_type": bridge_type,
        "relation": clean(b.get("relation") or bridge_type),
        "retrieval_bucket": clean(b.get("retrieval_bucket", "")),
        "retrieval_bucket_rank": int(b.get("retrieval_bucket_rank", 0) or 0),
        "retrieval_pair_key": clean(b.get("retrieval_pair_key") or retrieval_pair_key_for_bridge(b)),
        "retrieval_source_pool": clean(b.get("retrieval_source_pool", "")),
        "retrieval_target_pool": clean(b.get("retrieval_target_pool", "")),
        "source_side": clean(b.get("source_side") or b.get("left_side") or ""),
        "opposite_side": clean(b.get("opposite_side") or b.get("right_side") or ""),
        "source_surface": source_surface,
        "candidate_surface": candidate_surface,
        "source_ipa": source_ipa,
        "candidate_ipa": candidate_ipa,
        "a_surface": a_surface,
        "b_surface": b_surface,
        "a_ipa": clean(b.get("a_ipa") or source_ipa),
        "b_ipa": clean(b.get("b_ipa") or candidate_ipa),
        "phonetic_score": float(b.get("phonetic_score", 0.0) or 0.0),
        "semantic_A_score": float(b.get("semantic_A_score", 0.0) or 0.0),
        "semantic_B_score": float(b.get("semantic_B_score", 0.0) or 0.0),
        "opposite_semantic_score": float(b.get("opposite_semantic_score", 0.0) or 0.0),
        "cross_semantic_score": float(b.get("cross_semantic_score", 0.0) or 0.0),
        "source_semantic_score": float(b.get("source_semantic_score", 0.0) or 0.0),
        "bridge_score": float(b.get("bridge_score", 0.0) or 0.0),
        "llm_priority_score": float(b.get("llm_priority_score", llm_priority_score_for_bridge(b)) or 0.0),
        "naturalness_score": float(b.get("naturalness_score", b.get("quality_score", 0.0)) or 0.0),
        "surprise_score": float(b.get("surprise_score", 0.0) or 0.0),
        "pivotability_score": float(b.get("pivotability_score", bridge_pivotability_score(b)) or 0.0),
        "affordance_bucket": affordance_bucket_for_bridge(b),
        "semantic_verified": bool(b.get("semantic_verified", False)),
        "semantic_relation": clean(b.get("semantic_relation", "")),
        "affordance_stage": clean(b.get("affordance_stage", b.get("stage", ""))),
        "phonetic_relation": clean(b.get("phonetic_relation", "")),
        "source_level": int(b.get("source_level", b.get("left_level", -1)) or -1),
        "left_level": int(b.get("left_level", b.get("source_level", -1)) or -1),
        "right_level": int(b.get("right_level", -1) or -1),
        "is_trivial_inflection": bool(b.get("is_trivial_inflection", trivial_inflection_related(source_surface, candidate_surface))),
        "same_root_penalty_applied": bool(b.get("same_root_penalty_applied", same_root(source_surface, candidate_surface))),
        "lexical_novelty_bonus": float(b.get("lexical_novelty_bonus", 0.0) or 0.0),
        "commonness_bonus": float(b.get("commonness_bonus", 0.0) or 0.0),
        "cross_side_collision_bonus": float(b.get("cross_side_collision_bonus", 0.0) or 0.0),
        "detached_anchor_side": clean(b.get("detached_anchor_side", "")),
        "detached_free_side": clean(b.get("detached_free_side", "")),
        "detached_anchor_surface": clean(b.get("detached_anchor_surface", "")),
        "detached_free_surface": clean(b.get("detached_free_surface", "")),
        "detached_anchor_ipa": clean(b.get("detached_anchor_ipa", "")),
        "detached_free_ipa": clean(b.get("detached_free_ipa", "")),
        "detached_anchor_semantic_similarity": float(b.get("detached_anchor_semantic_similarity", 0.0) or 0.0),
        "detached_semantic_distance_from_anchor": float(b.get("detached_semantic_distance_from_anchor", 0.0) or 0.0),
        "detached_generation_warning": clean(b.get("detached_generation_warning", "")),
    }


def affordance_bucket_for_bridge(b: dict[str, Any]) -> str:
    """Generator-facing diversity bucket. Buckets are descriptive, not quality labels."""
    btype = clean(b.get("bridge_type", b.get("relation", "")))
    phon_rel = clean(b.get("phonetic_relation", ""))
    phon = clamp01(b.get("phonetic_score", 0.0))
    nat = clamp01(b.get("naturalness_score", b.get("quality_score", 0.0)))
    pivot = max(clamp01(b.get("pivotability_score", 0.0)), bridge_pivotability_score(b))
    surprise = clamp01(b.get("surprise_score", 0.0))
    left, right = bridge_surface_pair(b)
    phrase = len(clean(left).split()) > 1 or len(clean(right).split()) > 1

    if phrase:
        return "phrase_level"
    if "homophone" in btype or phon >= 0.96 or phon_rel == "exact_or_near_homophone":
        if nat >= 0.36 and pivot >= 0.34:
            return "safe_homophone"
        return "risky_homophone"
    if phon >= 0.82 and surprise >= 0.58 and pivot >= 0.30:
        return "surprising_near_phonetic"
    return "creative_risky"


def bridge_diversity_signature(b: dict[str, Any]) -> tuple[str, str, str, str, str]:
    left, right = bridge_surface_pair(b)
    return (
        affordance_bucket_for_bridge(b),
        clean(b.get("source_side", "")),
        clean(b.get("phonetic_relation", "")),
        rough_lemma_key(left),
        rough_lemma_key(right),
    )


def sort_bridge_key_for_generator(b: dict[str, Any]) -> tuple[int, float, float, float, float]:
    return (
        affordance_stage_rank(b.get("affordance_stage", ""), b.get("bridge_type", "")),
        -clamp01(b.get("llm_priority_score", b.get("bridge_score", 0.0))),
        -clamp01(b.get("pivotability_score", 0.0)),
        -clamp01(b.get("surprise_score", 0.0)),
        -clamp01(b.get("phonetic_score", 0.0)),
    )


def diversify_bridge_candidates_for_generator(bridges: list[dict[str, Any]], limit: int = MAX_GENERATOR_AFFORDANCES) -> list[dict[str, Any]]:
    """Keep multiple distinct affordances instead of one scalar winner.

    This is intentionally less aggressive than final structural sanitation: the
    generator needs a small menu of different possibilities, not a single best
    bridge.  It keeps the strongest item per exact diversity signature first,
    then backfills across buckets until the limit is reached.
    """
    if not bridges:
        return []
    ordered = sorted(bridges, key=sort_bridge_key_for_generator)
    buckets: dict[str, list[dict[str, Any]]] = {
        "safe_homophone": [],
        "surprising_near_phonetic": [],
        "phrase_level": [],
        "risky_homophone": [],
        "creative_risky": [],
    }
    seen_sig: set[tuple[str, str, str, str, str]] = set()
    for b in ordered:
        sig = bridge_diversity_signature(b)
        if sig in seen_sig:
            continue
        bucket = affordance_bucket_for_bridge(b)
        if bucket not in buckets:
            bucket = "creative_risky"
        if len(buckets[bucket]) >= MAX_AFFORDANCES_PER_BUCKET:
            continue
        buckets[bucket].append(b)
        seen_sig.add(sig)

    out: list[dict[str, Any]] = []
    # Interleave buckets so exact homophones do not suppress creative/phrase options.
    bucket_order = ["safe_homophone", "surprising_near_phonetic", "phrase_level", "risky_homophone", "creative_risky"]
    while len(out) < limit:
        added = False
        for bucket in bucket_order:
            if buckets[bucket]:
                out.append(buckets[bucket].pop(0))
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break

    # Backfill if the signature/bucket caps left room.
    selected_ids = {id(x) for x in out}
    for b in ordered:
        if len(out) >= limit:
            break
        if id(b) in selected_ids:
            continue
        out.append(b)
        selected_ids.add(id(b))
    return out[:limit]


def bucket_exported_affordances(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "safe_homophone": [],
        "surprising_near_phonetic": [],
        "phrase_level": [],
        "risky_homophone": [],
        "creative_risky": [],
    }
    for c in candidates:
        b = clean(c.get("affordance_bucket", "")) or affordance_bucket_for_bridge(c)
        if b not in buckets:
            b = "creative_risky"
        buckets[b].append(c)
    return {k: v for k, v in buckets.items() if v}










# ─────────────────────────────────────────────────────────────────────────────
# Compact storage helpers
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# Dataset execution and debugging
# ─────────────────────────────────────────────────────────────────────────────


def load_preprocessor_translation_outputs(model: str) -> pd.DataFrame:
    path = f"{translate_dir}{model}/"
    log("Loading translated outputs:", path)
    return load_all(path)


def retrieve_dataset(df: pd.DataFrame, model: str, start: int = 0, end: int = -1) -> None:
    validate_input(df)
    retriever = RetrievalPipeline()

    chunks = [df.iloc[i : i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
    if end == -1:
        end = len(chunks)

    run_name = model.replace("/", "__")
    out_dir = OUTPUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(start, end):
        chunk = chunks[i]
        compact_packs: list[str] = []
        debug_packs: list[str] = []
        semantic_counts: list[int] = []
        lexical_counts: list[int] = []
        phonetic_counts: list[int] = []
        bridge_counts: list[int] = []
        strong_bridge_counts: list[int] = []
        identity_bridge_counts: list[int] = []
        different_surface_bridge_counts: list[int] = []
        best_bridge_scores: list[float] = []
        fallback_levels: list[str] = []

        for _, row in chunk.iterrows():
            pack = retriever.retrieve_row(row)
            compact_pack = compact_retrieval_pack(pack)
            compact_packs.append(json.dumps(compact_pack, ensure_ascii=False))
            if RETRIEVAL_DEBUG_PACKS:
                debug_packs.append(json.dumps(pack, ensure_ascii=False))
            semantic_counts.append(len(pack["semantic_expressions"]))
            lexical_counts.append(len(pack["lexical_expressions"]))
            phonetic_counts.append(
                len(pack["phonetic_A_candidates"]) + len(pack["phonetic_B_candidates"])
            )
            diag = pack.get("bridge_diagnostics", {})
            bridge_counts.append(int(diag.get("bridge_count", len(pack["bridge_candidates"]))))
            strong_bridge_counts.append(int(diag.get("strong_bridge_count", 0)))
            identity_bridge_counts.append(int(diag.get("identity_bridge_count", 0)))
            different_surface_bridge_counts.append(int(diag.get("different_surface_bridge_count", 0)))
            best_bridge_scores.append(float(diag.get("best_bridge_score", 0.0)))
            fallback_levels.append(pack["fallback_level"])

        chunk["retrieval_pack_compact"] = compact_packs
        if RETRIEVAL_DEBUG_PACKS:
            chunk["retrieval_pack"] = debug_packs
        chunk["retrieval_semantic_count"] = semantic_counts
        chunk["retrieval_lexical_count"] = lexical_counts
        chunk["retrieval_phonetic_count"] = phonetic_counts
        chunk["retrieval_bridge_count"] = bridge_counts
        chunk["retrieval_strong_bridge_count"] = strong_bridge_counts
        chunk["retrieval_identity_bridge_count"] = identity_bridge_counts
        chunk["retrieval_different_surface_bridge_count"] = different_surface_bridge_counts
        chunk["retrieval_best_bridge_score"] = best_bridge_scores
        chunk["retrieval_fallback_level"] = fallback_levels

        out_path = out_dir / f"{i}.tsv"
        save(chunk, str(out_path))
        log(f"Saved {out_path} rows={len(chunk)}")


def debug_expression(query: str, top_k: int = 10) -> None:
    retriever = ExpressionRetriever(semantic_k=top_k, lexical_k=top_k)
    print("\nSEMANTIC\n")
    for x in retriever.semantic_search(query, top_k=top_k):
        print(f'{x["score"]:.4f}\t{x["surface"]}\t{x["source"]}')
    print("\nLEXICAL\n")
    for x in retriever.lexical_search(query, top_k=top_k):
        print(f'{x["score"]:.4f}\t{x["surface"]}\t{x["source"]}')


def debug_phonetic(query_ipa: str, top_k: int = 10) -> None:
    retriever = PhoneticRetriever(top_k=top_k)
    print("\n" + "=" * 80)
    print("QUERY:", query_ipa)
    print("=" * 80)
    results = retriever.search(query_ipa, top_k=top_k)
    for i, r in enumerate(results, 1):
        print(f'{i:02d}  {r["final_score"]:.4f}  {r["ipa"]}  {r["word"]}')



def debug_lemma() -> None:
    """Print morphology keys for known noisy bridge pairs."""
    pairs = [
        ("brûlant", "brulant"),
        ("séduisant", "séduisants"),
        ("fumant", "fumants"),
        ("insignifiance", "insignifiances"),
        ("zéro", "zéros"),
        ("acte", "actes"),
        ("courbé", "courbés"),
        ("brûlaient", "brûlait"),
        ("conte", "compte"),
        ("couvant", "couvent"),
    ]
    print("spaCy enabled:", USE_SPACY_LEMMAS)
    print("spaCy model:", SPACY_MODEL)
    # Trigger lazy load once so warnings happen before table output.
    _get_spacy_fr()
    for a, b in pairs:
        print()
        print(f"{a!r}: surface={surface_key(a)!r} spacy={_spacy_lemma_key_cached(norm_text(a))!r} crude={crude_fr_root_key(a)!r} rough={rough_lemma_key(a)!r}")
        print(f"{b!r}: surface={surface_key(b)!r} spacy={_spacy_lemma_key_cached(norm_text(b))!r} crude={crude_fr_root_key(b)!r} rough={rough_lemma_key(b)!r}")
        print("same_root:", same_root(a, b), "trivial_inflection:", trivial_inflection_related(a, b))


def debug_row_local(model: str, row_idx: int = 0) -> None:
    df = load_preprocessor_translation_outputs(model)
    validate_input(df)
    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"row_idx {row_idx} outside 0..{len(df)-1}")

    row = df.iloc[row_idx]
    retriever = RetrievalPipeline()
    pack = retriever.retrieve_row(row)

    print("\nROW", row_idx)
    print("A:", pack["meaning_A_terms"])
    print("B:", pack["meaning_B_terms"])
    print("fallback_level:", pack["fallback_level"])
    print("semantic_A_with_ipa_count:", pack["semantic_A_with_ipa_count"])
    print("semantic_B_with_ipa_count:", pack["semantic_B_with_ipa_count"])
    print("bridge_count:", len(pack["bridge_candidates"]))
    print("bridge_diagnostics:", json.dumps(pack.get("bridge_diagnostics", {}), ensure_ascii=False))

    print("\nTOP BRIDGES")
    for b in pack["bridge_candidates"][:10]:
        print(
            f'{b["bridge_score"]:.4f}\tphon={b["phonetic_score"]:.4f}\t'
            f'type={b.get("bridge_type", b.get("relation", ""))}\t'
            f'A:{b["left_text"]} [{b["left_ipa"]}]  <~>  B:{b["right_text"]} [{b["right_ipa"]}]'
        )

    print("\nTOP SEMANTIC A")
    for x in pack["semantic_A_expressions"][:5]:
        print(f'{x["score"]:.4f}\t{x["surface"]}\t{x["source"]}')

    print("\nTOP SEMANTIC B")
    for x in pack["semantic_B_expressions"][:5]:
        print(f'{x["score"]:.4f}\t{x["surface"]}\t{x["source"]}')



# ─────────────────────────────────────────────────────────────────────────────
# Hot-reload server entrypoints + thin CLI
# ─────────────────────────────────────────────────────────────────────────────

SERVER = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8765")
HTTP_TIMEOUT = int(os.environ.get("RETRIEVAL_TIMEOUT", "3600"))
PROCESSED_RETRIEVAL_DIR = ROOT / "data" / "processed" / "retrieval"
TRACE_DIR = PROCESSED_RETRIEVAL_DIR / "traces"
EVAL_DIR = PROCESSED_RETRIEVAL_DIR / "eval"
PACK_DIR = PROCESSED_RETRIEVAL_DIR / "packs"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_save(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _tsv_save(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _bridge_side_fields(b: dict[str, Any]) -> tuple[str, str, str, str]:
    """Normalize both direct-pair and expansion bridge schemas for exports."""
    left_text = clean(
        b.get("left_text")
        or b.get("sound_source")
        or b.get("source_surface")
        or b.get("a_surface")
        or ""
    )
    right_text = clean(
        b.get("right_text")
        or b.get("candidate")
        or b.get("candidate_surface")
        or b.get("b_surface")
        or ""
    )
    left_ipa = clean(
        b.get("left_ipa")
        or b.get("sound_source_ipa")
        or b.get("source_ipa")
        or ""
    )
    right_ipa = clean(
        b.get("right_ipa")
        or b.get("candidate_ipa")
        or ""
    )
    return left_text, right_text, left_ipa, right_ipa




# ─────────────────────────────────────────────────────────────────────────────
# FINAL LIVE QUALITY GUARD
# ─────────────────────────────────────────────────────────────────────────────
# This section is intentionally late in the file so hot-reload changes here take
# effect without rebuilding RetrievalPipeline assets.  It fixes the production
# failure mode where phonetic neighbors such as séduisant→déduisant/réduisant or
# convaincant→convainquant survived because they are not exact plural variants
# but are still boring same-family morphological echoes.

_FRENCH_PARTICIPLE_ENDINGS = (
    "ant", "ants", "ante", "antes",
    "isant", "isants", "isante", "isantes",
    "issant", "issants", "issante", "issantes",
)

_INFLECTIONAL_ENDINGS = (
    "ai", "ais", "ait", "aient", "as", "a", "at", "ât", "ames", "âmes",
    "ez", "er", "era", "eras", "erez", "erai", "erais", "erait", "eront",
    "e", "es", "ent", "ons", "ant", "ants", "ante", "antes",
    "i", "is", "it", "issent", "issait", "issais", "issaient", "issant",
)

_PREFIXES_TO_IGNORE_FOR_ECHO = (
    "de", "dé", "re", "ré", "in", "im", "en", "em", "con", "com", "sur", "sous", "a", "at",
)


def _plain_word(x: Any) -> str:
    s = strip_accents(norm_text(x))
    s = re.sub(r"[^a-zœæ]+", "", s)
    return s


def _levenshtein(a: str, b: str, max_cutoff: int = 6) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_cutoff:
        return max_cutoff + 1
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(val)
            row_min = min(row_min, val)
        if row_min > max_cutoff:
            return max_cutoff + 1
        prev = cur
    return prev[-1]


def _longest_common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _longest_common_suffix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        n += 1
    return n


def _strip_light_prefix(s: str) -> str:
    for p in sorted(_PREFIXES_TO_IGNORE_FOR_ECHO, key=len, reverse=True):
        if len(s) > len(p) + 5 and s.startswith(p):
            return s[len(p):]
    return s


def boring_morphophonetic_echo(a: Any, b: Any) -> bool:
    """Reject candidates that are only same-family phonetic morphology.

    This is intentionally stronger than structurally_trivial_variant().  It
    catches French productive echoes that are useless for joke retrieval:
      séduisant→déduisant/réduisant/induisant/conduisant
      convaincant→convainquant
      brûlant→brûlaient
      fumant→fumai/fuma/fumât/fûmes
    while preserving genuinely useful small collisions such as:
      plaisant→plaignant, attrayant→attrapant, couvant→coupant, suie→suit, très→trait.
    """
    sa, sb = _plain_word(a), _plain_word(b)
    if not sa or not sb or sa == sb:
        return bool(sa and sb and sa == sb)

    # Existing hard guards still apply.
    if structurally_trivial_variant(a, b):
        return True

    min_len = min(len(sa), len(sb))
    if min_len < 4:
        return False

    edit = _levenshtein(sa, sb, max_cutoff=4)
    lcp = _longest_common_prefix_len(sa, sb)
    lcs = _longest_common_suffix_len(sa, sb)

    # Same long stem with only an inflectional tail change.
    # brûlant/brûlaient, éblouissant/éblouissait.
    if lcp >= 5 and edit <= 4:
        tail_a = sa[lcp:]
        tail_b = sb[lcp:]
        if tail_a in _INFLECTIONAL_ENDINGS or tail_b in _INFLECTIONAL_ENDINGS:
            return True
        if sa.endswith(_FRENCH_PARTICIPLE_ENDINGS) or sb.endswith(_FRENCH_PARTICIPLE_ENDINGS):
            # Avoid rejecting attrayant/attrapant: their shared prefix is only 5
            # but the change creates a different stem before -ant.  Require a
            # very long prefix or tiny edit distance.
            if lcp >= 6 or edit <= 2:
                return True

    # Same long suffix with only a light prefix/consonant change.
    # séduisant/déduisant/réduisant/induisant/conduisant.
    if lcs >= 6 and lcs / max(1, min_len) >= 0.65:
        # Keep short true homophones like suie/suit and très/trait; this only
        # triggers on long shared endings.
        return True

    # Prefix-stripped forms collapse: convaincant/convainquant, etc.
    aa, bb = _strip_light_prefix(sa), _strip_light_prefix(sb)
    if aa != sa or bb != sb:
        if aa == bb:
            return True
        if min(len(aa), len(bb)) >= 6 and _levenshtein(aa, bb, max_cutoff=3) <= 2:
            return True

    # Very small edit on a long participial adjective is usually spelling or
    # conjugational noise, not a pun affordance.  This catches convaincant→convainquant
    # but does not catch plaisant→plaignant because edit distance is larger.
    if min_len >= 8 and edit <= 2:
        if sa.endswith(_FRENCH_PARTICIPLE_ENDINGS) or sb.endswith(_FRENCH_PARTICIPLE_ENDINGS):
            return True

    return False



# ─────────────────────────────────────────────────────────────────────────────
# Hard final universal cleaner helpers
# ─────────────────────────────────────────────────────────────────────────────

_HARD_BAD_BRIDGE_SURFACES = {
    # productive echoes around séduisant that look good phonetically but are not useful puns
    "deduisant", "reduisant", "induisant", "conduisant", "enduisant",
    # smoking row morphology/conjugation noise
    "brulaient", "brulait", "brulais", "fumai", "fuma", "fumat", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes", "fumes",
    "fumames", "fumates", "fumera", "fumerai", "fumerais", "fumerait",
    # spelling-only participle alternation
    "convainquant",
}

_USEFUL_ALLOWLIST_PAIRS = {
    tuple(sorted(("plaisant", "plaignant"))),
    tuple(sorted(("attrayant", "attrapant"))),
    tuple(sorted(("couvant", "coupant"))),
    tuple(sorted(("couvant", "coudant"))),
    tuple(sorted(("couvant", "cousant"))),
    tuple(sorted(("suie", "suit"))),
    tuple(sorted(("suie", "suis"))),
    tuple(sorted(("tres", "trait"))),
    tuple(sorted(("tres", "traits"))),
    tuple(sorted(("brulant", "brelan"))),
}

_PRODUCTIVE_PREFIXES = (
    "de", "re", "in", "im", "en", "em", "con", "com", "sur", "sous",
)

def _surface_plain(x: Any) -> str:
    return _plain_word(x)


def _strip_productive_prefixes_for_final_guard(s: str) -> str:
    changed = True
    while changed:
        changed = False
        for pref in sorted(_PRODUCTIVE_PREFIXES, key=len, reverse=True):
            if len(s) > len(pref) + 5 and s.startswith(pref):
                s = s[len(pref):]
                changed = True
                break
    return s


def _final_bridge_pair_allowed(a: Any, b: Any) -> bool:
    sa, sb = _surface_plain(a), _surface_plain(b)
    return tuple(sorted((sa, sb))) in _USEFUL_ALLOWLIST_PAIRS


def universal_trivial_bridge(a: Any, b: Any) -> bool:
    """Last-resort production cleaner for final bridge outputs.

    This is intentionally applied AFTER the normal pipeline returns and BEFORE
    any bridge is exposed/saved.  It is stricter than the internal miner because
    final output should prefer fewer, better bridges over morphology spam.
    """
    sa, sb = _surface_plain(a), _surface_plain(b)
    if not sa or not sb:
        return True
    if _final_bridge_pair_allowed(sa, sb):
        return False
    if sa == sb:
        return True
    if sa in _HARD_BAD_BRIDGE_SURFACES or sb in _HARD_BAD_BRIDGE_SURFACES:
        return True
    if structurally_trivial_variant(sa, sb):
        return True

    min_len = min(len(sa), len(sb))
    if min_len < 4:
        return False

    edit = _levenshtein(sa, sb, max_cutoff=5)
    lcp = _longest_common_prefix_len(sa, sb)
    lcs = _longest_common_suffix_len(sa, sb)

    # Very long common suffix with only a productive prefix/consonant swap:
    # séduisant/déduisant/réduisant/induisant/conduisant.
    if lcs >= 6 and lcs / max(1, min_len) >= 0.62:
        return True

    # Long same stem, only conjugational/participial ending changes:
    # brûlant/brûlaient, éblouissant/éblouissait, fumant/fumai/fuma/fumât.
    if lcp >= 5 and edit <= 5:
        tail_a = sa[lcp:]
        tail_b = sb[lcp:]
        if tail_a in _INFLECTIONAL_ENDINGS or tail_b in _INFLECTIONAL_ENDINGS:
            return True
        if sa.endswith(_FRENCH_PARTICIPLE_ENDINGS) or sb.endswith(_FRENCH_PARTICIPLE_ENDINGS):
            if lcp >= 6 or edit <= 3:
                return True

    # Productive-prefix stripping produces same/almost-same stem.
    aa = _strip_productive_prefixes_for_final_guard(sa)
    bb = _strip_productive_prefixes_for_final_guard(sb)
    if (aa != sa or bb != sb) and min(len(aa), len(bb)) >= 5:
        if aa == bb or _levenshtein(aa, bb, max_cutoff=3) <= 2:
            return True

    # Same crude root and same phonetic/orthographic participial family.
    if rough_lemma_key(sa) and rough_lemma_key(sa) == rough_lemma_key(sb):
        if edit <= 4 or lcp >= 5 or lcs >= 5:
            return True

    return False



_BAD_VERB_INFLECTION_SUFFIXES = (
    "assions", "assiez", "assent", "assais", "assait", "assaient",
    "erions", "eriez", "eraient", "erais", "erait",
    "irions", "iriez", "iraient", "irais", "irait",
    "èrent", "erent", "aient",
)
_BAD_SURFACE_PATTERNS = (
    r"^[a-zàâçéèêëîïôûùüÿñæœ]{4,}assions$",
    r"^[a-zàâçéèêëîïôûùüÿñæœ]{4,}assent$",
    r"^[a-zàâçéèêëîïôûùüÿñæœ]{4,}èrent$",
    r"^[a-zàâçéèêëîïôûùüÿñæœ]{4,}aient$",
)
_GOOD_SHORT_FUNCTION_WORDS = {"très", "trait", "suie", "suit", "thé", "nom", "nul", "donc", "acte", "texte"}


def lexically_bad_candidate_surface(x: Any) -> bool:
    """Cheap final lexical-quality filter for the candidate-generator path.

    This does not try to be a full French POS tagger. It only removes forms that
    repeatedly poisoned retrieval: rare conjugations/subjunctives/imperfect plural
    variants and empty/markup surfaces. We keep this cheap so retrieval can focus
    on recall for the later LLM judge.
    """
    raw = clean(x)
    if not raw:
        return True
    txt = norm_text(raw)
    plain = strip_accents(txt)
    if txt in _GOOD_SHORT_FUNCTION_WORDS or plain in {strip_accents(w) for w in _GOOD_SHORT_FUNCTION_WORDS}:
        return False
    if len(plain) <= 2:
        return True
    if any(ch in raw for ch in "{}[]<>|="):
        return True
    # Disallow long weird conjugational forms that are almost never good pun pivots.
    for pat in _BAD_SURFACE_PATTERNS:
        if re.match(pat, txt):
            return True
    if len(plain) >= 7 and plain.endswith(_BAD_VERB_INFLECTION_SUFFIXES):
        return True
    # Also catch stripped-accent variants of -èrent etc.
    if len(plain) >= 7 and plain.endswith(("erent", "aient", "assions", "assent")):
        return True
    return False

def is_detached_bridge(b: dict[str, Any]) -> bool:
    marker = " ".join([
        clean(b.get("retrieval_bucket", "")),
        clean(b.get("bridge_type", "")),
        clean(b.get("relation", "")),
        clean(b.get("semantic_relation", "")),
        clean(b.get("affordance_stage", "")),
    ]).lower()
    return "detached" in marker





_DETACHED_VERB_FORM_ALLOWLIST = {
    "danse", "dense", "froid", "effroi", "pris", "quoi", "cois",
    "temps", "tant", "tante", "conte", "compte", "comte",
}


def detached_low_value_verb_form_reason(surface: Any) -> str:
    """Reject detached free candidates that are likely bare conjugation junk."""
    raw = norm_text(surface)
    plain = _plain_word(surface)
    if not plain:
        return "empty_detached_verb_surface"
    allow_plain = {_plain_word(x) for x in _DETACHED_VERB_FORM_ALLOWLIST}
    if raw in _DETACHED_VERB_FORM_ALLOWLIST or plain in allow_plain:
        return ""

    if len(plain) >= 5 and plain.endswith(("ez", "iez")):
        return "detached_low_value_ez_verb_form"
    if len(plain) >= 6 and plain.endswith(("ent", "aient", "erent", "issent")):
        return "detached_low_value_ent_verb_form"

    if len(plain) >= 6 and plain.endswith(("er", "ir", "re")):
        if surface_recognizability_prior(raw) < 0.30:
            return "detached_low_value_infinitive_like_form"

    if len(plain) >= 6 and plain.endswith(("ee", "ees", "ue", "ues", "ie", "ies")):
        if surface_recognizability_prior(raw) < 0.20:
            return "detached_low_value_participle_like_form"

    return ""



def detached_low_commonness_reason(surface: Any) -> str:
    """Reject detached free candidates that are too rare/technical/inflectional-looking.

    Detached buckets have no semantic support on the free side. A detached free
    candidate therefore needs to be recognizable enough for the LLM/user to use as
    a pun pivot. This targets rare forms like comptât, engluait, maigrît while
    preserving common words and recognizable nouns/adjectives.
    """
    raw = norm_text(surface)
    plain = _plain_word(surface)
    if not plain:
        return "empty_detached_commonness_surface"

    recognizability = surface_recognizability_prior(raw)
    quality = expression_quality({"surface": raw, "source": ""})
    commonness = max(recognizability, quality)

    if len(plain) >= 5 and plain.endswith(("at", "ait", "aient", "it", "irent")):
        if commonness < 0.55:
            return "detached_rare_conjugation_commonness"

    if len(plain) >= 6 and plain.endswith(("at", "it")) and commonness < 0.60:
        return "detached_rare_literary_verb_commonness"

    if len(plain) >= 8 and commonness < DETACHED_MIN_FREE_RECOGNIZABILITY:
        return "detached_long_low_commonness"

    if commonness < 0.16:
        return "detached_very_low_commonness"

    return ""


def _char_ngrams_for_detached(s: str, n: int) -> set[str]:
    s = _plain_word(s)
    if len(s) < n:
        return set()
    return {s[i:i+n] for i in range(0, len(s) - n + 1)}


def _detached_ngram_jaccard(a: str, b: str, n: int) -> float:
    aa = _char_ngrams_for_detached(a, n)
    bb = _char_ngrams_for_detached(b, n)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def _detached_ordered_subsequence_ratio(shorter: str, longer: str) -> float:
    """How much of shorter appears in longer in order, not necessarily contiguously."""
    s = _plain_word(shorter)
    l = _plain_word(longer)
    if not s or not l:
        return 0.0
    j = 0
    for ch in l:
        if j < len(s) and s[j] == ch:
            j += 1
    return j / max(1, len(s))


def detached_high_orthographic_overlap_reason(anchor: Any, free: Any) -> str:
    """Stronger detached-only visual-overlap suppression.

    Targets remaining junk such as exporté~explorer, excepté~accepter,
    essor~efforts, and pare~parts.  This is intentionally stricter than the
    ordinary anti-echo guard because detached candidates otherwise have no
    semantic support on the free side.
    """
    a = _plain_word(anchor)
    f = _plain_word(free)
    if not a or not f:
        return "empty_anchor_or_free_for_ortho_overlap"
    if a == f:
        return "same_plain_surface_ortho"

    min_len = min(len(a), len(f))
    max_len = max(len(a), len(f))
    if min_len <= 2:
        return "too_short_ortho_overlap_surface"

    short, long = (a, f) if len(a) <= len(f) else (f, a)
    edit = _levenshtein(a, f, max_cutoff=8)
    lcp = _longest_common_prefix_len(a, f)
    lcs = _longest_common_suffix_len(a, f)
    tri = _detached_ngram_jaccard(a, f, 3)
    four = _detached_ngram_jaccard(a, f, 4)
    subseq = _detached_ordered_subsequence_ratio(short, long)

    # Short forms with almost all letters shared are almost always weak visual
    # echoes in detached mode: pare~parts, pars~parts, cois~quoi-like cases.
    if min_len <= 4:
        shared = len(set(a) & set(f)) / max(1, len(set(short)))
        if shared >= 0.75 and edit <= 3:
            return "short_high_letter_overlap_echo"
        if subseq >= 0.80 and edit <= 3:
            return "short_subsequence_echo"

    # Long substring/near-substring relation.
    if min_len >= 4 and short in long and (max_len - min_len) <= 5:
        return "substring_containment_ortho_echo"

    # High trigram/fourgram reuse plus small edit distance catches exporté~explorer,
    # excepté~accepter, effort~essor style leaks.
    if min_len >= 5:
        if tri >= 0.34 and edit <= 5:
            return "high_trigram_overlap_echo"
        if four >= 0.25 and edit <= 6:
            return "high_fourgram_overlap_echo"

    # Long common prefix/suffix is especially bad in detached buckets.
    if min_len >= 5:
        if lcp >= 4 and (lcp / min_len) >= 0.55:
            return "strong_shared_prefix_ortho_echo"
        if lcs >= 4 and (lcs / min_len) >= 0.55:
            return "strong_shared_suffix_ortho_echo"

    # Same ordered skeleton means it often feels like a mechanical mutation.
    if min_len >= 5 and subseq >= 0.82 and edit <= 6:
        return "ordered_subsequence_ortho_echo"

    return ""


def detached_clipped_or_shorthand_reason(surface: Any) -> str:
    """Reject obvious clipped/domain shorthand detached pivots.

    Examples: compta-like business shorthand and fragmentary clipped forms.
    """
    raw = norm_text(surface)
    plain = _plain_word(surface)
    if not plain:
        return "empty_detached_shorthand_surface"

    # Common clipped/domain abbreviations that are valid French but poor detached
    # pun pivots in this dataset.
    explicit_bad = {
        "compta", "proba", "maths", "info", "admin", "promo",
    }
    if raw in explicit_bad or plain in explicit_bad:
        return "detached_clipped_shorthand_surface"

    # Short final-a/o clipped style with low recognizability.
    if 5 <= len(plain) <= 7 and plain.endswith(("a", "o")):
        if surface_recognizability_prior(raw) < 0.45:
            return "detached_probable_clipped_shorthand"

    return ""



def detached_orthographic_echo_reason(anchor: Any, free: Any) -> str:
    """Detached-bucket anti-echo guard.

    Catches mechanical spelling/inflection echoes that survive ordinary root
    filters, such as pars~parts, essor~efforts, essorent~efforts,
    remarié~remarquer, dépendue~défendu.
    """
    a = _plain_word(anchor)
    f = _plain_word(free)
    if not a or not f:
        return "empty_anchor_or_free_for_echo_check"
    if a == f:
        return "same_plain_surface"

    min_len = min(len(a), len(f))
    if min_len <= 2:
        return "too_short_detached_surface"

    if min_len <= 4:
        edit = _levenshtein(a, f, max_cutoff=2)
        if edit <= 1:
            return "short_one_edit_echo"
        if a.startswith(f) or f.startswith(a):
            return "short_prefix_echo"

    edit = _levenshtein(a, f, max_cutoff=5)
    lcp = _longest_common_prefix_len(a, f)
    lcs = _longest_common_suffix_len(a, f)

    if min_len >= 4 and (a.startswith(f) or f.startswith(a)):
        if abs(len(a) - len(f)) <= 4:
            return "contains_surface_extension_echo"

    if min_len >= 5:
        if lcp >= 4 and lcp / max(1, min_len) >= 0.62 and edit <= 4:
            return "long_shared_prefix_echo"
        if lcs >= 4 and lcs / max(1, min_len) >= 0.62 and edit <= 4:
            return "long_shared_suffix_echo"

    if min_len >= 6:
        aset = set(a)
        fset = set(f)
        char_jaccard = len(aset & fset) / max(1, len(aset | fset))
        if char_jaccard >= 0.72 and edit <= 5:
            return "high_character_overlap_echo"

    aa = _strip_productive_prefixes_for_final_guard(a)
    ff = _strip_productive_prefixes_for_final_guard(f)
    if (aa != a or ff != f) and min(len(aa), len(ff)) >= 4:
        if aa == ff:
            return "productive_prefix_same_stem_echo"
        if _levenshtein(aa, ff, max_cutoff=3) <= 2:
            return "productive_prefix_near_stem_echo"

    overlap_reason = detached_high_orthographic_overlap_reason(anchor, free)
    if overlap_reason:
        return overlap_reason

    return ""


def detached_bridge_final_rejection_reason(b: dict[str, Any]) -> str:
    """Return why a detached bridge fails final sanitation, or '' if valid.

    This is intentionally separate from bridge_is_structurally_valid(). Detached
    bridges have one semantic anchor only; they must NOT be rejected for missing
    opposite semantic verification or for the normal hard pivotability gate.
    """
    left, right = bridge_surface_pair(b)
    if not left or not right:
        return "missing_left_or_right_surface"
    if lexically_bad_candidate_surface(left):
        return "lexically_bad_left_surface"
    if lexically_bad_candidate_surface(right):
        return "lexically_bad_right_surface"
    free_for_verb_check = clean(b.get("detached_free_surface", b.get("candidate", right)))
    verb_reason = detached_low_value_verb_form_reason(free_for_verb_check)
    if verb_reason:
        return verb_reason + "_final"
    commonness_reason = detached_low_commonness_reason(free_for_verb_check)
    if commonness_reason:
        return commonness_reason + "_final"
    shorthand_reason = detached_clipped_or_shorthand_reason(free_for_verb_check)
    if shorthand_reason:
        return shorthand_reason + "_final"
    if surface_key(left) == surface_key(right):
        return "same_surface_pair"
    if same_root(left, right):
        return "same_root_pair"
    if structurally_trivial_variant(left, right):
        return "structurally_trivial_pair"
    if boring_morphophonetic_echo(left, right):
        return "boring_morphophonetic_echo_pair"
    if universal_trivial_bridge(left, right):
        return "universal_trivial_pair"

    phon = float(b.get("phonetic_score", 0.0) or 0.0)
    if phon < DETACHED_MIN_PHONETIC:
        return "phonetic_below_detached_min"

    naturalness = float(b.get("naturalness_score", b.get("quality_score", 0.0)) or 0.0)
    recognizability = max(surface_recognizability_prior(left), surface_recognizability_prior(right))
    if max(naturalness, recognizability) < DETACHED_MIN_NATURALNESS:
        return "naturalness_or_recognizability_below_detached_min"

    anchor_sim = float(b.get("detached_anchor_semantic_similarity", 0.0) or 0.0)
    if anchor_sim >= DETACHED_MAX_ANCHOR_SEMANTIC_SIM:
        return "semantic_too_close_to_anchor"

    anchor = clean(b.get("detached_anchor_surface", b.get("sound_source", "")))
    free = clean(b.get("detached_free_surface", b.get("candidate", "")))
    if anchor and free:
        if surface_key(anchor) == surface_key(free):
            return "same_surface_anchor_free"
        if same_root(anchor, free):
            return "same_root_anchor_free"
        if structurally_trivial_variant(anchor, free):
            return "structurally_trivial_anchor_free"
        if boring_morphophonetic_echo(anchor, free):
            return "boring_morphophonetic_echo_anchor_free"
        if universal_trivial_bridge(anchor, free):
            return "universal_trivial_anchor_free"
        echo_reason = detached_orthographic_echo_reason(anchor, free)
        if echo_reason:
            return echo_reason + "_final"

    return ""


def detached_bridge_is_structurally_valid(b: dict[str, Any]) -> bool:
    return detached_bridge_final_rejection_reason(b) == ""

def bridge_is_structurally_valid(b: dict[str, Any]) -> bool:
    if is_detached_bridge(b):
        return detached_bridge_is_structurally_valid(b)
    left, right = bridge_surface_pair(b)
    sound_source = clean(b.get("sound_source", b.get("source_surface", "")))
    candidate = clean(b.get("candidate", b.get("candidate_surface", "")))

    # Direct pair guard.
    if not left or not right:
        return False
    if lexically_bad_candidate_surface(left) or lexically_bad_candidate_surface(right):
        return False
    if universal_trivial_bridge(left, right):
        return False
    if boring_morphophonetic_echo(left, right):
        return False

    # Expansion route guard. This catches cases where exported left/right fields
    # differ from the actual sound_source→candidate relationship.
    if sound_source and candidate:
        if lexically_bad_candidate_surface(candidate):
            return False
        if universal_trivial_bridge(sound_source, candidate):
            return False
        if boring_morphophonetic_echo(sound_source, candidate):
            return False

    # Broad/proxy routes must be phonetically strong enough to justify LLM cost.
    phon = float(b.get("phonetic_score", 0.0) or 0.0)
    st_rank = affordance_stage_rank(b.get("affordance_stage", ""), b.get("bridge_type", ""))
    if st_rank >= 2 and phon < MIN_LLM_CANDIDATE_PHONETIC_BROAD:
        return False
    if st_rank <= 1 and phon < MIN_LLM_CANDIDATE_PHONETIC_STRICT:
        return False

    naturalness = float(b.get("naturalness_score", b.get("quality_score", 0.0)) or 0.0)
    recognizability = max(surface_recognizability_prior(left), surface_recognizability_prior(right))
    exactish = phon >= 0.96 or clean(b.get("phonetic_relation", "")) == "exact_or_near_homophone"
    pivotability = max(float(b.get("pivotability_score", 0.0) or 0.0), bridge_pivotability_score(b))
    if naturalness < MIN_LLM_CANDIDATE_NATURALNESS and recognizability < MIN_LLM_CANDIDATE_NATURALNESS:
        # Exact homophony alone is not enough: axions/action and saque/sacs are
        # technically neat but low-value for native French comedy without extra
        # evidence.  Let common recognizable homophones through; drop the rest.
        return False
    if not exactish and naturalness < MIN_LLM_CANDIDATE_NATURALNESS:
        return False
    if pivotability < MIN_LLM_CANDIDATE_PIVOTABILITY:
        return False

    # Function/support-word sound sources create many perfect but weak accidents
    # (e.g. très→trait, être→entre, avoir→voire).  Keep them only when the
    # opposite semantic evidence is actually verified/strong; otherwise they are
    # not worth sending to the generator.
    src_plain = strip_accents(norm_text(sound_source or left))
    cand_plain = strip_accents(norm_text(candidate or right))
    low_pivot_forms = {strip_accents(x) for x in _LOW_PIVOT_FUNCTION_SURFACES}
    if src_plain in low_pivot_forms or cand_plain in low_pivot_forms:
        sem_evidence = max(
            clamp01(b.get("source_semantic_score", 0.0)),
            clamp01(b.get("opposite_semantic_score", 0.0)),
            clamp01(b.get("semantic_A_score", 0.0)),
            clamp01(b.get("semantic_B_score", 0.0)),
        )
        if not bool(b.get("semantic_verified", False)) and sem_evidence < 0.50:
            return False

    return True



_LAST_FINAL_SANITIZE_DIAGNOSTICS: dict[str, Any] = {}


def final_sanitize_bridge_list(bridges: list[dict[str, Any]], limit: int = MAX_BRIDGES) -> list[dict[str, Any]]:
    """Final no-exceptions bridge guard used by the live server path.

    v6 adds explicit detached final rejection accounting. Detached bridges are
    routed through detached_bridge_final_rejection_reason() and never through the
    normal semantic/pivotability gate.
    """
    global _LAST_FINAL_SANITIZE_DIAGNOSTICS

    cleaned: list[dict[str, Any]] = []
    seen_pair_keys: set[str] = set()
    root_counts: dict[str, int] = {}

    final_diag: dict[str, Any] = {
        "input_count": int(len(bridges or [])),
        "detached_input_count": 0,
        "detached_kept_count": 0,
        "detached_rejections": {},
        "detached_rejections_by_bucket": {},
        "non_detached_rejected_structural": 0,
        "duplicate_pair_rejections": 0,
        "root_cap_rejections": 0,
    }

    def reject_detached(bucket_id: str, reason: str) -> None:
        totals = final_diag.setdefault("detached_rejections", {})
        totals[reason] = int(totals.get(reason, 0)) + 1
        by_bucket = final_diag.setdefault("detached_rejections_by_bucket", {})
        bstats = by_bucket.setdefault(bucket_id or "unknown", {})
        bstats[reason] = int(bstats.get(reason, 0)) + 1

    for b in order_bridges_by_retrieval_bucket(bridges or []):
        detached = is_detached_bridge(b)
        bucket_id = clean(b.get("retrieval_bucket", ""))
        if detached:
            final_diag["detached_input_count"] = int(final_diag.get("detached_input_count", 0)) + 1
            reason = detached_bridge_final_rejection_reason(b)
            if reason:
                reject_detached(bucket_id, reason)
                continue
        else:
            if not bridge_is_structurally_valid(b):
                final_diag["non_detached_rejected_structural"] = int(final_diag.get("non_detached_rejected_structural", 0)) + 1
                continue

        left, right = bridge_surface_pair(b)
        key = retrieval_pair_key_for_bridge(b)
        if key in seen_pair_keys:
            if detached:
                reject_detached(bucket_id, "duplicate_pair_key_final")
            final_diag["duplicate_pair_rejections"] = int(final_diag.get("duplicate_pair_rejections", 0)) + 1
            continue

        # Avoid many variants from one sound family dominating top-k.
        lr = rough_lemma_key(left) or surface_key(left)
        rr = rough_lemma_key(right) or surface_key(right)
        if lr and root_counts.get("L:" + lr, 0) >= MAX_BRIDGES_PER_ROOT:
            if detached:
                reject_detached(bucket_id, "left_root_cap_final")
            final_diag["root_cap_rejections"] = int(final_diag.get("root_cap_rejections", 0)) + 1
            continue
        if rr and root_counts.get("R:" + rr, 0) >= MAX_BRIDGES_PER_ROOT:
            if detached:
                reject_detached(bucket_id, "right_root_cap_final")
            final_diag["root_cap_rejections"] = int(final_diag.get("root_cap_rejections", 0)) + 1
            continue

        # Mark the actual hard guard for debugging.
        b = dict(b)
        b["retrieval_pair_key"] = b.get("retrieval_pair_key") or key
        b["structural_guard_passed"] = True
        if detached:
            b["detached_final_guard_passed"] = True

        # Backfill naturalness for bridges produced by a pre-hot-reload miner or
        # by phonetic neighbors without source/frequency metadata.
        b["naturalness_score"] = max(
            float(b.get("naturalness_score", b.get("quality_score", 0.0)) or 0.0),
            surface_recognizability_prior(left),
            surface_recognizability_prior(right),
        )
        b["surprise_score"] = max(
            float(b.get("surprise_score", 0.0) or 0.0),
            humor_surprise_score(
                float(b.get("phonetic_score", 0.0) or 0.0),
                float(b.get("source_semantic_score", b.get("semantic_A_score", 0.0)) or 0.0),
                float(b.get("opposite_semantic_score", b.get("semantic_B_score", 0.0)) or 0.0),
                bool(b.get("same_root_penalty_applied", False)),
            ),
        )
        b["pivotability_score"] = max(
            float(b.get("pivotability_score", 0.0) or 0.0),
            bridge_pivotability_score(b),
        )
        b["llm_priority_score"] = llm_priority_score_for_bridge(b)
        b["same_root_penalty_applied"] = bool(
            same_root(left, right)
            or structurally_trivial_variant(left, right)
            or boring_morphophonetic_echo(left, right)
        )
        cleaned.append(b)
        seen_pair_keys.add(key)
        if detached:
            final_diag["detached_kept_count"] = int(final_diag.get("detached_kept_count", 0)) + 1
        if lr:
            root_counts["L:" + lr] = root_counts.get("L:" + lr, 0) + 1
        if rr:
            root_counts["R:" + rr] = root_counts.get("R:" + rr, 0) + 1
        if len(cleaned) >= limit:
            break

    final_diag["output_count"] = int(len(cleaned))
    _LAST_FINAL_SANITIZE_DIAGNOSTICS = final_diag
    return cleaned

def final_sanitize_phonetic_pool(records: list[dict[str, Any]], probe_field: str = "probe_text", limit: int = PHONETIC_K) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in records or []:
        word = clean(r.get("word", r.get("surface", "")))
        ipa = clean(r.get("ipa", ""))
        probe = clean(r.get(probe_field, ""))
        if lexically_bad_candidate_surface(word):
            continue
        if probe and word and universal_trivial_bridge(probe, word):
            continue
        key = phonetic_family_key(word, ipa)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def final_sanitize_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Apply the final live guard to every route after the pipeline returns.

    This is the safety net that prevents older BridgeMiner internals from leaking
    morphology spam into the server response.  It is deliberately applied in
    build_row_pack(), so hot reloading retrieval_refactored.py changes behavior without
    touching or restarting retrieval_server.py.
    """
    pack = dict(pack)
    old_diag = dict(pack.get("bridge_diagnostics", {}) or {})
    pack["phonetic_A_candidates"] = final_sanitize_phonetic_pool(pack.get("phonetic_A_candidates", []), limit=PHONETIC_K)
    pack["phonetic_B_candidates"] = final_sanitize_phonetic_pool(pack.get("phonetic_B_candidates", []), limit=PHONETIC_K)

    bridges = final_sanitize_bridge_list(pack.get("bridge_candidates", []), limit=MAX_BRIDGES)
    pack["bridge_candidates"] = bridges
    new_diag = bridge_diagnostics(bridges)
    # Preserve runtime diagnostics from the miner before replacing counts.
    for k, v in old_diag.items():
        if k not in new_diag:
            new_diag[k] = v
    if "stage_times_sec" in old_diag:
        new_diag["stage_times_sec"] = dict(old_diag.get("stage_times_sec") or {})
    if "retrieval_bucket_order" not in new_diag:
        new_diag["retrieval_bucket_order"] = list(RETRIEVAL_BUCKET_ORDER)
    if "retrieval_bucket_regimes" not in new_diag:
        new_diag["retrieval_bucket_regimes"] = [dict(x) for x in RETRIEVAL_BUCKET_REGIMES]
    final_sanitize_diag = dict(globals().get("_LAST_FINAL_SANITIZE_DIAGNOSTICS", {}) or {})
    new_diag["final_sanitize_diagnostics"] = final_sanitize_diag
    new_diag["detached_final_rejection_totals"] = dict(final_sanitize_diag.get("detached_rejections", {}) or {})
    new_diag["detached_final_rejections_by_bucket"] = dict(final_sanitize_diag.get("detached_rejections_by_bucket", {}) or {})
    new_diag["detached_final_input_count"] = int(final_sanitize_diag.get("detached_input_count", 0) or 0)
    new_diag["detached_final_kept_count"] = int(final_sanitize_diag.get("detached_kept_count", 0) or 0)
    final_bucket_counts = retrieval_bucket_counts(bridges)
    new_diag["retrieval_bucket_counts_after_final_sanitize"] = final_bucket_counts
    new_diag["retrieval_exported_bucket_counts"] = final_bucket_counts
    pack["bridge_diagnostics"] = new_diag
    if bridges and any(bool(b.get("semantic_verified", False)) for b in bridges):
        pack["fallback_level"] = "verified_affordances"
    elif bridges:
        pack["fallback_level"] = "judge_ready_affordances"
    else:
        pack["fallback_level"] = "no_bridge"

    if "generator_affordance_pack" in pack and isinstance(pack["generator_affordance_pack"], dict):
        gen = dict(pack["generator_affordance_pack"])
        gen["bridge_diagnostics"] = pack["bridge_diagnostics"]
        gen["top_bridge_candidates"] = export_bridge_candidates(bridges, MAX_GENERATOR_AFFORDANCES)
        gen["top_phonetic_A"] = pack.get("phonetic_A_candidates", [])[:5]
        gen["top_phonetic_B"] = pack.get("phonetic_B_candidates", [])[:5]
        gen["fallback_level"] = pack["fallback_level"]
        pack["generator_affordance_pack"] = gen
    return pack



def _hot_swap_instance_class(obj: Any, cls: type) -> Any:
    """Use newly hot-reloaded methods on an already-loaded object.

    retrieval_server.py intentionally keeps model/index assets alive.  That means
    nested objects created at server startup (BridgeMiner, PhoneticRetriever,
    ExpressionRetriever, FastTextExpansionBackend) also keep their old class
    methods unless we explicitly swap them.  Swapping __class__ preserves all
    heavyweight attributes while making logic-only edits take effect immediately.
    """
    if obj is None:
        return None
    if obj.__class__ is cls:
        return obj
    try:
        obj.__class__ = cls
    except TypeError:
        pass
    return obj

def _get_pipeline(assets: dict[str, Any] | None = None) -> RetrievalPipeline:
    """Return the persistent pipeline, but attach the latest hot-reloaded class.

    The server builds RetrievalPipeline once and stores it in ASSETS.  On each
    request the server reloads this module, so this function swaps the existing
    instance onto the newly loaded RetrievalPipeline class.  That keeps the heavy
    loaded models/indexes in memory while allowing method/threshold/reranking
    changes in retrieval_refactored.py to take effect without restarting the server.
    """
    if assets is None:
        return RetrievalPipeline()

    pipeline = assets.get("_retrieval_pipeline")
    if pipeline is None:
        pipeline = RetrievalPipeline()
        assets["_retrieval_pipeline"] = pipeline
        return pipeline

    # Hot-reload behavior: keep the same loaded instance, use the latest methods.
    if pipeline.__class__ is not RetrievalPipeline:
        try:
            pipeline.__class__ = RetrievalPipeline
        except TypeError:
            # Fallback should be rare; it reloads only if class layout becomes incompatible.
            # Normal bridge/ranking/canonicalization edits should never hit this.
            pipeline = RetrievalPipeline()
            assets["_retrieval_pipeline"] = pipeline

    # Critical: hot-swap nested helper objects too.  Otherwise edits to
    # BridgeMiner/PhoneticRetriever methods do not take effect until a server
    # restart, even though retrieval_refactored.py was reloaded.
    if hasattr(pipeline, "expression"):
        _hot_swap_instance_class(pipeline.expression, ExpressionRetriever)
    if hasattr(pipeline, "phonetic"):
        _hot_swap_instance_class(pipeline.phonetic, PhoneticRetriever)
    if hasattr(pipeline, "fasttext") and pipeline.fasttext is not None:
        _hot_swap_instance_class(pipeline.fasttext, FastTextExpansionBackend)
    if hasattr(pipeline, "bridge_miner"):
        _hot_swap_instance_class(pipeline.bridge_miner, BridgeMiner)
        # Preserve loaded backends but point the miner at the hot-swapped objects.
        try:
            pipeline.bridge_miner.expression = pipeline.expression
            pipeline.bridge_miner.phonetic = pipeline.phonetic
            pipeline.bridge_miner.fasttext = pipeline.fasttext
        except Exception:
            pass
    return pipeline


def _get_dataset_cached(assets: dict[str, Any] | None, model: str) -> pd.DataFrame:
    if assets is not None:
        cache = assets.setdefault("_dataset_cache", {})
        if model in cache:
            return cache[model]
        df = load_preprocessor_translation_outputs(model)
        validate_input(df)
        cache[model] = df
        return df
    df = load_preprocessor_translation_outputs(model)
    validate_input(df)
    return df


def build_row_pack(assets: dict[str, Any] | None, model: str, row_index: int) -> dict[str, Any]:
    t_total = time.time()
    df = _get_dataset_cached(assets, model)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index {row_index} outside 0..{len(df)-1}")
    row = df.iloc[row_index]
    pack = _get_pipeline(assets).retrieve_row(row)
    pack = final_sanitize_pack(pack)
    diag = pack.setdefault("bridge_diagnostics", {})
    stage = dict(diag.get("stage_times_sec", {}) or {})
    stage["total_build_row_pack"] = round(time.time() - t_total, 3)
    diag["stage_times_sec"] = stage
    return pack


def save_row_outputs(model: str, row_index: int, row_pack: dict[str, Any]) -> dict[str, str]:
    run_name = model.replace("/", "__")

    trace_path = TRACE_DIR / run_name / f"{row_index:04d}.json"
    if RETRIEVAL_SAVE_TRACES:
        _json_save(trace_path, row_pack)

    compact = compact_retrieval_pack(row_pack)
    pack_path = PACK_DIR / run_name / f"{row_index:04d}.json"
    _json_save(pack_path, compact)

    bridges = row_pack.get("bridge_candidates", []) or []
    bridge_rows: list[dict[str, Any]] = []
    for rank, b in enumerate(bridges, 1):
        left_text, right_text, left_ipa, right_ipa = _bridge_side_fields(b)
        bridge_rows.append({
            "row_id": row_index,
            "rank": rank,
            "bridge_type": clean(b.get("bridge_type", b.get("relation", ""))),
            "left_text": left_text,
            "right_text": right_text,
            "left_ipa": left_ipa,
            "right_ipa": right_ipa,
            "bridge_score": float(b.get("bridge_score", b.get("score", 0.0)) or 0.0),
            "phonetic_score": float(b.get("phonetic_score", 0.0) or 0.0),
            "semantic_A_score": float(b.get("semantic_A_score", b.get("source_semantic_score", 0.0)) or 0.0),
            "semantic_B_score": float(b.get("semantic_B_score", b.get("opposite_semantic_score", 0.0)) or 0.0),
            })

    bridges_path = PACK_DIR / run_name / f"{row_index:04d}.bridges.tsv"
    _tsv_save(bridges_path, bridge_rows)

    return {
        "trace_path": str(trace_path),
        "pack_path": str(pack_path),
        "bridges_path": str(bridges_path),
    }


def save_eval_outputs(model: str, start: int, end: int, row_metrics: list[dict[str, Any]], bridge_metrics: list[dict[str, Any]]) -> dict[str, str]:
    run_name = model.replace("/", "__")
    eval_dir = EVAL_DIR / run_name
    rows_path = eval_dir / f"rows_{start}_{end}.tsv"
    bridges_path = eval_dir / f"bridges_{start}_{end}.tsv"
    summary_path = eval_dir / f"summary_{start}_{end}.json"

    _tsv_save(rows_path, row_metrics)
    _tsv_save(bridges_path, bridge_metrics)

    bridge_counts = [int(r.get("bridge_count", 0) or 0) for r in row_metrics]
    best_scores = [float(r.get("best_bridge_score", 0.0) or 0.0) for r in row_metrics]
    elapsed = [float(r.get("elapsed_sec", 0.0) or 0.0) for r in row_metrics]
    summary = {
        "model": model,
        "start": start,
        "end": end,
        "row_count": len(row_metrics),
        "bridge_count": len(bridge_metrics),
        "mean_bridge_count": float(sum(bridge_counts) / max(1, len(bridge_counts))),
        "mean_best_bridge_score": float(sum(best_scores) / max(1, len(best_scores))),
        "total_elapsed_sec": float(sum(elapsed)),
        "mean_elapsed_sec": float(sum(elapsed) / max(1, len(elapsed))),
    }
    _json_save(summary_path, summary)

    return {
        "rows_path": str(rows_path),
        "bridges_path": str(bridges_path),
        "summary_path": str(summary_path),
    }


def evaluate_saved_outputs(model: str, start: int, end: int) -> dict[str, Any]:
    run_name = model.replace("/", "__")
    rows: list[dict[str, Any]] = []
    bridges: list[dict[str, Any]] = []

    for row_index in range(start, end):
        pack_path = PACK_DIR / run_name / f"{row_index:04d}.json"
        bridge_path = PACK_DIR / run_name / f"{row_index:04d}.bridges.tsv"
        if not pack_path.exists() and not bridge_path.exists():
            continue

        pack: dict[str, Any] = {}
        if pack_path.exists():
            try:
                with open(pack_path, "r", encoding="utf-8") as f:
                    pack = json.load(f)
            except Exception:
                pack = {}

        bridge_len = 0
        if bridge_path.exists():
            try:
                bdf = pd.read_csv(bridge_path, sep="	")
                bridge_records = bdf.to_dict("records")
                bridge_len = len(bridge_records)
                bridges.extend(bridge_records)
            except Exception:
                bridge_len = 0

        diag = pack.get("bridge_diagnostics", {}) if isinstance(pack, dict) else {}
        rows.append({
            "row_id": row_index,
            "bridge_count": int(diag.get("bridge_count", bridge_len) or 0),
            "strong_bridge_count": int(diag.get("strong_bridge_count", 0) or 0),
            "identity_bridge_count": int(diag.get("identity_bridge_count", 0) or 0),
            "trivial_inflection_bridge_count": int(diag.get("trivial_inflection_bridge_count", 0) or 0),
            "different_surface_bridge_count": int(diag.get("different_surface_bridge_count", bridge_len) or 0),
            "best_bridge_score": float(diag.get("best_bridge_score", 0.0) or 0.0),
            "fallback_level": clean(pack.get("fallback_level", "")) if isinstance(pack, dict) else "",
        })

    paths = save_eval_outputs(model, start, end, rows, bridges)
    return {"ok": True, "model": model, "start": start, "end": end, "row_count": len(rows), "bridge_count": len(bridges), **paths}


# Server hot-reload entrypoints.  retrieval_server.py calls these with the
# persistent ASSETS dict.  These functions never make HTTP calls.


def debug_row(assets: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    model = clean(payload.get("model", TRANSLATE_MODEL))
    row_index = int(payload.get("row_index", 0))
    pack = build_row_pack(assets, model, row_index)
    return {"ok": True, "model": model, "row_index": row_index, "pack": pack}




# Thin CLI.  When this file is executed directly it only talks to the permanent
# server.  It does not run local retrieval unless you call old local helpers by
# importing them yourself.
def _http_post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    import requests
    url = f"{os.environ.get('RETRIEVAL_SERVER_URL', 'http://127.0.0.1:8765')}{path}"
    r = requests.post(url, json=payload or {}, timeout=int(os.environ.get("RETRIEVAL_TIMEOUT", "3600")))
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {url}\n{r.text}")
    return r.json()


def _http_get(path: str) -> dict[str, Any]:
    import requests
    url = f"{os.environ.get('RETRIEVAL_SERVER_URL', 'http://127.0.0.1:8765')}{path}"
    r = requests.get(url, timeout=int(os.environ.get("RETRIEVAL_TIMEOUT", "3600")))
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {url}\n{r.text}")
    return r.json()



def _row_range_output_path(model: str, row_start: int, row_end: int) -> Path:
    """Return the normal pipeline TSV path for a row slice.

    Retrieval output must match the other stages: data/processed/retrieval/{model}/0.tsv,
    1.tsv, ... .  Even partial dev runs inside a chunk write that chunk filename.
    Example: rows 0:10 and rows 0:100 both write 0.tsv; rows 100:110 writes 1.tsv.
    Cross-chunk row ranges are intentionally rejected in build_and_save_retrieval_rows();
    use retrieve_chunks for multiple chunks.
    """
    out_dir = _retrieval_output_dir(model)
    if row_start < 0 or row_end <= row_start:
        raise ValueError(f"Invalid row range start={row_start} end={row_end}")
    return out_dir / f"{row_start // CHUNK_SIZE}.tsv"


def _request_cancel_retrieval(assets: dict[str, Any] | None) -> dict[str, Any]:
    """Ask the running retrieval loop to stop at the next row boundary.

    This is cooperative cancellation: it keeps loaded models/indexes in memory and
    does not restart the server.  It cannot interrupt a single row while that row
    is inside FAISS/model code, but every retrieval loop must check this flag
    before starting the next row.

    Important control-plane behavior: clear the active-job lock immediately.  The
    running loop will still see the cancel flag before/after the current row and
    exit cleanly, but the user is no longer blocked by a stale lock if the client
    died or the cancel was requested after the lock became inconsistent.
    """
    if assets is None:
        return {"ok": True, "cancel_requested": True, "active_job": None, "cleared_active_job": None}
    active = assets.pop("_retrieval_active", None)
    assets["_retrieval_cancel_requested"] = True
    # Backward compatibility with older server fallback code.
    assets["cancel_requested"] = True
    return {
        "ok": True,
        "cancel_requested": True,
        "active_job": active,
        "cleared_active_job": active,
    }


def cancel(assets: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Server /cancel entrypoint.  Does not reload models."""
    return _request_cancel_retrieval(assets)


def _retrieval_cancel_requested(assets: dict[str, Any] | None) -> bool:
    return bool(assets is not None and assets.get("_retrieval_cancel_requested"))


def _clear_retrieval_cancel(assets: dict[str, Any] | None) -> None:
    if assets is not None:
        assets.pop("_retrieval_cancel_requested", None)





def _prewarm_row_range_retrieval_caches(assets: dict[str, Any] | None, df: pd.DataFrame, row_start: int, row_end: int) -> None:
    """Batch warm caches for a row slice without rebuilding models.

    This must mirror RetrievalPipeline.retrieve_row(): semantic launch is per
    French list term, not joined A/B strings.  The same semantic_side_requests()
    helper is used here and in retrieve_row(), so the cache keys match exactly.
    """
    if assets is None:
        return
    pipe = _get_pipeline(assets)
    rows = df.iloc[row_start:row_end]

    semantic_requests: list[tuple[str, int, str]] = []
    seen_requests: set[tuple[str, int, str]] = set()
    per_row_terms: list[tuple[list[str], list[str]]] = []

    for _, row in rows.iterrows():
        a_terms, b_terms = side_terms(row)
        per_row_terms.append((a_terms, b_terms))
        for req in semantic_side_requests(a_terms, "A", SEMANTIC_K) + semantic_side_requests(b_terms, "B", SEMANTIC_K):
            if req in seen_requests:
                continue
            seen_requests.add(req)
            semantic_requests.append(req)

    semantic_many: dict[str, list[dict[str, Any]]] = {}
    if semantic_requests:
        t = time.time()
        print(
            f"[retrieve rows={row_start}:{row_end}] "
            f"prewarm semantic term searches={len(semantic_requests)}",
            flush=True,
        )
        semantic_many = pipe.expression.semantic_search_many(semantic_requests)
        print(
            f"[retrieve rows={row_start}:{row_end}] "
            f"prewarm semantic done elapsed={time.time()-t:.2f}s",
            flush=True,
        )

    # Warm FastText only if it is already loaded/enabled.  Use the same per-term
    # semantic cache entries as retrieve_row(), not obsolete joined A/B queries.
    ft = getattr(pipe, "fasttext", None)
    if ft is not None and getattr(ft, "enabled", False):
        seed_terms: list[str] = []
        for a_terms, b_terms in per_row_terms:
            seed_terms.extend(a_terms)
            seed_terms.extend(b_terms)

            a_reqs = semantic_side_requests(a_terms, "A", SEMANTIC_K)
            b_reqs = semantic_side_requests(b_terms, "B", SEMANTIC_K)
            # semantic_many already contains uncached results from the prewarm call;
            # if a request was already cached before this run, semantic_search_many
            # also returned it above.  Fall back to cached lookup only defensively.
            if not all(channel in semantic_many for _, _, channel in a_reqs + b_reqs):
                semantic_many.update(pipe.expression.semantic_search_many(a_reqs + b_reqs))

            sem_a = merge_semantic_side_results(semantic_many, [channel for _, _, channel in a_reqs])
            sem_b = merge_semantic_side_results(semantic_many, [channel for _, _, channel in b_reqs])
            seed_terms.extend([clean(x.get("surface", "")) for x in sem_a[:8]])
            seed_terms.extend([clean(x.get("surface", "")) for x in sem_b[:8]])

        seed_terms = unique_keep_order([x for x in seed_terms if clean(x)], limit=512)
        if seed_terms:
            t = time.time()
            print(f"[retrieve rows={row_start}:{row_end}] prewarm fasttext seeds={len(seed_terms)}", flush=True)
            for i in range(0, len(seed_terms), max(1, FASTTEXT_SEED_LIMIT)):
                ft.expand(seed_terms[i:i + FASTTEXT_SEED_LIMIT], side="prewarm", level=1, limit=FASTTEXT_MAX_CANDIDATES_PER_SIDE)
            print(f"[retrieve rows={row_start}:{row_end}] prewarm fasttext done elapsed={time.time()-t:.2f}s", flush=True)

def build_and_save_retrieval_rows(assets: dict[str, Any] | None, model: str, row_start: int, row_end: int) -> dict[str, Any]:
    """Process an explicit row range and save normal chunk TSV outputs.

    start/end are row indices.  A range may cross 100-row chunk boundaries;
    this function splits it into chunk-local slices and writes the normal
    retrieval/{model}/{chunk_index}.tsv files for each touched chunk.  For
    example, rows 0:500 write 0.tsv, 1.tsv, 2.tsv, 3.tsv, and 4.tsv.
    """
    df = _get_dataset_cached(assets, model)
    validate_input(df)
    row_start = max(0, int(row_start))
    row_end = len(df) if int(row_end) == -1 else min(len(df), int(row_end))
    if row_end <= row_start:
        raise ValueError(f"Invalid row range start={row_start} end={row_end}")

    def run_one_chunk_slice(slice_start: int, slice_end: int) -> dict[str, Any]:
        if (slice_end - 1) // CHUNK_SIZE != slice_start // CHUNK_SIZE:
            raise ValueError(f"Internal error: row slice {slice_start}:{slice_end} crosses chunk boundary")

        chunk = df.iloc[slice_start:slice_end].copy()
        out_path = _row_range_output_path(model, slice_start, slice_end)
        _ensure_dir(out_path.parent)

        _prewarm_row_range_retrieval_caches(assets, df, slice_start, slice_end)

        retrieval_cols: list[dict[str, Any]] = []
        bridge_metrics: list[dict[str, Any]] = []
        t_slice = time.time()
        cancelled = False
        for local_i, (row_index, row) in enumerate(chunk.iterrows(), 1):
            if _retrieval_cancel_requested(assets):
                cancelled = True
                print(f"[retrieve rows={slice_start}:{slice_end}] cancel requested before row={row_index}; saving partial output", flush=True)
                break
            t0 = time.time()
            print(f"[retrieve rows={slice_start}:{slice_end} {local_i}/{len(chunk)}] row={row_index}", flush=True)
            pack = build_row_pack(assets, model, int(row_index))
            diag = pack.get("bridge_diagnostics", {}) or {}
            stage = diag.get("stage_times_sec", {}) or {}
            print(
                f"[retrieve rows={slice_start}:{slice_end} {local_i}/{len(chunk)}] done row={row_index} "
                f"bridges={int(diag.get('bridge_count', len(pack.get('bridge_candidates', []) or [])) or 0)} "
                f"elapsed={time.time() - t0:.2f}s bridge_mining={float(stage.get('bridge_mining', 0.0) or 0.0):.3f}s",
                flush=True,
            )
            retrieval_cols.append(_retrieval_columns_from_pack(pack))
            for rank, b in enumerate(pack.get("bridge_candidates", []) or [], 1):
                bridge_metrics.append(_compact_bridge_metric(int(row_index), rank, b))
            if _retrieval_cancel_requested(assets):
                cancelled = True
                print(f"[retrieve rows={slice_start}:{slice_end}] cancel requested after row={row_index}; saving partial output", flush=True)
                break

        processed_count = len(retrieval_cols)
        if retrieval_cols:
            chunk_out = pd.concat([chunk.iloc[:processed_count].reset_index(drop=True), pd.DataFrame(retrieval_cols)], axis=1)
        else:
            chunk_out = chunk.iloc[:0].copy()
        save(chunk_out, str(out_path))
        return {
            "ok": True,
            "cancelled": bool(cancelled),
            "model": model,
            "row_start": int(slice_start),
            "row_end": int(slice_end),
            "row_count": int(len(chunk_out)),
            "bridge_count": int(len(bridge_metrics)),
            "rows_path": str(out_path),
            "elapsed_sec": round(time.time() - t_slice, 3),
        }

    t_all = time.time()
    results: list[dict[str, Any]] = []
    current = row_start
    while current < row_end:
        slice_end = min(row_end, ((current // CHUNK_SIZE) + 1) * CHUNK_SIZE)
        result = run_one_chunk_slice(current, slice_end)
        results.append(result)
        current = slice_end
        if result.get("cancelled") or _retrieval_cancel_requested(assets):
            break

    if len(results) == 1:
        single = dict(results[0])
        single["chunk_results"] = results
        single["chunk_paths"] = [single.get("rows_path", "")]
        single["elapsed_sec"] = round(time.time() - t_all, 3)
        return single

    return {
        "ok": True,
        "cancelled": any(bool(r.get("cancelled")) for r in results),
        "model": model,
        "row_start": int(row_start),
        "row_end": int(row_end),
        "row_count": int(sum(int(r.get("row_count", 0) or 0) for r in results)),
        "bridge_count": int(sum(int(r.get("bridge_count", 0) or 0) for r in results)),
        "rows_path": str(_retrieval_output_dir(model)),
        "chunk_paths": [str(r.get("rows_path", "")) for r in results],
        "chunk_results": results,
        "elapsed_sec": round(time.time() - t_all, 3),
    }

def retrieve_rows(assets: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if bool(payload.get("cancel", False)) or clean(payload.get("task", "")).lower() == "cancel":
        return _request_cancel_retrieval(assets)
    model = clean(payload.get("model", TRANSLATE_MODEL))
    row_start = int(payload.get("row_start", payload.get("start", 0)))
    row_end = int(payload.get("row_end", payload.get("end", row_start + 10)))
    force = bool(payload.get("force", False))
    if assets is not None:
        active = assets.get("_retrieval_active")
        if active and not force:
            raise RuntimeError(f"Retrieval job already running. Active job: {active}")
        _clear_retrieval_cancel(assets)
        assets["_retrieval_active"] = {"model": model, "row_start": row_start, "row_end": row_end, "started_at": time.time()}
    try:
        return build_and_save_retrieval_rows(assets, model, row_start, row_end)
    finally:
        if assets is not None:
            assets.pop("_retrieval_active", None)
            _clear_retrieval_cancel(assets)
            _clear_retrieval_cancel(assets)




# ─────────────────────────────────────────────────────────────────────────────
# Chunk-table retrieval output API
# ─────────────────────────────────────────────────────────────────────────────
# This overrides the older per-row "packs" writer above.  The permanent server
# still calls retrieval.retrieve(ASSETS, payload); only the hot-reloaded logic in
# this file changes.  Retrieval now behaves like the rest of the pipeline:
#   input:  translate/{model}/{chunk}.tsv-style rows
#   output: data/processed/retrieval/{model}/{chunk}.tsv
# with retrieval columns appended to the original table.


def _retrieval_run_name(model: str) -> str:
    return model.replace("/", "__")


def _retrieval_output_dir(model: str) -> Path:
    return PROCESSED_RETRIEVAL_DIR / _retrieval_run_name(model)


def _chunk_count(n_rows: int) -> int:
    return int(math.ceil(n_rows / max(1, CHUNK_SIZE)))


def _chunk_bounds_for_index(n_rows: int, chunk_index: int) -> tuple[int, int]:
    if chunk_index < 0:
        raise IndexError(f"chunk_index must be >= 0, got {chunk_index}")
    start = chunk_index * CHUNK_SIZE
    end = min(n_rows, start + CHUNK_SIZE)
    if start >= n_rows:
        raise IndexError(f"chunk_index {chunk_index} outside 0..{max(0, _chunk_count(n_rows)-1)}")
    return start, end


def _safe_json_for_tsv(obj: Any) -> str:
    return json.dumps(obj if obj is not None else [], ensure_ascii=False, separators=(",", ":"))


def _compact_bridge_metric(row_index: int, rank: int, b: dict[str, Any]) -> dict[str, Any]:
    left_text, right_text, left_ipa, right_ipa = _bridge_side_fields(b)
    return {
        "row_id": row_index,
        "rank": rank,
        "bridge_type": clean(b.get("bridge_type", b.get("relation", ""))),
        "affordance_stage": clean(b.get("affordance_stage", "")),
        "phonetic_relation": clean(b.get("phonetic_relation", "")),
        "semantic_relation": clean(b.get("semantic_relation", "")),
        "semantic_verified": bool(b.get("semantic_verified", False)),
        "left_text": left_text,
        "right_text": right_text,
        "left_ipa": left_ipa,
        "right_ipa": right_ipa,
        "bridge_score": float(b.get("bridge_score", b.get("score", 0.0)) or 0.0),
        "llm_priority_score": float(b.get("llm_priority_score", b.get("bridge_score", 0.0)) or 0.0),
        "phonetic_score": float(b.get("phonetic_score", 0.0) or 0.0),
        "naturalness_score": float(b.get("naturalness_score", 0.0) or 0.0),
        "surprise_score": float(b.get("surprise_score", 0.0) or 0.0),
        "pivotability_score": float(b.get("pivotability_score", 0.0) or 0.0),
        "opposite_semantic_score": float(b.get("opposite_semantic_score", 0.0) or 0.0),
    }




def build_and_save_retrieval_chunk(assets: dict[str, Any] | None, model: str, chunk_index: int) -> dict[str, Any]:
    df = _get_dataset_cached(assets, model)
    validate_input(df)
    start_row, end_row = _chunk_bounds_for_index(len(df), chunk_index)
    chunk = df.iloc[start_row:end_row].copy()
    out_dir = _retrieval_output_dir(model)
    _ensure_dir(out_dir)

    row_metrics: list[dict[str, Any]] = []
    bridge_metrics: list[dict[str, Any]] = []
    retrieval_cols: list[dict[str, Any]] = []
    t_chunk = time.time()

    for local_i, (row_index, row) in enumerate(chunk.iterrows(), 1):
        t0 = time.time()
        print(f"[retrieve chunk={chunk_index} {local_i}/{len(chunk)}] row={row_index}", flush=True)
        pack = build_row_pack(assets, model, int(row_index))
        row_elapsed = time.time() - t0
        if RETRIEVAL_STAGE_TIMINGS:
            diag_for_log = pack.get("bridge_diagnostics", {}) or {}
            stage_for_log = diag_for_log.get("stage_times_sec", {}) or {}
            bridge_count_for_log = int(diag_for_log.get("bridge_count", len(pack.get("bridge_candidates", []) or [])) or 0)
            print(
                f"[retrieve chunk={chunk_index} {local_i}/{len(chunk)}] done row={row_index} "
                f"bridges={bridge_count_for_log} elapsed={row_elapsed:.2f}s "
                f"bridge_mining={float(stage_for_log.get('bridge_mining', 0.0) or 0.0):.3f}s",
                flush=True,
            )
        if RETRIEVAL_SAVE_TRACES:
            trace_path = TRACE_DIR / _retrieval_run_name(model) / f"{int(row_index):04d}.json"
            _json_save(trace_path, pack)

        cols = _retrieval_columns_from_pack(pack)
        retrieval_cols.append(cols)
        diag = pack.get("bridge_diagnostics", {}) or {}
        row_metric = {
            "row_id": int(row_index),
            "chunk_index": int(chunk_index),
            "bridge_count": cols["retrieval_bridge_count"],
            "affordance_count": cols["retrieval_affordance_count"],
            "best_bridge_score": cols["retrieval_best_bridge_score"],
            "fallback_level": cols["retrieval_fallback_level"],
            "elapsed_sec": round(time.time() - t0, 3),
        }
        for k, v in (diag.get("stage_times_sec", {}) or {}).items():
            row_metric[f"stage_{k}_sec"] = v
        row_metrics.append(row_metric)

        for rank, b in enumerate(pack.get("bridge_candidates", []) or [], 1):
            bridge_metrics.append(_compact_bridge_metric(int(row_index), rank, b))

    if retrieval_cols:
        chunk = pd.concat([chunk.reset_index(drop=True), pd.DataFrame(retrieval_cols)], axis=1)
    out_path = out_dir / f"{chunk_index}.tsv"
    save(chunk, str(out_path))

    return {
        "chunk_index": int(chunk_index),
        "row_start": int(start_row),
        "row_end": int(end_row),
        "row_count": int(len(chunk)),
        "bridge_count": int(len(bridge_metrics)),
        "rows_path": str(out_path),
        "elapsed_sec": round(time.time() - t_chunk, 3),
        "row_metrics": row_metrics,
        "bridge_metrics": bridge_metrics,
    }


def save_retrieval_summary(model: str, chunk_start: int, chunk_end: int, chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = _retrieval_output_dir(model)
    _ensure_dir(out_dir)
    row_count = sum(int(x.get("row_count", 0) or 0) for x in chunk_results)
    bridge_count = sum(int(x.get("bridge_count", 0) or 0) for x in chunk_results)
    elapsed = sum(float(x.get("elapsed_sec", 0.0) or 0.0) for x in chunk_results)
    paths = [x.get("rows_path", "") for x in chunk_results]
    summary = {
        "ok": True,
        "model": model,
        "chunk_start": int(chunk_start),
        "chunk_end": int(chunk_end),
        "chunk_count": len(chunk_results),
        "row_count": int(row_count),
        "bridge_count": int(bridge_count),
        "output_dir": str(out_dir),
        "chunk_paths": paths,
        "total_elapsed_sec": round(elapsed, 3),
    }
    summary_path = out_dir / f"summary_{chunk_start}_{chunk_end}.json"
    _json_save(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def retrieve(assets: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Server entrypoint.

    Default behavior: start/end are ROW indices.  This is what the CLI command
    `python retrieval_refactored.py retrieve gemini 0 10` means.

    Chunk mode is explicit only: pass mode=chunks or chunk_start/chunk_end.
    This preserves normal pipeline files like retrieval/{model}/0.tsv while
    avoiding accidental 1000-row dev calls.

    Safety: the permanent server keeps ASSETS alive, so we keep a lightweight
    in-ASSETS lock to prevent duplicate long jobs.
    """
    if bool(payload.get("cancel", False)) or clean(payload.get("task", "")).lower() == "cancel":
        return _request_cancel_retrieval(assets)
    model = clean(payload.get("model", TRANSLATE_MODEL))
    mode = clean(payload.get("mode", "rows"))
    if mode != "chunks" and "chunk_start" not in payload and "chunk_end" not in payload:
        return retrieve_rows(assets, payload)

    chunk_start = int(payload.get("chunk_start", payload.get("start", 0)))
    raw_end = payload.get("chunk_end", payload.get("end", chunk_start + 1))
    chunk_end = int(raw_end)
    force = bool(payload.get("force", False))

    if assets is not None:
        active = assets.get("_retrieval_active")
        if active and not force:
            raise RuntimeError(
                "Retrieval job already running on this server. Wait for it to finish, "
                "or pass force=true only if you are sure the old client died. "
                f"Active job: {active}"
            )

    df = _get_dataset_cached(assets, model)
    n_chunks = _chunk_count(len(df))
    if chunk_end == -1:
        chunk_end = n_chunks
    if chunk_end <= chunk_start:
        raise ValueError(f"Invalid chunk range start={chunk_start} end={chunk_end}")
    if chunk_start < 0 or chunk_start >= n_chunks:
        raise IndexError(f"chunk_start {chunk_start} outside 0..{n_chunks-1}")
    chunk_end = min(chunk_end, n_chunks)
    chunk_count = chunk_end - chunk_start

    if MAX_CHUNKS_PER_CALL > 0 and chunk_count > MAX_CHUNKS_PER_CALL and not force:
        raise RuntimeError(
            f"Refusing to process {chunk_count} chunks ({chunk_count * CHUNK_SIZE} nominal rows) in one dev call. "
            f"Run one chunk at a time, e.g. `python retrieval_refactored.py retrieve {model} {chunk_start} {chunk_start + 1}`, "
            "or set RETRIEVAL_MAX_CHUNKS_PER_CALL=0 / pass force=true for production."
        )

    if assets is not None:
        _clear_retrieval_cancel(assets)
        assets["_retrieval_active"] = {
            "model": model,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "started_at": time.time(),
        }

    try:
        results: list[dict[str, Any]] = []
        cancelled = False
        for chunk_index in range(chunk_start, chunk_end):
            if _retrieval_cancel_requested(assets):
                cancelled = True
                print(f"[retrieve chunks={chunk_start}:{chunk_end}] cancel requested before chunk={chunk_index}", flush=True)
                break
            results.append(build_and_save_retrieval_chunk(assets, model, chunk_index))
            if _retrieval_cancel_requested(assets):
                cancelled = True
                print(f"[retrieve chunks={chunk_start}:{chunk_end}] cancel requested after chunk={chunk_index}", flush=True)
                break

        summary = save_retrieval_summary(model, chunk_start, chunk_end, results)
        summary["cancelled"] = bool(cancelled)
        summary["chunk_results"] = results
        return summary
    finally:
        if assets is not None:
            assets.pop("_retrieval_active", None)
            _clear_retrieval_cancel(assets)


def eval_rows(assets: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize saved retrieval chunk TSVs, not per-row pack files."""
    model = clean(payload.get("model", TRANSLATE_MODEL))
    if clean(payload.get("mode", "")) == "rows" or "row_start" in payload or "row_end" in payload:
        return retrieve_rows(assets, payload)

    chunk_start = int(payload.get("chunk_start", payload.get("start", 0)))
    raw_end = payload.get("chunk_end", payload.get("end", chunk_start + 1))
    out_dir = _retrieval_output_dir(model)
    if int(raw_end) == -1:
        files = sorted(out_dir.glob("*.tsv"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    else:
        chunk_end = int(raw_end)
        files = [out_dir / f"{i}.tsv" for i in range(chunk_start, chunk_end) if (out_dir / f"{i}.tsv").exists()]

    row_count = 0
    bridge_count = 0
    affordance_count = 0
    paths: list[str] = []
    for p in files:
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception:
            continue
        paths.append(str(p))
        row_count += len(df)
        if "retrieval_bridge_count" in df.columns:
            bridge_count += int(pd.to_numeric(df["retrieval_bridge_count"], errors="coerce").fillna(0).sum())
        if "retrieval_affordance_count" in df.columns:
            affordance_count += int(pd.to_numeric(df["retrieval_affordance_count"], errors="coerce").fillna(0).sum())
    return {
        "ok": True,
        "model": model,
        "output_dir": str(out_dir),
        "chunk_paths": paths,
        "chunk_count": len(paths),
        "row_count": int(row_count),
        "bridge_count": int(bridge_count),
        "affordance_count": int(affordance_count),
    }





# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# This section intentionally lives at the end so hot-reload uses these functions
# without changing retrieval_server.py or reloading model/index assets.


_BAD_PHRASE_EDGES = {
    "de", "du", "des", "le", "la", "les", "un", "une", "à", "au", "aux",
    "et", "ou", "mais", "que", "qui", "quoi", "ne", "pas", "plus", "très",
}


def _phrase_surface_ok(surface: Any) -> bool:
    s = norm_text(surface)
    if not s:
        return False
    words = [w for w in s.split() if w]
    if not (2 <= len(words) <= 6):
        return False
    if words[0] in _BAD_PHRASE_EDGES or words[-1] in _BAD_PHRASE_EDGES:
        return False
    if any(lexically_bad_candidate_surface(w) for w in words):
        return False
    # Avoid raw corpus lemmatization artifacts that are visibly ungrammatical.
    bad_patterns = (
        r"\bde le\b", r"\bde les\b", r"\bavoir un faim\b", r"\bboire de thé\b",
        r"\bfaire un tisane\b", r"\boccuper un place\b", r"\bchose de approprié\b",
    )
    return not any(re.search(p, s) for p in bad_patterns)


def _phrase_affordance_score(item: dict[str, Any]) -> float:
    s = norm_text(item.get("surface", ""))
    if not _phrase_surface_ok(s):
        return 0.0
    base = float(item.get("score", 0.0) or 0.0)
    source = norm_text(item.get("source", ""))
    freq = 0.0
    try:
        freq = math.log1p(float(item.get("frequency", 0.0) or 0.0)) / 8.0
    except Exception:
        freq = 0.0
    src_bonus = 0.0
    if "opensubtitles" in source:
        src_bonus += 0.18
    if "parseme" in source:
        src_bonus += 0.16
    if "wiktionary" in source:
        src_bonus += 0.10
    wc = len(s.split())
    length_bonus = 0.10 if 2 <= wc <= 4 else 0.04
    return clamp01(0.55 * base + src_bonus + 0.17 * freq + length_bonus)


def _phrase_level_affordances_from_pack(pack: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("A", pack.get("semantic_A_expressions", []) or []),
        ("B", pack.get("semantic_B_expressions", []) or []),
        ("blended", pack.get("semantic_expressions", []) or []),
    ]
    seen: set[str] = set()
    for side, items in sources:
        for item in items:
            if not isinstance(item, dict):
                continue
            surface = norm_text(item.get("surface", ""))
            if not surface or surface in seen or not _phrase_surface_ok(surface):
                continue
            score = _phrase_affordance_score(item)
            if score <= 0.0:
                continue
            seen.add(surface)
            rows.append({
                "affordance_bucket": "phrase_level",
                "surface": surface,
                "semantic_side": side,
                "source": clean(item.get("source", "")),
                "semantic_score": float(item.get("score", 0.0) or 0.0),
                "phrase_affordance_score": score,
                "frequency": clean(item.get("frequency", "")),
                "pmi": clean(item.get("pmi", "")),
            })
    rows.sort(key=lambda r: -float(r.get("phrase_affordance_score", 0.0) or 0.0))
    return rows[:limit]








# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# This override intentionally removes fake phrase placeholders and any verbose
# debug payload from the default TSV output. Phrase-level now means: an actual
# retrieved left/right surface pair where at least one side is multiword.


def _is_real_phrase_pair_idea(b: dict[str, Any]) -> bool:
    left, right = bridge_surface_pair(b)
    return bool(clean(left) and clean(right) and (len(clean(left).split()) > 1 or len(clean(right).split()) > 1))















# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Generator-facing output remains compact. Internal fields may be computed, but
# the TSV default only writes concrete surface-pair ideas and approved scores.

_STRONG_PIVOT_OVERRIDES = {
    "conte": 0.82, "comte": 0.82, "compte": 0.74,
    "verre": 0.78, "verres": 0.72, "vert": 0.78, "vers": 0.70,
    "foi": 0.82, "foie": 0.82, "fois": 0.76,
    "pansé": 0.72, "pensée": 0.74, "pense": 0.58, "penser": 0.46,
    "crise": 0.66, "brise": 0.70,
    "mère": 0.78, "mer": 0.78, "maire": 0.72,
    "poisson": 0.70, "boisson": 0.62, "boissons": 0.60,
    "désert": 0.62, "presse": 0.56, "récite": 0.48,
}
_WEAK_PIVOT_OVERRIDES = {
    "très": 0.02, "trait": 0.16,
    "être": 0.04, "avoir": 0.04, "voire": 0.06, "entre": 0.08,
    "soif": 0.26, "coiffe": 0.22,
    "convenir": 0.20, "contenir": 0.24,
    "auprès": 0.05, "au vrai": 0.05, "en titre": 0.06,
}






def bridge_pivotability_score(b: dict[str, Any]) -> float:
    left, right = bridge_surface_pair(b)
    surfaces = [clean(x) for x in [left, right] if clean(x)]
    if len(surfaces) < 2:
        return 0.0
    vals = [surface_pivotability(x) for x in surfaces]
    if min(vals) <= 0.06:
        return clamp01(0.15 * max(vals))
    pair_balance = min(vals)
    pair_strength = max(vals)
    score = 0.55 * pair_balance + 0.45 * pair_strength
    phon = clamp01(b.get("phonetic_score", 0.0))
    if phon >= 0.96:
        score += 0.08
    elif phon >= 0.82:
        score += 0.03
    # Penalize source-side support/function accidents even if the candidate word
    # itself is recognizable. This directly suppresses très→trait, être→entre,
    # avoir→voire, etc.
    source = norm_text(b.get("source_surface", b.get("sound_source", "")))
    if source in _LOW_PIVOT_FUNCTION_SURFACES or strip_accents(source) in {strip_accents(x) for x in _LOW_PIVOT_FUNCTION_SURFACES}:
        score -= 0.28
    return clamp01(score)




def _is_low_leap_bridge(b: dict[str, Any]) -> bool:
    marker = " ".join([
        clean(b.get("bridge_type", "")),
        clean(b.get("relation", "")),
        clean(b.get("semantic_relation", "")),
        clean(b.get("affordance_stage", "")),
    ]).lower()
    return "low_leap" in marker

def score_profile_for_generator(b: dict[str, Any]) -> dict[str, float]:
    surfaces = [x for x in bridge_surface_pair(b) if clean(x)]
    phonetic_match = clamp01(b.get("phonetic_score", 0.0))
    french_naturalness = max(
        [clamp01(b.get("naturalness_score", b.get("quality_score", 0.0)))]
        + [surface_naturalness_score(x) for x in surfaces]
    ) if surfaces else clamp01(b.get("naturalness_score", b.get("quality_score", 0.0)))
    semantic_surprise = clamp01(b.get("surprise_score", 0.0))
    semantic_domain_similarity = max(
        clamp01(b.get("source_semantic_score", 0.0)),
        clamp01(b.get("opposite_semantic_score", 0.0)),
        clamp01(b.get("semantic_A_score", 0.0)),
        clamp01(b.get("semantic_B_score", 0.0)),
    )
    pun_pivot_usability = bridge_pivotability_score(b)
    overall_score = clamp01(
        0.28 * phonetic_match
        + 0.18 * french_naturalness
        + 0.18 * semantic_surprise
        + 0.26 * pun_pivot_usability
        + 0.10 * semantic_domain_similarity
    )
    return {
        "phonetic_match": round(float(phonetic_match), 4),
        "french_naturalness": round(float(french_naturalness), 4),
        "semantic_surprise": round(float(semantic_surprise), 4),
        "semantic_domain_similarity": round(float(semantic_domain_similarity), 4),
        "pun_pivot_usability": round(float(pun_pivot_usability), 4),
        "overall_score": round(float(overall_score), 4),
    }


def compact_generator_idea(b: dict[str, Any]) -> dict[str, Any]:
    left, right = bridge_surface_pair(b)
    left = clean(left)
    right = clean(right)
    if not left or not right:
        return {}
    relation_raw = clean(b.get("phonetic_relation") or b.get("relation") or b.get("bridge_type") or "")
    relation = "same_sound" if ("homophone" in relation_raw or clamp01(b.get("phonetic_score", 0.0)) >= 0.96) else "similar_sound"
    return {
        "left": left,
        "right": right,
        "relation": relation,
        "retrieval_bucket": clean(b.get("retrieval_bucket", "")),
        "retrieval_bucket_rank": int(b.get("retrieval_bucket_rank", 0) or 0),
        "scores": score_profile_for_generator(b),
    }




def _dedupe_generator_ideas(ideas: list[dict[str, Any]], limit: int = MAX_GENERATOR_AFFORDANCES) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    def key_score(x: dict[str, Any]) -> float:
        s = x.get("scores", {}) or {}
        return float(s.get("overall_score", 0.0) or 0.0)
    for idea in sorted([x for x in ideas if x], key=key_score, reverse=True):
        k = (strip_accents(idea.get("left", "")), strip_accents(idea.get("right", "")), clean(idea.get("relation", "")))
        rk = (k[1], k[0], k[2])
        if k in seen or rk in seen:
            continue
        seen.add(k)
        out.append(idea)
        if len(out) >= limit:
            break
    return out




def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python retrieval_refactored.py status\n"
            "  python retrieval_refactored.py cancel\n"
            "  python retrieval_refactored.py debug_row gemini 0\n"
            "  python retrieval_refactored.py retrieve gemini 0 100\n"
            "  python retrieval_refactored.py retrieve_chunks gemini 0 1\n"
            "  python retrieval_refactored.py eval gemini 0 100"
        )
    task = sys.argv[1]
    if task == "status":
        print(json.dumps(_http_get("/status"), ensure_ascii=False, indent=2))
        return
    if task == "debug_row":
        model = sys.argv[2] if len(sys.argv) > 2 else TRANSLATE_MODEL
        row_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        print(json.dumps(_http_post("/debug_row", {"model": model, "row_index": row_index}), ensure_ascii=False, indent=2))
        return
    if task == "cancel":
        print(json.dumps(_http_post("/cancel", {}), ensure_ascii=False, indent=2))
        return
    if task in {"retrieve", "retrieve_rows"}:
        model = sys.argv[2] if len(sys.argv) > 2 else TRANSLATE_MODEL
        row_start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        row_end = int(sys.argv[4]) if len(sys.argv) > 4 else row_start + 10
        server = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8765")
        print(f"retrieve model={model} rows={row_start}:{row_end} server={server}", flush=True)
        t0 = time.time()
        summary = _http_post("/retrieve", {"model": model, "mode": "rows", "row_start": row_start, "row_end": row_end})
        summary["client_total_elapsed_sec"] = round(time.time() - t0, 3)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if task == "retrieve_chunks":
        model = sys.argv[2] if len(sys.argv) > 2 else TRANSLATE_MODEL
        chunk_start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        chunk_end = int(sys.argv[4]) if len(sys.argv) > 4 else chunk_start + 1
        server = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8765")
        print(f"retrieve_chunks model={model} chunks={chunk_start}:{chunk_end} server={server}", flush=True)
        t0 = time.time()
        summary = _http_post("/retrieve", {"model": model, "mode": "chunks", "chunk_start": chunk_start, "chunk_end": chunk_end})
        summary["client_total_elapsed_sec"] = round(time.time() - t0, 3)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if task == "eval":
        model = sys.argv[2] if len(sys.argv) > 2 else TRANSLATE_MODEL
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        end = int(sys.argv[4]) if len(sys.argv) > 4 else start + 1
        print(json.dumps(_http_post("/eval", {"model": model, "start": start, "end": end}), ensure_ascii=False, indent=2))
        return
    raise SystemExit(f"Unknown task: {task}")


# ─────────────────────────────────────────────────────────────────────────────
# Keep the approved generator-facing schema; tighten what is allowed through.
# ─────────────────────────────────────────────────────────────────────────────
_FINITE_FORM_JUNK_SURFACES = {
    "sacquent", "saquent", "boitons", "cessons", "soufrait", "coursé",
    "totalisé", "commenté", "convulsionnons", "frusques", "axions", "fab",
    "liseur", "sacque", "saque",
}
_FINITE_FORM_ALLOWED_SURFACES = {
    "boisson", "boissons", "poisson", "poissons", "maison", "maisons",
    "raison", "raisons", "saison", "saisons", "garçon", "garçons",
}
_BAD_FINITE_SUFFIXES_STRICT = (
    "quent", "guent", "chent", "ssent", "aient", "èrent", "erent",
    "irent", "èrent", "assent", "issions", "assions", "erons", "irons",
    "urons", "tons", "tions",
)


def looks_like_low_value_finite_form(surface: Any) -> bool:
    s = norm_text(surface)
    plain = strip_accents(s)
    if not s or " " in s:
        return False
    if s in _FINITE_FORM_ALLOWED_SURFACES or plain in {strip_accents(x) for x in _FINITE_FORM_ALLOWED_SURFACES}:
        return False
    if s in _STRONG_PIVOT_OVERRIDES or plain in _STRONG_PIVOT_OVERRIDES:
        return False
    if s in _FINITE_FORM_JUNK_SURFACES or plain in {strip_accents(x) for x in _FINITE_FORM_JUNK_SURFACES}:
        return True
    if len(plain) >= 6 and plain.endswith(_BAD_FINITE_SUFFIXES_STRICT):
        return True
    # Plural-looking -s is common, but -ons is often a verb artifact in this
    # candidate stream. Keep known common nouns via whitelist above.
    if len(plain) >= 7 and plain.endswith("ons"):
        return True
    return False


# Strengthen/adjust learned manual priors without changing output schema.
_STRONG_PIVOT_OVERRIDES.update({
    "conte": 0.90, "comte": 0.88, "compte": 0.82,
    "verre": 0.86, "verres": 0.82, "vert": 0.86, "vers": 0.78,
    "foi": 0.90, "foie": 0.90, "fois": 0.84,
    "pansé": 0.82, "penser": 0.58,
    "crise": 0.76, "brise": 0.78,
    "poisson": 0.80, "boisson": 0.72, "boissons": 0.70,
})
_WEAK_PIVOT_OVERRIDES.update({
    "très": 0.01, "trait": 0.08,
    "être": 0.02, "avoir": 0.02, "voire": 0.04, "entre": 0.05,
    "soif": 0.18, "coiffe": 0.12,
    "convenir": 0.12, "contenir": 0.16,
    "presse": 0.38, "récite": 0.34, "pensée": 0.34,
    "liseur": 0.02,
})


def surface_pivotability(surface: Any) -> float:
    """Strict estimate of whether a surface can carry a French pun pivot."""
    s = norm_text(surface)
    plain = strip_accents(s)
    if not s or lexically_bad_candidate_surface(s) or looks_like_low_value_finite_form(s):
        return 0.0
    if s in _BAD_LOW_VALUE_PUN_PIVOTS or plain in {strip_accents(x) for x in _BAD_LOW_VALUE_PUN_PIVOTS}:
        return 0.0
    if s in _WEAK_PIVOT_OVERRIDES:
        return clamp01(_WEAK_PIVOT_OVERRIDES[s])
    if plain in _WEAK_PIVOT_OVERRIDES:
        return clamp01(_WEAK_PIVOT_OVERRIDES[plain])
    if s in _STRONG_PIVOT_OVERRIDES:
        return clamp01(_STRONG_PIVOT_OVERRIDES[s])
    if plain in _STRONG_PIVOT_OVERRIDES:
        return clamp01(_STRONG_PIVOT_OVERRIDES[plain])
    if s in _LOW_PIVOT_FUNCTION_SURFACES or plain in {strip_accents(x) for x in _LOW_PIVOT_FUNCTION_SURFACES}:
        return 0.02

    words = [w for w in s.split() if w]
    wc = len(words)
    score = 0.0
    if 2 <= wc <= 4:
        score += 0.55 if all(w not in _LOW_PIVOT_FUNCTION_SURFACES for w in words) else 0.16
    elif re.fullmatch(r"[a-zàâçéèêëîïôûùüÿñæœ'-]{4,10}", s, flags=re.I):
        score += 0.42
    elif re.fullmatch(r"[a-zàâçéèêëîïôûùüÿñæœ'-]{3}", s, flags=re.I):
        score += 0.12

    if plain.endswith(_HIGH_PIVOT_NOUNISH_SUFFIXES):
        score += 0.12
    if re.search(r"(er|ir|re)$", plain) and wc == 1:
        score -= 0.14
    if len(plain) <= 3 and s not in _STRONG_PIVOT_OVERRIDES:
        score -= 0.16
    return clamp01(score)


def surface_naturalness_score(surface: Any) -> float:
    """Ordinary/native French recognizability for generator-facing scoring."""
    s = norm_text(surface)
    if not s or looks_like_low_value_finite_form(s):
        return 0.0
    if s in _BAD_LOW_VALUE_PUN_PIVOTS or strip_accents(s) in {strip_accents(x) for x in _BAD_LOW_VALUE_PUN_PIVOTS}:
        return 0.0
    return clamp01(max(surface_recognizability_prior(s), surface_pivotability(s)))





# ─────────────────────────────────────────────────────────────────────────────
#
# This does NOT change the generator-facing schema.  It changes selection only:
# candidates can survive because they are unusually strong on different approved
# score dimensions (sound, natural French, surprise, French semantic domain-meaning similarity,
# pivot usability, overall), while hard anti-junk gates remain in place.
# ─────────────────────────────────────────────────────────────────────────────
MAX_GENERATOR_IDEAS_INTERNAL_LANES = int(os.environ.get("RETRIEVAL_MAX_GENERATOR_IDEAS", str(MAX_GENERATOR_AFFORDANCES)))
MAX_IDEAS_PER_INTERNAL_LANE = int(os.environ.get("RETRIEVAL_MAX_IDEAS_PER_INTERNAL_LANE", "3"))


def _idea_pair_key(idea: dict[str, Any]) -> tuple[str, str, str]:
    left = strip_accents(clean(idea.get("left", "")))
    right = strip_accents(clean(idea.get("right", "")))
    relation = clean(idea.get("relation", ""))
    a, b = sorted([left, right])
    return a, b, relation


def _bridge_is_actual_phrase_pair(b: dict[str, Any]) -> bool:
    left, right = bridge_surface_pair(b)
    return len(clean(left).split()) > 1 or len(clean(right).split()) > 1


def _candidate_hard_reject(b: dict[str, Any]) -> bool:
    idea = compact_generator_idea(b)
    if not idea:
        return True
    left, right = clean(idea.get("left", "")), clean(idea.get("right", ""))
    if not left or not right:
        return True
    if looks_like_low_value_finite_form(left) or looks_like_low_value_finite_form(right):
        return True
    if lexically_bad_candidate_surface(left) or lexically_bad_candidate_surface(right):
        return True
    if structurally_trivial_variant(left, right) or boring_morphophonetic_echo(left, right):
        return True
    # Keep exact identity/polysemy only if the original bridge explicitly marked it.
    if surface_key(left) == surface_key(right) and "identity" not in clean(b.get("bridge_type", b.get("relation", ""))):
        return True
    # Function/support-word accidents are almost never useful pun pivots.
    if min(surface_pivotability(left), surface_pivotability(right)) <= 0.08:
        return True
    return False


def _candidate_internal_lanes(b: dict[str, Any]) -> list[str]:
    """Internal retention lanes. These names are not exported."""
    if _candidate_hard_reject(b):
        return []
    idea = compact_generator_idea(b)
    scores = idea.get("scores", {}) or {}
    phon = float(scores.get("phonetic_match", 0.0) or 0.0)
    natural = float(scores.get("french_naturalness", 0.0) or 0.0)
    surprise = float(scores.get("semantic_surprise", 0.0) or 0.0)
    semantic_domain = float(scores.get("semantic_domain_similarity", 0.0) or 0.0)
    pivot = float(scores.get("pun_pivot_usability", 0.0) or 0.0)
    overall = float(scores.get("overall_score", 0.0) or 0.0)
    phrase = _bridge_is_actual_phrase_pair(b)

    # Non-negotiable base floor: still ordinary French, still sound-related, still usable as a pivot.
    if phon < 0.76 or natural < 0.40 or pivot < 0.36:
        return []

    lanes: list[str] = []

    # Very strong sound collision.  Allows wider semantic range, but only if both words are usable French pivots.
    if phon >= 0.94 and natural >= 0.46 and pivot >= 0.42:
        lanes.append("sound")

    # Strong surprise/discontinuity.  This is how we widen semantic range without letting in random junk.
    if phon >= 0.80 and natural >= 0.48 and pivot >= 0.46 and surprise >= 0.62:
        lanes.append("surprise")

    # French semantic domain-meaning similarity is a bonus lane, not the main objective.
    if phon >= 0.78 and natural >= 0.44 and pivot >= 0.42 and semantic_domain >= 0.50:
        lanes.append("meaning")

    # Real multiword surface pairs get a lane, but still need the same quality gates.
    if phrase and phon >= 0.76 and natural >= 0.44 and pivot >= 0.42:
        lanes.append("phrase")

    # Overall lane catches balanced candidates that are not category winners.
    if overall >= 0.60 and natural >= 0.44 and pivot >= 0.42:
        lanes.append("overall")

    return lanes


def _internal_lane_sort_key(lane: str, b: dict[str, Any]) -> tuple[float, float, float, float]:
    scores = compact_generator_idea(b).get("scores", {}) or {}
    phon = float(scores.get("phonetic_match", 0.0) or 0.0)
    natural = float(scores.get("french_naturalness", 0.0) or 0.0)
    surprise = float(scores.get("semantic_surprise", 0.0) or 0.0)
    semantic_domain = float(scores.get("semantic_domain_similarity", 0.0) or 0.0)
    pivot = float(scores.get("pun_pivot_usability", 0.0) or 0.0)
    overall = float(scores.get("overall_score", 0.0) or 0.0)
    if lane == "sound":
        return (-phon, -pivot, -natural, -overall)
    if lane == "surprise":
        return (-surprise, -phon, -pivot, -overall)
    if lane == "meaning":
        return (-semantic_domain, -phon, -pivot, -overall)
    if lane == "phrase":
        return (0 if _bridge_is_actual_phrase_pair(b) else 1, -overall, -phon, -pivot)
    return (-overall, -phon, -pivot, -surprise)








# ─────────────────────────────────────────────────────────────────────────────
#
# Goal: return more quality candidates without forcing close semantic range.
# Keep the same generator-facing schema.  The new selection first keeps strict
# candidates, then backfills rows with too few ideas using high-quality French
# phonetic reinterpretations that are natural, usable pivots, and semantically
# surprising even when French semantic domain-meaning similarity is weak.
# ─────────────────────────────────────────────────────────────────────────────
TARGET_GENERATOR_IDEAS_STRICT_CREATIVE = int(os.environ.get("RETRIEVAL_TARGET_GENERATOR_IDEAS", "6"))
MAX_GENERATOR_IDEAS_STRICT_CREATIVE = int(os.environ.get("RETRIEVAL_MAX_GENERATOR_IDEAS", str(max(MAX_GENERATOR_IDEAS_INTERNAL_LANES, 10))))
MAX_IDEAS_PER_SURFACE = int(os.environ.get("RETRIEVAL_MAX_IDEAS_PER_SURFACE", "2"))

_STRONG_PIVOT_OVERRIDES.update({
    "brise": 0.82,
    "grise": 0.68,
    "réussi": 0.66,
    "reussi": 0.66,
    "récit": 0.72,
    "recit": 0.72,
    "pansée": 0.54,
    "pensee": 0.54,
    "vers": 0.80,
})
_WEAK_PIVOT_OVERRIDES.update({
    "liseur": 0.08,
    "coiffe": 0.10,
    "trait": 0.06,
    "presse": 0.32,
    "récite": 0.28,
    "recite": 0.28,
    "contenir": 0.12,
    "convenir": 0.10,
})


def _candidate_quality_values(b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    idea = compact_generator_idea(b)
    scores = idea.get("scores", {}) if idea else {}
    return idea, {
        "phon": float(scores.get("phonetic_match", 0.0) or 0.0),
        "natural": float(scores.get("french_naturalness", 0.0) or 0.0),
        "surprise": float(scores.get("semantic_surprise", 0.0) or 0.0),
        "semantic_domain": float(scores.get("semantic_domain_similarity", 0.0) or 0.0),
        "pivot": float(scores.get("pun_pivot_usability", 0.0) or 0.0),
        "overall": float(scores.get("overall_score", 0.0) or 0.0),
    }


def _surface_pair_hard_reject(left: str, right: str, b: dict[str, Any]) -> bool:
    if not left or not right:
        return True
    if looks_like_low_value_finite_form(left) or looks_like_low_value_finite_form(right):
        return True
    if lexically_bad_candidate_surface(left) or lexically_bad_candidate_surface(right):
        return True
    if structurally_trivial_variant(left, right) or boring_morphophonetic_echo(left, right):
        return True
    if surface_key(left) == surface_key(right) and "identity" not in clean(b.get("bridge_type", b.get("relation", ""))):
        return True
    if min(surface_pivotability(left), surface_pivotability(right)) <= 0.08:
        return True
    return False


def _strict_candidate(b: dict[str, Any]) -> bool:
    idea, q = _candidate_quality_values(b)
    if not idea:
        return False
    left, right = clean(idea.get("left", "")), clean(idea.get("right", ""))
    if _surface_pair_hard_reject(left, right, b):
        return False
    if q["phon"] >= 0.96:
        return q["natural"] >= 0.50 and q["pivot"] >= 0.50 and q["overall"] >= 0.60
    return q["phon"] >= 0.80 and q["natural"] >= 0.54 and q["pivot"] >= 0.52 and q["overall"] >= 0.58


def _creative_candidate(b: dict[str, Any]) -> bool:
    """Low-style compensation: semantics may drift, but sound/French/pivot/surprise must be strong."""
    idea, q = _candidate_quality_values(b)
    if not idea:
        return False
    left, right = clean(idea.get("left", "")), clean(idea.get("right", ""))
    if _surface_pair_hard_reject(left, right, b):
        return False
    if q["phon"] >= 0.94 and q["natural"] >= 0.46 and q["pivot"] >= 0.46 and q["surprise"] >= 0.42:
        return True
    if q["phon"] >= 0.82 and q["natural"] >= 0.52 and q["pivot"] >= 0.50 and q["surprise"] >= 0.58:
        return True
    if _bridge_is_actual_phrase_pair(b) and q["phon"] >= 0.76 and q["natural"] >= 0.56 and q["pivot"] >= 0.52:
        return True
    return False


def _selection_rank(b: dict[str, Any]) -> tuple[float, float, float, float, float]:
    _, q = _candidate_quality_values(b)
    if _is_low_leap_bridge(b):
        rank = (
            0.34 * q["phon"]
            + 0.26 * q["pivot"]
            + 0.22 * q["natural"]
            + 0.18 * q["surprise"]
        )
    else:
        rank = (
            0.30 * q["phon"]
            + 0.24 * q["pivot"]
            + 0.20 * q["natural"]
            + 0.18 * q["surprise"]
            + 0.08 * q["semantic_domain"]
        )
    return (-rank, -q["phon"], -q["pivot"], -q["surprise"], -q["natural"])




def _candidate_passes_generator_floor(b: dict[str, Any]) -> bool:
    return _strict_candidate(b) or _creative_candidate(b)

# ─────────────────────────────────────────────────────────────────────────────
# without weakening junk filters.
#
# This does not lower export floors.  It adds another candidate source only when
# the strict bridge miner has too few ideas: clean phonetic-neighbor affordances
# from the already-computed phonetic A/B/pun lists.  These candidates are allowed
# to have weak French semantic domain-meaning similarity, but they must still pass the same
# natural French, pivot-usability, and sound/surprise gates.
# ─────────────────────────────────────────────────────────────────────────────
TARGET_GENERATOR_IDEAS_COMPENSATION = int(os.environ.get("RETRIEVAL_TARGET_GENERATOR_IDEAS", "6"))
MAX_GENERATOR_IDEAS_COMPENSATION = int(os.environ.get("RETRIEVAL_MAX_GENERATOR_IDEAS", str(max(MAX_GENERATOR_IDEAS_STRICT_CREATIVE, 10))))
MAX_PHONETIC_COMPENSATION_PER_SOURCE = int(os.environ.get("RETRIEVAL_MAX_PHONETIC_COMPENSATION_PER_SOURCE", "18"))


def _phonetic_affordance_bridge(r: dict[str, Any], default_side: str) -> dict[str, Any]:
    """Convert an already-computed phonetic neighbor into a bridge-shaped item.

    This is the Low-style compensation source: the right side can live outside
    the original semantic range, but it must still be an ordinary French pun
    pivot.  No new generator schema is introduced.
    """
    probe = clean(r.get("probe_text", ""))
    cand = clean(r.get("word", r.get("surface", "")))
    if not probe or not cand:
        return {}
    if surface_key(probe) == surface_key(cand):
        return {}
    if _surface_pair_hard_reject(probe, cand, {"bridge_type": "phonetic_compensation"}):
        return {}
    phon = float(r.get("final_score", r.get("phonetic_score", 0.0)) or 0.0)
    if phon < MIN_EXPANSION_PHONETIC:
        return {}
    # Strong sound collision with deliberately weak opposite semantic grounding:
    # this searches other semantic ranges while preserving recoverability through
    # the original probe side.
    source_sem = 0.62 if default_side in {"A", "B"} else 0.48
    opp_sem = 0.05
    same_root_flag = same_root(probe, cand) or structurally_trivial_variant(probe, cand) or boring_morphophonetic_echo(probe, cand)
    naturalness = max(surface_naturalness_score(probe), surface_naturalness_score(cand), expression_quality({"surface": cand}))
    surprise = max(0.0, humor_surprise_score(phon, source_sem, opp_sem, same_root_flag))
    b = {
        "bridge_type": "phonetic_compensation",
        "relation": "phonetic_compensation",
        "source_side": default_side,
        "opposite_side": "other_semantic_range",
        "sound_source": probe,
        "candidate": cand,
        "source_ipa": clean(r.get("probe_ipa", "")),
        "candidate_ipa": clean(r.get("ipa", "")),
        "source_semantic_score": source_sem,
        "opposite_semantic_score": opp_sem,
        "phonetic_score": phon,
        "naturalness_score": float(naturalness),
        "surprise_score": float(surprise),
        "semantic_verified": False,
        "semantic_relation": "other_semantic_range",
        "affordance_stage": f"{default_side}_phonetic_compensation",
        "phonetic_relation": phonetic_relation_label(phon, same_ipa=clean(r.get("probe_ipa", "")) == clean(r.get("ipa", ""))),
        "same_root_penalty_applied": bool(same_root_flag),
    }
    b["llm_priority_score"] = llm_priority_score_for_bridge(b)
    return b


def _compensation_candidates_from_pack(pack: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sources = [
        ("A", pack.get("phonetic_A_candidates", []) or []),
        ("B", pack.get("phonetic_B_candidates", []) or []),
    ]
    for side, rows in sources:
        taken = 0
        for r in rows:
            if taken >= MAX_PHONETIC_COMPENSATION_PER_SOURCE:
                break
            b = _phonetic_affordance_bridge(r, side)
            if not b:
                continue
            # Same quality bar as : no weaker junk gate for compensation.
            if not (_strict_candidate(b) or _creative_candidate(b)):
                continue
            out.append(b)
            taken += 1
    out.sort(key=_selection_rank)
    return out


def _append_ideas(selected: list[dict[str, Any]], bridges: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set(_idea_pair_key(x) for x in selected if x)
    surface_counts: dict[str, int] = {}
    for idea in selected:
        if not idea:
            continue
        for side in ("left", "right"):
            k = surface_key(idea.get(side, ""))
            if k:
                surface_counts[k] = surface_counts.get(k, 0) + 1
    for b in sorted(bridges, key=_selection_rank):
        if len(selected) >= limit:
            break
        idea = compact_generator_idea(b)
        if not idea:
            continue
        key = _idea_pair_key(idea)
        if key in seen:
            continue
        left, right = clean(idea.get("left", "")), clean(idea.get("right", ""))
        if _surface_pair_hard_reject(left, right, b):
            continue
        if not (_strict_candidate(b) or _creative_candidate(b)):
            continue
        lk, rk = surface_key(left), surface_key(right)
        if surface_counts.get(lk, 0) >= MAX_IDEAS_PER_SURFACE or surface_counts.get(rk, 0) >= MAX_IDEAS_PER_SURFACE:
            continue
        selected.append(idea)
        seen.add(key)
        surface_counts[lk] = surface_counts.get(lk, 0) + 1
        surface_counts[rk] = surface_counts.get(rk, 0) + 1
    return selected[:limit]




# ─────────────────────────────────────────────────────────────────────────────
#
# This is additive only. It preserves the existing strict retrieval/export gates
# and adds more candidates only when the row has too few generator affordances.
# Both A and B sides expand outward. Stopping is controlled by target count,
# depth, and a small runtime budget; semantic closeness to French semantic domain is not a gate.
# ─────────────────────────────────────────────────────────────────────────────
LOW_LEAP_TARGET_GENERATOR_IDEAS = int(os.environ.get("RETRIEVAL_TARGET_GENERATOR_IDEAS", "6"))
LOW_LEAP_MAX_DEPTH = int(os.environ.get("RETRIEVAL_LOW_LEAP_MAX_DEPTH", "4"))
LOW_LEAP_FRONTIER_PER_SIDE = int(os.environ.get("RETRIEVAL_LOW_LEAP_FRONTIER_PER_SIDE", "8"))
LOW_LEAP_EXPANSIONS_PER_QUERY = int(os.environ.get("RETRIEVAL_LOW_LEAP_EXPANSIONS_PER_QUERY", "6"))
LOW_LEAP_PHONETIC_NEIGHBORS = int(os.environ.get("RETRIEVAL_LOW_LEAP_PHONETIC_NEIGHBORS", "10"))
LOW_LEAP_MAX_BRIDGES = int(os.environ.get("RETRIEVAL_LOW_LEAP_MAX_BRIDGES", "40"))
LOW_LEAP_TIME_BUDGET_SEC = float(os.environ.get("RETRIEVAL_LOW_LEAP_TIME_BUDGET_SEC", "3.5"))


def _low_leap_seed_surfaces(pack: dict[str, Any], side: str) -> list[str]:
    if side == "A":
        base = list(pack.get("meaning_A_terms", []) or [])
        sem = pack.get("semantic_A_expressions", []) or []
    else:
        base = list(pack.get("meaning_B_terms", []) or [])
        sem = pack.get("semantic_B_expressions", []) or []
    extra = [clean(x.get("surface", "")) for x in sem[:SIDE_SEMANTIC_K]]
    return [x for x in unique_keep_order(base + extra, limit=LOW_LEAP_FRONTIER_PER_SIDE) if x and not lexically_bad_candidate_surface(x)]


def _low_leap_bridge_from_pair(source: str, cand: str, source_side: str, depth: int, phon: float, source_sem: float = 0.35) -> dict[str, Any]:
    if not source or not cand or surface_key(source) == surface_key(cand):
        return {}
    if _surface_pair_hard_reject(source, cand, {"bridge_type": "low_leap"}):
        return {}
    if looks_like_low_value_finite_form(source) or looks_like_low_value_finite_form(cand):
        return {}
    same_root_flag = same_root(source, cand) or structurally_trivial_variant(source, cand) or boring_morphophonetic_echo(source, cand)
    if same_root_flag:
        return {}
    naturalness = max(
        surface_naturalness_score(source),
        surface_naturalness_score(cand),
        expression_quality({"surface": source}),
        expression_quality({"surface": cand}),
    )
    # Low-leap candidates are allowed to be far from the French semantic domain meanings. Keep a
    # small nonzero French semantic domain signal so existing score profile stays well-formed.
    semantic_domain_sim = max(0.02, min(0.28, float(source_sem or 0.0)))
    surprise = humor_surprise_score(float(phon), float(source_sem or 0.0), 0.03, same_root_flag)
    b = {
        "bridge_type": "low_leap",
        "relation": "low_leap",
        "source_side": source_side,
        "opposite_side": "low_leap",
        "sound_source": source,
        "candidate": cand,
        "source_semantic_score": float(source_sem or 0.0),
        "opposite_semantic_score": float(semantic_domain_sim),
        "phonetic_score": float(phon),
        "naturalness_score": float(naturalness),
        "surprise_score": float(surprise),
        "semantic_verified": False,
        "semantic_relation": "low_leap",
        "affordance_stage": f"{source_side}{depth}_low_leap",
        "phonetic_relation": phonetic_relation_label(float(phon), same_ipa=False),
        "same_root_penalty_applied": False,
    }
    b["llm_priority_score"] = llm_priority_score_for_bridge(b)
    return b


def _mine_iterative_low_leap_bridges(pipe: RetrievalPipeline, pack: dict[str, Any], existing_ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk outward from both meaning sides and mine high-quality phonetic pairs.

    This does not relax junk gates. It changes where we look: after the close
    semantic region fails to produce enough ideas, it makes Low-style semantic
    leaps outward and searches for French sound collisions there.
    """
    t0 = time.time()
    seen_surfaces: set[str] = set()
    for idea in existing_ideas:
        if not idea:
            continue
        seen_surfaces.add(surface_key(idea.get("left", "")))
        seen_surfaces.add(surface_key(idea.get("right", "")))

    all_bridges: list[dict[str, Any]] = []
    frontiers = {
        "A": _low_leap_seed_surfaces(pack, "A"),
        "B": _low_leap_seed_surfaces(pack, "B"),
    }
    visited = {"A": {surface_key(x) for x in frontiers["A"]}, "B": {surface_key(x) for x in frontiers["B"]}}

    for depth in range(1, LOW_LEAP_MAX_DEPTH + 1):
        if len(existing_ideas) + len(all_bridges) >= LOW_LEAP_TARGET_GENERATOR_IDEAS:
            break
        if time.time() - t0 > LOW_LEAP_TIME_BUDGET_SEC:
            break

        requests: list[tuple[str, int, str]] = []
        channel_meta: dict[str, tuple[str, str]] = {}
        for side in ("A", "B"):
            for i, term in enumerate(frontiers[side][:LOW_LEAP_FRONTIER_PER_SIDE]):
                if not term:
                    continue
                ch = f"low_leap_{side}_{depth}_{i}"
                requests.append((term, LOW_LEAP_EXPANSIONS_PER_QUERY, ch))
                channel_meta[ch] = (side, term)
        if not requests:
            break

        try:
            expanded = pipe.expression.semantic_search_many(requests)
        except Exception:
            expanded = {}

        next_frontiers = {"A": [], "B": []}
        source_nodes: list[dict[str, Any]] = []
        for ch, rows in expanded.items():
            side, parent = channel_meta.get(ch, ("", ""))
            if side not in {"A", "B"}:
                continue
            for r in rows or []:
                surf = clean(r.get("surface", ""))
                if not surf or lexically_bad_candidate_surface(surf) or looks_like_low_value_finite_form(surf):
                    continue
                sk = surface_key(surf)
                if not sk or sk in visited[side]:
                    continue
                # Avoid semantically trivial expansion chains; we are looking for
                # usable leaps, not variants of the same surface/root.
                if same_root(parent, surf) or structurally_trivial_variant(parent, surf):
                    continue
                visited[side].add(sk)
                sem_score = float(r.get("score", 0.0) or 0.0)
                node = {"side": side, "surface": surf, "semantic_score": sem_score, "depth": depth}
                source_nodes.append(node)
                if len(next_frontiers[side]) < LOW_LEAP_FRONTIER_PER_SIDE:
                    next_frontiers[side].append(surf)

        # Phonetic search for all leapt surfaces, batched by existing helper.
        # search_from_text already uses caches; the bounded source_nodes list keeps
        # runtime predictable.
        for node in source_nodes[: LOW_LEAP_FRONTIER_PER_SIDE * 2]:
            source = node["surface"]
            side = node["side"]
            try:
                neigh = pipe.phonetic.search_from_text(source, top_k=LOW_LEAP_PHONETIC_NEIGHBORS)
            except Exception:
                neigh = []
            for r in neigh or []:
                cand = clean(r.get("word", r.get("surface", "")))
                if not cand or surface_key(cand) in seen_surfaces:
                    continue
                phon = float(r.get("final_score", r.get("phonetic_score", 0.0)) or 0.0)
                b = _low_leap_bridge_from_pair(source, cand, side, depth, phon, float(node.get("semantic_score", 0.0)))
                if not b:
                    continue
                if not (_strict_candidate(b) or _creative_candidate(b)):
                    continue
                all_bridges.append(b)
                if len(all_bridges) >= LOW_LEAP_MAX_BRIDGES:
                    break
            if len(all_bridges) >= LOW_LEAP_MAX_BRIDGES:
                break
        if len(all_bridges) >= LOW_LEAP_MAX_BRIDGES:
            break

        # Continue walking outward from both sides.
        frontiers = next_frontiers
        if not frontiers["A"] and not frontiers["B"]:
            break

    all_bridges.sort(key=_selection_rank)
    return all_bridges[:LOW_LEAP_MAX_BRIDGES]


# ─────────────────────────────────────────────────────────────────────────────
#
# This fixes the export bottleneck without changing retrieval_server.py and
# without requiring a server restart. The server hot-reloads retrieval_refactored.py on each
# request, so these overrides take effect on the next /retrieve call.
#
# Selection policy:
#   1. Generate/scoring remains shared and cached upstream.
#   2. Close-semantic, compensation, and Low-leap candidates are retained in
#      independent lanes.
#   3. A-side and B-side candidates are retained in independent lanes.
#   4. Low-leap depth levels are retained in independent lanes.
#   5. The only cross-lane removal is exact duplicate surface/relation pairs.
#   6. No root-collapse, no per-surface cap, no global top-N competition.
# ─────────────────────────────────────────────────────────────────────────────

# Large by default because this file is an affordance miner. Downstream generator
# or judge stages can choose how many to use. Override in env if needed.
MAX_GENERATOR_IDEAS = int(os.environ.get(
    "RETRIEVAL_MAX_GENERATOR_IDEAS",
    str(max(MAX_GENERATOR_IDEAS_COMPENSATION, MAX_GENERATOR_AFFORDANCES, 64)),
))
# Safety cap only, not a ranking target. Set 0 for unlimited per row.
MAX_GENERATOR_IDEAS_ABSOLUTE = int(os.environ.get("RETRIEVAL_MAX_GENERATOR_IDEAS_ABSOLUTE", "200"))


def _bridge_exact_export_key(b: dict[str, Any]) -> tuple[str, str, str]:
    """Exact exported-idea duplicate key only.

    Deliberately does NOT use roots/lemmas and does NOT canonicalize A/B order:
    A→B and B→A are allowed to coexist unless their exported surfaces and
    relation are literally the same.
    """
    idea = compact_generator_idea(b)
    if not idea:
        left, right = bridge_surface_pair(b)
        relation = clean(b.get("phonetic_relation") or b.get("relation") or b.get("bridge_type") or "")
    else:
        left, right = idea.get("left", ""), idea.get("right", "")
        relation = idea.get("relation", "")
    return (surface_key(left), surface_key(right), clean(relation))


def _low_leap_depth(b: dict[str, Any]) -> int:
    stage = clean(b.get("affordance_stage", b.get("stage", ""))).lower()
    m = re.search(r"(?:^|[^ab])([ab])(\d+)_low_leap|([ab])(\d+)_low_leap", stage)
    if m:
        return int(m.group(2) or m.group(4) or 0)
    # Fallback for any future low-leap bridge shape.
    for key in ("depth", "low_leap_depth", "source_level", "left_level", "right_level"):
        try:
            val = int(b.get(key, 0) or 0)
            if val > 0:
                return val
        except Exception:
            pass
    return 0


def _bridge_side(b: dict[str, Any]) -> str:
    side = clean(b.get("source_side") or b.get("left_side") or "").upper()
    if side in {"A", "B"}:
        return side
    stage = clean(b.get("affordance_stage", b.get("stage", ""))).upper()
    if stage.startswith("A") or "_A_" in stage or "A_TO" in stage:
        return "A"
    if stage.startswith("B") or "_B_" in stage or "B_TO" in stage:
        return "B"
    return "AB"


def _bridge_export_lane(b: dict[str, Any]) -> str:
    """Independent retention lane for additive export."""
    side = _bridge_side(b)
    if is_detached_bridge(b):
        bucket = clean(b.get("retrieval_bucket", "detached")) or "detached"
        return f"detached_{bucket}"
    if _is_low_leap_bridge(b):
        depth = _low_leap_depth(b)
        return f"low_{side}_depth_{depth}"

    stage = clean(b.get("affordance_stage", b.get("stage", ""))).lower()
    btype = clean(b.get("bridge_type", b.get("relation", ""))).lower()
    if "compensation" in stage or "compensation" in btype:
        return f"compensation_{side}"

    # Direct A×B bridges are their own close lane; expansion bridges keep their
    # source side, so A→B and B→A never compete.
    if side == "AB":
        return "close_AB_direct"
    return f"close_{side}_to_other"


def _candidate_passes_export_junk_gates(b: dict[str, Any]) -> bool:
    """Same junk/quality gates as current strict/creative, but no competitive caps."""
    if not b:
        return False
    idea = compact_generator_idea(b)
    if not idea:
        return False
    left, right = clean(idea.get("left", "")), clean(idea.get("right", ""))
    if _surface_pair_hard_reject(left, right, b):
        return False
    if is_detached_bridge(b):
        # Detached buckets have their own hard junk gate and intentionally do not
        # require the normal strict/creative pivotability threshold.
        return detached_bridge_is_structurally_valid(b)
    # Preserve the current quality floor. Low-leap is allowed semantic distance,
    # not lower French/phonetic/pivot quality.
    return bool(_strict_candidate(b) or _creative_candidate(b))


def _lane_sort_key(b: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """Lane-local ordering only. This never decides between lanes."""
    _, q = _candidate_quality_values(b)
    # Close candidates may use French semantic domain similarity; Low-leap candidates should not
    # be punished for being far from French semantic domain, so use the  rank behavior.
    base = _selection_rank(b)
    lane = _bridge_export_lane(b)
    # Stable tie-breakers for deterministic TSVs.
    left, right = bridge_surface_pair(b)
    return (*base, surface_key(left), surface_key(right), lane)


def _additive_generator_ideas_from_bridges(
    bridges: list[dict[str, Any]],
    limit: int = MAX_GENERATOR_IDEAS,
) -> list[dict[str, Any]]:
    """Return all good generator ideas by additive independent lanes.

    This is the core fix: candidates only compete with duplicates of themselves,
    not with other directions, depths, or strict-vs-Low lanes.
    """
    if not bridges:
        return []

    lanes: dict[str, list[dict[str, Any]]] = {}
    rejected = 0
    for b in bridges:
        if not _candidate_passes_export_junk_gates(b):
            rejected += 1
            continue
        lane = _bridge_export_lane(b)
        lanes.setdefault(lane, []).append(b)

    # Deterministic lane order: close first, then compensation, then Low-leap by
    # depth and side. Unknown/new lanes are appended alphabetically.
    def lane_order_key(lane: str) -> tuple[int, int, str]:
        if lane == "close_AB_direct":
            return (0, 0, lane)
        if lane.startswith("close_A"):
            return (1, 0, lane)
        if lane.startswith("close_B"):
            return (1, 1, lane)
        if lane.startswith("compensation_A"):
            return (2, 0, lane)
        if lane.startswith("compensation_B"):
            return (2, 1, lane)
        if lane.startswith("detached_A2"):
            return (3, 0, lane)
        if lane.startswith("detached_B2"):
            return (3, 1, lane)
        if lane.startswith("detached_"):
            return (3, 2, lane)
        m = re.match(r"low_([AB]+)_depth_(\d+)", lane)
        if m:
            side_rank = 0 if m.group(1) == "A" else 1 if m.group(1) == "B" else 2
            return (4 + int(m.group(2)), side_rank, lane)
        return (99, 0, lane)

    for lane in lanes:
        lanes[lane].sort(key=_lane_sort_key)

    selected: list[dict[str, Any]] = []
    seen_exact: set[tuple[str, str, str]] = set()

    # Additive concatenation. No lane gets a budget that can be stolen by another
    # lane. The absolute limit is a safety rail only.
    absolute = MAX_GENERATOR_IDEAS_ABSOLUTE if MAX_GENERATOR_IDEAS_ABSOLUTE > 0 else 10**9
    requested_limit = limit if limit and limit > 0 else absolute
    final_limit = min(requested_limit, absolute)

    for lane in sorted(lanes, key=lane_order_key):
        for b in lanes[lane]:
            key = _bridge_exact_export_key(b)
            if key in seen_exact:
                continue
            idea = compact_generator_idea(b)
            if not idea:
                continue
            idea["export_lane"] = lane
            selected.append(idea)
            seen_exact.add(key)
            if len(selected) >= final_limit:
                return selected

    return selected


# Override the historical selector name used throughout the file. The name stays
# the same so retrieval_server.py and older call sites continue to work.
def _select_generator_ideas_from_bridges(
    bridges: list[dict[str, Any]],
    limit: int = MAX_GENERATOR_IDEAS,
) -> list[dict[str, Any]]:
    return _additive_generator_ideas_from_bridges(bridges, limit=limit)


def _retrieval_columns_from_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Return TSV retrieval columns.

    Default output is intentionally clean: only generator-facing retrieval outputs
    are written. Set RETRIEVAL_OUTPUT_DEBUG=1 to include diagnostics/bucket counts.
    Set RETRIEVAL_DEBUG_PACKS=1 to additionally include the compact debug pack.
    """
    diag = pack.get("bridge_diagnostics", {}) or {}
    bridge_candidates = pack.get("bridge_candidates", []) or []
    ideas = _additive_generator_ideas_from_bridges(
        bridge_candidates,
        limit=MAX_GENERATOR_IDEAS,
    )

    # Core output columns only. These are the actual retrieval payload consumed by
    # the next stage, not diagnostics.
    out = {
        "retrieval_affordances_json": _safe_json_for_tsv(ideas),
        "retrieval_affordance_count": int(len(ideas)),
    }

    if not RETRIEVAL_OUTPUT_DEBUG and not RETRIEVAL_DEBUG_PACKS:
        return out

    # Debug counts make it obvious whether generation found candidates but export
    # removed them. These are intentionally gated out of normal TSV output.
    lane_counts: dict[str, int] = {}
    pass_count = 0
    for b in bridge_candidates:
        if _candidate_passes_export_junk_gates(b):
            pass_count += 1
            lane = _bridge_export_lane(b)
            lane_counts[lane] = lane_counts.get(lane, 0) + 1

    exported_bridge_candidates = export_bridge_candidates(bridge_candidates, MAX_GENERATOR_AFFORDANCES)
    bucket_counts = diag.get("retrieval_exported_bucket_counts") or diag.get("retrieval_bucket_counts_after_final_sanitize") or retrieval_bucket_counts(bridge_candidates)
    affordance_buckets = group_exported_affordances_by_retrieval_bucket(exported_bridge_candidates)

    if RETRIEVAL_OUTPUT_DEBUG:
        out.update({
            "retrieval_fallback_level": clean(pack.get("fallback_level", "")),
            "retrieval_bridge_count": int(diag.get("bridge_count", len(bridge_candidates)) or 0),
            "retrieval_prefinal_bridge_candidate_count": int(diag.get("prefinal_bridge_candidate_count", 0) or 0),
            "retrieval_prefinal_bridge_candidate_limit": int(diag.get("prefinal_bridge_candidate_limit", 0) or 0),
            "retrieval_best_bridge_score": float(diag.get("best_bridge_score", 0.0) or 0.0),
            "retrieval_bucket_counts_json": _safe_json_for_tsv(bucket_counts),
            "retrieval_a1_pool_size": int(diag.get("a1_pool_size", 0) or 0),
            "retrieval_b1_pool_size": int(diag.get("b1_pool_size", 0) or 0),
            "retrieval_a1b1_widening_enabled": bool(diag.get("a1b1_widening_enabled", False)),
            "retrieval_a1b1_per_seed_semantic_k": int(diag.get("a1b1_per_seed_semantic_k", 0) or 0),
            "retrieval_fasttext_default_disabled_v15": bool(diag.get("fasttext_default_disabled_v15", False)),
            "retrieval_bucket_counts_before_filter_json": _safe_json_for_tsv(diag.get("retrieval_bucket_counts_before_filter", {})),
            "retrieval_bucket_counts_after_filter_json": _safe_json_for_tsv(diag.get("retrieval_bucket_counts_after_filter", {})),
            "retrieval_bucket_counts_after_dedupe_json": _safe_json_for_tsv(diag.get("retrieval_bucket_counts_after_dedupe", {})),
            "retrieval_detached_rejection_totals_json": _safe_json_for_tsv(diag.get("detached_rejection_totals", {})),
            "retrieval_detached_rejection_stats_json": _safe_json_for_tsv(diag.get("detached_rejection_stats", [])),
            "retrieval_detached_diag_version": DETACHED_DIAGNOSTICS_VERSION,
            "retrieval_detached_neighbors_seen": int(diag.get("detached_neighbors_seen_total", 0) or 0),
            "retrieval_detached_presemantic_count": int(diag.get("detached_candidates_before_semantic_distance_total", 0) or 0),
            "retrieval_detached_semantic_scored_count": int(diag.get("detached_semantic_distance_scored_total", 0) or 0),
            "retrieval_detached_accepted_before_anchor_cap": int(diag.get("detached_accepted_before_anchor_cap_total", 0) or 0),
            "retrieval_detached_accepted_after_anchor_cap": int(diag.get("detached_accepted_after_anchor_cap_total", 0) or 0),
            "retrieval_detached_final_input_count": int(diag.get("detached_final_input_count", 0) or 0),
            "retrieval_detached_final_kept_count": int(diag.get("detached_final_kept_count", 0) or 0),
            "retrieval_detached_final_rejection_totals_json": _safe_json_for_tsv(diag.get("detached_final_rejection_totals", {})),
            "retrieval_detached_final_rejections_by_bucket_json": _safe_json_for_tsv(diag.get("detached_final_rejections_by_bucket", {})),
            "retrieval_affordance_buckets_json": _safe_json_for_tsv(affordance_buckets),
        })

    if RETRIEVAL_DEBUG_PACKS:
        compact = compact_retrieval_pack(pack)
        compact["additive_export_pass_count"] = pass_count
        compact["additive_export_lane_counts"] = lane_counts
        out["retrieval_debug_json"] = _safe_json_for_tsv(compact)

    return out


def compact_retrieval_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Compact generator-facing retrieval payload with additive affordances."""
    gen = dict(pack.get("generator_affordance_pack", {}))
    gen["meaning_A_terms"] = pack.get("meaning_A_terms", gen.get("meaning_A_terms", []))[:8]
    gen["meaning_B_terms"] = pack.get("meaning_B_terms", gen.get("meaning_B_terms", []))[:8]
    gen["fallback_level"] = pack.get("fallback_level", gen.get("fallback_level", ""))
    gen["bridge_diagnostics"] = pack.get("bridge_diagnostics", gen.get("bridge_diagnostics", {}))

    bridges = pack.get("bridge_candidates", gen.get("top_bridge_candidates", [])) or []
    gen["top_bridge_candidates"] = [
        export_bridge_candidate(b) for b in _additive_export_bridges(bridges, limit=MAX_GENERATOR_IDEAS)
    ]
    gen["top_semantic_A"] = pack.get("semantic_A_expressions", gen.get("top_semantic_A", []))[:5]
    gen["top_semantic_B"] = pack.get("semantic_B_expressions", gen.get("top_semantic_B", []))[:5]
    gen["top_semantic_blended"] = pack.get("semantic_expressions", gen.get("top_semantic_blended", []))[:5]
    gen["top_phonetic_A"] = pack.get("phonetic_A_candidates", gen.get("top_phonetic_A", []))[:5]
    gen["top_phonetic_B"] = pack.get("phonetic_B_candidates", gen.get("top_phonetic_B", []))[:5]

    judge_source = gen.get("top_bridge_candidates", [])[:LLM_JUDGE_CANDIDATE_LIMIT]
    gen["llm_judge_candidates"] = [
        {
            "left": c.get("source_surface", c.get("a_surface", "")),
            "right": c.get("candidate_surface", c.get("b_surface", "")),
            "relation": c.get("phonetic_relation", c.get("relation", "")),
            "score": c.get("llm_priority_score", c.get("bridge_score", 0.0)),
            "affordance_stage": c.get("affordance_stage", ""),
        }
        for c in judge_source
    ]
    return gen


def _additive_export_bridges(bridges: list[dict[str, Any]], limit: int = MAX_GENERATOR_IDEAS) -> list[dict[str, Any]]:
    """Same additive policy as ideas, but returns bridge dicts for debug pack export."""
    lanes: dict[str, list[dict[str, Any]]] = {}
    for b in bridges or []:
        if not _candidate_passes_export_junk_gates(b):
            continue
        lanes.setdefault(_bridge_export_lane(b), []).append(b)
    for lane in lanes:
        lanes[lane].sort(key=_lane_sort_key)
    def lane_order_key(lane: str) -> tuple[int, int, str]:
        if lane == "close_AB_direct":
            return (0, 0, lane)
        if lane.startswith("close_A"):
            return (1, 0, lane)
        if lane.startswith("close_B"):
            return (1, 1, lane)
        if lane.startswith("compensation_A"):
            return (2, 0, lane)
        if lane.startswith("compensation_B"):
            return (2, 1, lane)
        if lane.startswith("detached_A2"):
            return (3, 0, lane)
        if lane.startswith("detached_B2"):
            return (3, 1, lane)
        if lane.startswith("detached_"):
            return (3, 2, lane)
        m = re.match(r"low_([AB]+)_depth_(\d+)", lane)
        if m:
            return (4 + int(m.group(2)), 0 if m.group(1) == "A" else 1, lane)
        return (99, 0, lane)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    absolute = MAX_GENERATOR_IDEAS_ABSOLUTE if MAX_GENERATOR_IDEAS_ABSOLUTE > 0 else 10**9
    final_limit = min(limit if limit and limit > 0 else absolute, absolute)
    for lane in sorted(lanes, key=lane_order_key):
        for b in lanes[lane]:
            key = _bridge_exact_export_key(b)
            if key in seen:
                continue
            nb = dict(b)
            nb["export_lane"] = lane
            nb["structural_guard_passed"] = True
            out.append(nb)
            seen.add(key)
            if len(out) >= final_limit:
                return out
    return out


def export_bridge_candidates(bridges: list[dict[str, Any]], limit: int = MAX_GENERATOR_IDEAS) -> list[dict[str, Any]]:
    return [export_bridge_candidate(b) for b in _additive_export_bridges(bridges, limit=limit)]




if __name__ == "__main__":
    main()
