import asyncio
import ast
import os
import sys
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import pandas as pd

from data import load, load_all, save
from config import combined_en_path, homonym_dir, identify_dir, translate_dir, similarity_dir
from utils import get_model, get_response_async

import re
import difflib
import collections
from dataclasses import dataclass

pd.options.mode.chained_assignment = None

DEFAULT_MODEL = os.environ.get("PREPROCESSOR_DEFAULT_MODEL", "google/gemini-3-pro")
VERBOSE = os.environ.get("PREPROCESSOR_VERBOSE", "1") == "1"
MAX_CONCURRENCY = int(os.environ.get("PREPROCESSOR_MAX_CONCURRENCY", "8"))
# LEXIQUE_PATH = os.environ.get("LEXIQUE_PATH", "Lexique383.tsv")
BASE_DIR = os.path.dirname(__file__)
LEXIQUE_PATH = os.environ.get(
    "LEXIQUE_PATH",
    os.path.join(BASE_DIR, "Lexique383.tsv")
)

def log(*args):
    if VERBOSE:
        print(*args)

def safe_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            value = ast.literal_eval(x)
            return value if isinstance(value, list) else []
        except (ValueError, SyntaxError):
            try:
                import json
                value = json.loads(x.replace("'", '"'))
                return value if isinstance(value, list) else []
            except Exception:
                return []
    return []

def log_and_build_fallback(error: Exception, payload: dict[str, Any]) -> pd.Series:
    print(f"Error: {error}")
    return pd.Series(payload)


async def run_async_apply(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[Any]],
    result_columns: list[str],
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(index, row):
        async with semaphore:
            result = await apply_async_fn(row)
            return index, result

    tasks = [asyncio.create_task(worker(index, row)) for index, row in chunk_df.iterrows()]
    results = {}

    try:
        for task in asyncio.as_completed(tasks):
            index, result = await task
            results[index] = result
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    ordered_rows = [results[index] for index in chunk_df.index]
    result_df = pd.DataFrame(ordered_rows, index=chunk_df.index)
    return result_df[result_columns]




# ─────────────────────────────────────────────────────────────────────────────
# Low's Algorithm — polygon translator
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class LowCandidate:
    """A French word found by Low's polygon search."""
    word: str
    polygon_level: int          # 4=square, 5=pentagon, 6=hexagon, 7=heptagon+
    path: List[str]             # human-readable search path
    sim1: float                 # semantic overlap with meaning1
    sim2: float                 # semantic overlap with meaning2
    similarity_score: float            # composite Low score
    relation_type: str          # "homophone" | "synonym" | "direct"
 
 
@dataclass
class LowResult:
    """Full output of the polygon search for one row."""
    pun_word_low: str           # best French pun candidate (empty string if none)
    polygon_level: int          # level at which it was found (-1 = fallback)
    similarity_score: float
    path: List[str]
    strategy: str               # "square" | "pentagon" | "hexagon" | "heptagon" | "literal_fallback"
    error: str                  # non-empty if something went wrong
 
 
class LexiquePhoneticIndex:
    """
    Loads Lexique383 and provides:
      - word_to_phon : dict[str, str]   word → IPA/phonetic string
      - phon_to_words: dict[str, list]  phonetic → all words sharing it
      - find_homophones(word)           words that sound the same
    """
 
    def __init__(self, lexique_path: str):
        if not os.path.exists(lexique_path):
            raise FileNotFoundError(
                f"Lexique383 not found at '{lexique_path}'.\n"
                "Download from http://lexique.org and set LEXIQUE_PATH env var."
            )
        df = self._load(lexique_path)
        cols = {str(c).lower(): c for c in df.columns}
        ortho = cols.get("ortho") or cols.get("word") or list(df.columns)[0]
        phon  = cols.get("phon") or cols.get("phonology") or cols.get("ipa")
        if phon is None:
            raise ValueError(f"No phonetic column found in Lexique. Columns: {list(df.columns)[:10]}")
 
        self.word_to_phon: Dict[str, str] = {}
        self.phon_to_words: Dict[str, List[str]] = {}
        self.vocab: set = set()
 
        for _, row in df[[ortho, phon]].dropna().iterrows():
            w = str(row[ortho]).strip().lower()
            p = str(row[phon]).strip()
            if not w or not p:
                continue
            self.word_to_phon[w] = p
            self.phon_to_words.setdefault(p, []).append(w)
            self.vocab.add(w)
 
        log(f"  Lexique loaded: {len(self.word_to_phon):,} words, "
            f"{len(self.phon_to_words):,} phonetic groups")
 
    @staticmethod
    def _load(path: str) -> pd.DataFrame:
        for sep in ["\t", ";", ",", None]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
                if df is not None and len(df.columns) >= 2 and len(df) > 0:
                    return df
            except Exception:
                pass
        raise ValueError("Could not parse Lexique file with any separator.")
 
    def find_homophones(self, word: str, limit: int = 30) -> List[str]:
        """Return French words that share the same phonetic representation."""
        p = self.word_to_phon.get(word.lower().strip())
        if not p:
            return []
        return [w for w in self.phon_to_words.get(p, []) if w != word.lower()][:limit]
 
    def near_homophones(self, word: str, limit: int = 20, cutoff: float = 0.80) -> List[str]:
        """
        Return French words whose phonetic string is very similar (not exact).
        Useful when the pun word itself isn't in Lexique but a close variant is.
        """
        target_phon = self.word_to_phon.get(word.lower().strip(), "")
        if not target_phon:
            return []
        candidates = []
        for phon_str, words in self.phon_to_words.items():
            if phon_str == target_phon:
                continue
            ratio = difflib.SequenceMatcher(None, target_phon, phon_str).ratio()
            if ratio >= cutoff:
                candidates.extend(words)
        return candidates[:limit]
 
 
class FrenchSemanticIndex:
    """
    Lightweight semantic similarity for French words using:
      1. WordNet (via NLTK's multilingual support, lang='fra')
      2. Edit-distance fallback when no synsets available
    """
 
    def __init__(self):
        try:
            from nltk.corpus import wordnet as wn
            self._wn = wn
        except ImportError:
            self._wn = None
        self._cache: Dict[Tuple[str, str], float] = {}
 
    def _synsets(self, word: str):
        if self._wn is None:
            return []
        try:
            ss = self._wn.synsets(word, lang="fra")
            if not ss:
                ss = self._wn.synsets(word, lang="eng")
            return ss
        except Exception:
            return []
 
    def similarity(self, w1: str, w2: str) -> float:
        w1, w2 = w1.strip().lower(), w2.strip().lower()
        if not w1 or not w2 or w1 == w2:
            return 1.0 if w1 == w2 else 0.0
        key = (min(w1, w2), max(w1, w2))
        if key in self._cache:
            return self._cache[key]
 
        ss1, ss2 = self._synsets(w1), self._synsets(w2)
        if ss1 and ss2:
            best = 0.0
            for s1 in ss1[:5]:
                for s2 in ss2[:5]:
                    try:
                        v = s1.wup_similarity(s2)
                        if v and v > best:
                            best = v
                    except Exception:
                        pass
            score = best
        else:
            # Graceful fallback: edit similarity (gives ~0.3 for related words)
            score = difflib.SequenceMatcher(None, w1, w2).ratio() * 0.35
 
        self._cache[key] = score
        return score
 
    def max_sim(self, candidate: str, targets: List[str]) -> float:
        """Max similarity between candidate and any word in targets."""
        if not targets:
            return 0.0
        return max(self.similarity(candidate, t) for t in targets)
 
    def fr_synonyms(self, fr_word: str, limit: int = 12) -> List[str]:
        """WordNet synonyms/hypernyms of a French word, returned as French strings."""
        out, seen = [], {fr_word.lower()}
        for ss in self._synsets(fr_word)[:6]:
            for lem in ss.lemmas(lang="fra"):
                n = lem.name().replace("_", " ").lower().strip()
                if not n or n in seen or " " in n or len(n) < 2:
                    continue
                seen.add(n)
                out.append(n)
                if len(out) >= limit:
                    return out
        return out
 
 
# Module-level singletons — initialised lazily so import doesn't fail if
# Lexique is not present (the LLM tasks work fine without it).
_lexique: Optional[LexiquePhoneticIndex] = None
_semantic: Optional[FrenchSemanticIndex] = None
 
 
def _get_lexique() -> Optional[LexiquePhoneticIndex]:
    global _lexique
    if _lexique is None:
        try:
            _lexique = LexiquePhoneticIndex(LEXIQUE_PATH)
        except FileNotFoundError as e:
            print(f"[Low] WARNING: {e}")
    return _lexique
 
 
def _get_semantic() -> FrenchSemanticIndex:
    global _semantic
    if _semantic is None:
        _semantic = FrenchSemanticIndex()
    return _semantic
 
 
def _norm_fr(text: str) -> str:
    return re.sub(r"[^\w' \-]", "", str(text).lower()).strip()
 
 
def _tokenize_fr(text: str) -> List[str]:
    parts = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ'\-]+", _norm_fr(text), flags=re.I)
    return [p.strip(" -'") for p in parts if len(p.strip(" -'")) >= 2]
 
 
def _similarity_score(
    sim1: float,
    sim2: float,
    relation_type: str,
    candidate: str,
    base_fr: str,
) -> float:
    """
    Low's composite score:
      - rewards candidates that overlap with BOTH meanings (the pun signal)
      - bonus for phonetic relations (more "pun-like")
      - small penalty when candidate == base_fr (no movement from direct translation)
    """
    relation_bonus = {"homophone": 0.90, "near_homophone": 0.70, "synonym": 0.55, "direct": 0.25}
    rb = relation_bonus.get(relation_type, 0.25)
    balance_penalty = abs(sim1 - sim2) * 0.15   # penalise lopsided matches
    identity_penalty = 0.10 if candidate == base_fr else 0.0
    return (
        0.45 * min(sim1, sim2)   # balance: both meanings must be covered
        + 0.30 * max(sim1, sim2) # strength: at least one meaning must be strong
        + 0.25 * rb              # relation quality
        - balance_penalty
        - identity_penalty
    )
 
 
def _score_candidate(
    candidate: str,
    base_fr: str,
    relation_type: str,
    targets1: List[str],
    targets2: List[str],
    semantic: FrenchSemanticIndex,
) -> dict:
    sim1 = semantic.max_sim(candidate, targets1)
    sim2 = semantic.max_sim(candidate, targets2)
    return {
        "candidate": candidate,
        "base_fr": base_fr,
        "relation_type": relation_type,
        "sim1": sim1,
        "sim2": sim2,
        "similarity_score": _similarity_score(sim1, sim2, relation_type, candidate, base_fr),
    }
 
 
def _best_above_threshold(
    scored: List[dict],
    min_sim: float = 0.05,
) -> Optional[dict]:
    """Return the highest-scoring candidate that clears the minimum bar."""
    scored.sort(key=lambda x: -x["similarity_score"])
    for s in scored:
        if max(s["sim1"], s["sim2"]) >= min_sim:
            return s
    return None
 
 
def _run_polygon_search(
    pun_word_fr: str,
    first_meaning_fr: List[str],
    second_meaning_fr: List[str],
    lexique: LexiquePhoneticIndex,
    semantic: FrenchSemanticIndex,
) -> LowResult:
    """
    Core Low polygon search operating entirely in French.
 
    The input meanings already come from the translate step, so we work
    in the target language throughout — no translation API calls needed here.
 
    Levels:
      4 (square)    — direct FR translation + its homophones & synonyms
      5 (pentagon)  — homophones of FR synonyms
      6 (hexagon)   — synonyms-of-synonyms + their homophones
      7 (heptagon)  — near-homophones (phonetically similar, not exact)
    """
    base = _norm_fr(pun_word_fr)
    if not base:
        return LowResult("", -1, 0.0, [], "literal_fallback", "empty pun_word_fr")
 
    t1 = [_norm_fr(w) for w in first_meaning_fr if w]
    t2 = [_norm_fr(w) for w in second_meaning_fr if w]
 
    if not t1 and not t2:
        return LowResult(base, -1, 0.0, [base], "literal_fallback", "empty meaning lists")
 
    # ── helpers ──────────────────────────────────────────────────────────────
 
    def collect_scored(
        candidates_with_type: List[Tuple[str, str, str]],  # (candidate, base_fr, relation_type)
        seen: set,
    ) -> List[dict]:
        out = []
        for cand, bfr, rtype in candidates_with_type:
            cand = _norm_fr(cand)
            if not cand:
                continue
            key = (cand, bfr, rtype)
            if key in seen:
                continue
            seen.add(key)
            out.append(_score_candidate(cand, bfr, rtype, t1, t2, semantic))
        return out
 
    def to_candidate(best: dict, level: int, strategy: str, path: List[str]) -> LowResult:
        return LowResult(
            pun_word_low=best["candidate"],
            polygon_level=level,
            similarity_score=best["similarity_score"],
            path=path,
            strategy=strategy,
            error="",
        )
 
    seen: set = set()
 
    # ── Level 4: square ───────────────────────────────────────────────────────
    # Candidates: the base word itself, its direct synonyms, its homophones.
    sq_candidates = (
        [(base, base, "direct")]
        + [(s, base, "synonym") for s in semantic.fr_synonyms(base)]
        + [(h, base, "homophone") for h in lexique.find_homophones(base, limit=25)]
    )
    sq_scored = collect_scored(sq_candidates, seen)
    best = _best_above_threshold(sq_scored)
    if best:
        return to_candidate(
            best, 4, "square",
            [pun_word_fr, best["base_fr"], best["relation_type"], best["candidate"]],
        )
 
    # ── Level 5: pentagon ──────────────────────────────────────────────────────
    # Candidates: homophones of synonyms.
    pg_candidates = []
    for syn in semantic.fr_synonyms(base, limit=15):
        for h in lexique.find_homophones(syn, limit=15):
            pg_candidates.append((h, syn, "homophone"))
    # Also try synonyms of meaning-cue words directly.
    for cue in (t1 + t2)[:6]:
        pg_candidates += [(s, cue, "synonym") for s in semantic.fr_synonyms(cue, limit=6)]
 
    pg_scored = collect_scored(pg_candidates, seen)
    best = _best_above_threshold(pg_scored)
    if best:
        return to_candidate(
            best, 5, "pentagon",
            [pun_word_fr, "synonym expansion", best["base_fr"], best["relation_type"], best["candidate"]],
        )
 
    # ── Level 6: hexagon ──────────────────────────────────────────────────────
    # Candidates: synonyms-of-synonyms + their homophones.
    # Also: homophones of the meaning-cue words themselves.
    hx_candidates = []
    for syn in semantic.fr_synonyms(base, limit=10):
        for syn2 in semantic.fr_synonyms(syn, limit=6):
            hx_candidates.append((syn2, syn, "synonym"))
            for h in lexique.find_homophones(syn2, limit=10):
                hx_candidates.append((h, syn2, "homophone"))
    for cue in (t1 + t2)[:8]:
        for h in lexique.find_homophones(cue, limit=10):
            hx_candidates.append((h, cue, "homophone"))
 
    hx_scored = collect_scored(hx_candidates, seen)
    best = _best_above_threshold(hx_scored)
    if best:
        return to_candidate(
            best, 6, "hexagon",
            [pun_word_fr, "syn²/cue-homophone", best["base_fr"], best["candidate"]],
        )
 
    # ── Level 7: heptagon — near-homophones ───────────────────────────────────
    # Last resort before literal fallback: phonetically close words.
    hp_candidates = []
    for word in [base] + semantic.fr_synonyms(base, limit=8):
        for nh in lexique.near_homophones(word, limit=15, cutoff=0.80):
            hp_candidates.append((nh, word, "near_homophone"))
 
    hp_scored = collect_scored(hp_candidates, seen)
    best = _best_above_threshold(hp_scored, min_sim=0.03)
    if best:
        return to_candidate(
            best, 7, "heptagon",
            [pun_word_fr, "near-homophone expansion", best["base_fr"], best["candidate"]],
        )
 
    # ── Literal fallback ───────────────────────────────────────────────────────
    return LowResult(
        pun_word_low=base,
        polygon_level=-1,
        similarity_score=0.0,
        path=[pun_word_fr, "no polygon found"],
        strategy="literal_fallback",
        error="",
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LLM-assisted Low: ask the model to score / improve the polygon candidate
# ─────────────────────────────────────────────────────────────────────────────
 
async def _llm_verify_candidate(
    row: pd.Series,
    candidate: LowResult,
    model: str,
) -> dict:
    """
    Optional step: ask the LLM whether the polygon candidate actually works as
    a French pun bridging both meanings, and request an improved suggestion if
    it doesn't.  Returns a dict with llm_pun_word and llm_pun_confidence.
    """
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "works_as_pun": {"type": "integer"},        # 1 = yes, 0 = no
            "improved_word": {"type": "string"},         # better FR word (or same)
            "explanation": {"type": "string"},
        },
        "required": ["works_as_pun", "improved_word", "explanation"],
    }
 
    prompt = f"""
You are a French linguistics expert working on pun translation using Low's polygon algorithm.
 
English pun word: "{row['pun_word']}"
English pun type: "{row['pun_type']}"
Meaning 1 (French synonyms): {safe_list(row['first_meaning_fr'])}
Meaning 2 (French synonyms): {safe_list(row['second_meaning_fr'])}
 
The polygon search found this French candidate: "{candidate.pun_word_low}"
Search path: {candidate.path}
Strategy used: {candidate.strategy} (polygon level {candidate.polygon_level})
 
Question 1: Does "{candidate.pun_word_low}" work as a French pun that plausibly bridges both meaning-lists above?
Output 1 for yes, 0 for no.
 
Question 2: If it does not work well, suggest a better single French word that sounds like one meaning
and carries the other (Low's polygon idea). If the candidate is already good, repeat it unchanged.
 
Return only valid JSON.
"""
 
    try:
        response = await get_response_async(
            prompt,
            model,
            response_schema=response_schema,
            required_keys=["works_as_pun", "improved_word", "explanation"],
            routing_preset="fast",
        )
        improved = str(response.get("improved_word", "")).strip() or candidate.pun_word_low
        return {
            "llm_pun_word": improved,
            "llm_pun_works": int(response.get("works_as_pun", 0)),
            "llm_pun_explanation": str(response.get("explanation", "")),
        }
    except Exception as e:
        return {
            "llm_pun_word": candidate.pun_word_low,
            "llm_pun_works": -1,
            "llm_pun_explanation": f"LLM verify error: {e}",
        }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main public function — drop-in replacement for get_cosine_similarity
# ─────────────────────────────────────────────────────────────────────────────
 
async def apply_low_algorithm(
    df: pd.DataFrame,
    model: str,
    start: int = 0,
    end: int = -1,
    llm_verify: bool = True,
) -> None:
    """
    For each row, run Low's polygon search on the French meaning lists
    (first_meaning_fr, second_meaning_fr) to find a French word that bridges
    both meanings of the original English pun.
 
    Optionally asks the LLM to verify / improve the polygon candidate.
 
    Saves chunked output to low_dir/{model}/{i}.tsv.
 
    Output columns added:
      pun_word_low        — best French pun word found by polygon search
      polygon_level       — 4=square, 5=pentagon, 6=hexagon, 7=heptagon, -1=literal
      similarity_score           — composite Low score
      low_path            — search path (stringified list)
      low_strategy        — square / pentagon / hexagon / heptagon / literal_fallback
      low_error           — non-empty if something went wrong
      llm_pun_word        — LLM-verified / improved word (if llm_verify=True)
      llm_pun_works       — 1/0/-1 from LLM verification
      llm_pun_explanation — LLM's reasoning
    """
    lexique = _get_lexique()
    semantic = _get_semantic()
 
    base_columns = [
        "pun_word_low",
        "polygon_level",
        "similarity_score",
        "low_path",
        "low_strategy",
        "low_error",
    ]
    llm_columns = [
        "llm_pun_word",
        "llm_pun_works",
        "llm_pun_explanation",
    ]
    output_columns = base_columns + (llm_columns if llm_verify else [])
 
    # ── synchronous polygon pass (CPU-bound, fast, no API cost) ──────────────
 
    def polygon_row(row: pd.Series) -> pd.Series:
        pun_word_fr = str(row.get("pun_word_fr", "") or "").strip()
        first_fr    = safe_list(row.get("first_meaning_fr", []))
        second_fr   = safe_list(row.get("second_meaning_fr", []))
 
        if lexique is None:
            result = LowResult(
                pun_word_low=pun_word_fr,
                polygon_level=-1,
                similarity_score=0.0,
                path=[],
                strategy="literal_fallback",
                error="Lexique not available",
            )
        else:
            try:
                result = _run_polygon_search(
                    pun_word_fr, first_fr, second_fr, lexique, semantic
                )
            except Exception as e:
                result = LowResult(
                    pun_word_low=pun_word_fr,
                    polygon_level=-1,
                    similarity_score=0.0,
                    path=[],
                    strategy="literal_fallback",
                    error=str(e),
                )
 
        log(
            row.name,
            row.get("pun_word", ""),
            "→",
            pun_word_fr,
            "→",
            result.pun_word_low,
            f"[{result.strategy}, level={result.polygon_level}, score={result.similarity_score:.3f}]",
        )
 
        return pd.Series({
            "pun_word_low":  result.pun_word_low,
            "polygon_level": result.polygon_level,
            "similarity_score": round(result.similarity_score, 2),
            "low_path":      str(result.path),
            "low_strategy":  result.strategy,
            "low_error":     result.error,
        })
 
    # ── async LLM verification pass ───────────────────────────────────────────
 
    async def llm_verify_row(row: pd.Series) -> pd.Series:
        candidate = LowResult(
            pun_word_low=str(row.get("pun_word_low", "") or ""),
            polygon_level=int(row.get("polygon_level", -1)),
            similarity_score=float(row.get("similarity_score", 0.0)),
            path=[],
            strategy=str(row.get("low_strategy", "")),
            error="",
        )
        out = await _llm_verify_candidate(row, candidate, model)
        return pd.Series(out)
 
    # ── chunked execution ─────────────────────────────────────────────────────
 
    chunk_size = 100
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    _end = end if end > 0 else len(chunks)
 
    for i in range(start, _end):
        chunk = chunks[i]
 
        # 1. Polygon search (sync, no API calls)
        chunk[base_columns] = chunk.apply(polygon_row, axis=1)
 
        # 2. LLM verification (async, optional)
        if llm_verify:
            chunk[llm_columns] = await run_async_chunk(chunk, llm_verify_row, llm_columns)
 
        save(chunk, f"{similarity_dir}{model}/{i}.tsv")
 
# ─────────────────────────────────────────────────────────────────────────────
# Legacy cosine similarity (kept for reference / ablation)
# ─────────────────────────────────────────────────────────────────────────────
 
def get_cosine_similarity(df, model, start=0, end=-1):
    """
    Original cosine similarity measurement between EN and FR embeddings.
    Kept for ablation / comparison against apply_low_algorithm results.
    """
    import torch
    from sentence_transformers import util
 
    def mean_embedding_or_zero(st_model, values):
        values = safe_list(values)
        if not values:
            dim = st_model.get_sentence_embedding_dimension()
            return torch.zeros((1, dim))
        return torch.mean(st_model.encode(values, convert_to_tensor=True), dim=0, keepdim=True)
 
    def apply(row, st_model):
        pun_word_embedding_en = st_model.encode([row["pun_word"]], convert_to_tensor=True)
        first_meaning_embedding_en = mean_embedding_or_zero(st_model, row["first_meaning"])
        second_meaning_embedding_en = mean_embedding_or_zero(st_model, row["second_meaning"])
 
        pun_word_embedding_fr = st_model.encode([row["pun_word_fr"]], convert_to_tensor=True)
        first_meaning_embedding_fr = mean_embedding_or_zero(st_model, row["first_meaning_fr"])
        second_meaning_embedding_fr = mean_embedding_or_zero(st_model, row["second_meaning_fr"])
 
        first_similarity_en = util.cos_sim(pun_word_embedding_en, first_meaning_embedding_en).item()
        second_similarity_en = util.cos_sim(pun_word_embedding_en, second_meaning_embedding_en).item()
        first_similarity_fr = util.cos_sim(pun_word_embedding_fr, first_meaning_embedding_fr).item()
        second_similarity_fr = util.cos_sim(pun_word_embedding_fr, second_meaning_embedding_fr).item()
 
        first_similarity_diff = first_similarity_en - first_similarity_fr
        second_similarity_diff = second_similarity_en - second_similarity_fr
 
        log(row.name, row["pun_word"], row["pun_word_fr"], row["pun_type"])
        log("first en", first_similarity_en, "fr", first_similarity_fr, "diff", first_similarity_diff)
        log("second en", second_similarity_en, "fr", second_similarity_fr, "diff", second_similarity_diff)
 
        result = {
            "first_similarity_en": first_similarity_en,
            "second_similarity_en": second_similarity_en,
            "first_similarity_fr": first_similarity_fr,
            "second_similarity_fr": second_similarity_fr,
            "first_similarity_diff": first_similarity_diff,
            "second_similarity_diff": second_similarity_diff,
        }
        return pd.Series(result)
 
    st_model = get_model(model)
    chunk_size = 10
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)
 
    for i in range(start, end):
        current_df = chunks[i]
        current_df[
            [
                "first_similarity_en",
                "second_similarity_en",
                "first_similarity_fr",
                "second_similarity_fr",
                "first_similarity_diff",
                "second_similarity_diff",
            ]
        ] = current_df.apply(apply, axis=1, args=(st_model,))
        save(current_df, f"{similarity_dir}{model}/{i}.tsv")
 



async def run_async_chunk(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[Any]],
    result_columns: list[str],
) -> pd.DataFrame:
    return await run_async_apply(chunk_df, apply_async_fn, result_columns)


async def identify_pun_meanings(df, model, start=0, end=-1):
    output_columns = [
        "pun_word",
        "pun_type",
        "first_meaning",
        "second_meaning",
    ]

    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pun_word": {"type": "string"},
            "pun_type": {"type": "string"},
            "first_meaning": {"type": "array", "items": {"type": "string"}},
            "second_meaning": {"type": "array", "items": {"type": "string"}},
        },
        "required": output_columns,
    }

    async def apply(row):
        text_clean = row["text_clean"]

        prompt = f"""
Text: {text_clean}

Step 1: Identify the pun word in this text. Output one word.
Step 2: Determine whether the pun is homographic or homophonic. Output either "homographic" or "homophonic".
Step 3: Give a list of synonyms for each of the two meanings of the pun. If it is a homophonic pun, include the relevant homophones in the appropriate lists.

Return only valid JSON.
"""

        log(row.name, text_clean)
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=response_schema,
                required_keys=output_columns,
                routing_preset="stable",
            )
        except Exception as e:
            response = log_and_build_fallback(
                e,
                {
                    "pun_word": "ERROR",
                    "pun_type": "",
                    "first_meaning": [],
                    "second_meaning": [],
                },
            )
        return response

    chunk_size = 100
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)

    for i in range(start, end):
        chunks[i][output_columns] = await run_async_chunk(chunks[i], apply, output_columns)
        save(chunks[i], f"{identify_dir}{model}/{i}.tsv")


async def translate_pun_meanings(df, model, start=0, end=-1, translate_flag=True):
    fr_columns = [
        "pun_word_fr",
        "first_meaning_fr",
        "second_meaning_fr",
    ]
    bt_columns = [
        "pun_word_bt",
        "first_meaning_bt",
        "second_meaning_bt",
    ]

    fr_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pun_word_fr": {"type": "string"},
            "first_meaning_fr": {"type": "array", "items": {"type": "string"}},
            "second_meaning_fr": {"type": "array", "items": {"type": "string"}},
        },
        "required": fr_columns,
    }

    bt_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pun_word_bt": {"type": "string"},
            "first_meaning_bt": {"type": "array", "items": {"type": "string"}},
            "second_meaning_bt": {"type": "array", "items": {"type": "string"}},
        },
        "required": bt_columns,
    }

    async def translate(row):
        row_dict = row.to_dict()
        payload = {
            "pun_word_fr": row_dict["pun_word"],
            "first_meaning_fr": safe_list(row_dict["first_meaning"]),
            "second_meaning_fr": safe_list(row_dict["second_meaning"]),
        }

        prompt = f"""
Translate only the VALUES of this JSON object from English to French.
Do not change the keys.
Preserve the structure exactly.
If a value is a list, translate each element.

Input JSON:
{payload}

Return only valid JSON.
"""

        log(
            row.name,
            payload["pun_word_fr"],
            payload["first_meaning_fr"],
            payload["second_meaning_fr"],
        )
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=fr_schema,
                required_keys=fr_columns,
                routing_preset="fast",
            )
        except Exception as e:
            response = log_and_build_fallback(
                e,
                {
                    "pun_word_fr": "ERROR",
                    "first_meaning_fr": [],
                    "second_meaning_fr": [],
                },
            )
        return response

    async def back_translate(row):
        payload = {
            "pun_word_bt": row["pun_word_fr"],
            "first_meaning_bt": safe_list(row["first_meaning_fr"]),
            "second_meaning_bt": safe_list(row["second_meaning_fr"]),
        }

        prompt = f"""
Translate only the VALUES of this JSON object from French to English.
Do not change the keys.
Preserve the structure exactly.
If a value is a list, translate each element.

Input JSON:
{payload}

Return only valid JSON.
"""

        log(
            row.name,
            payload["pun_word_bt"],
            payload["first_meaning_bt"],
            payload["second_meaning_bt"],
        )
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=bt_schema,
                required_keys=bt_columns,
                routing_preset="fast",
            )
        except Exception as e:
            response = log_and_build_fallback(
                e,
                {
                    "pun_word_bt": "ERROR",
                    "first_meaning_bt": [],
                    "second_meaning_bt": [],
                },
            )
        return response

    chunk_size = 100
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    end = end if end > 0 else len(chunks)

    for i in range(start, end):
        if translate_flag:
            chunks[i][fr_columns] = await run_async_chunk(chunks[i], translate, fr_columns)
            save(chunks[i], f"{translate_dir}{model}/t/{i}.tsv")

        translate_df = load(f"{translate_dir}{model}/t/{i}.tsv")
        translate_df[bt_columns] = await run_async_chunk(translate_df, back_translate, bt_columns)
        save(translate_df, f"{translate_dir}{model}/{i}.tsv")

async def check_french_homonyms(df, model, start=0, end=-1):
    output_columns = ["is_homonym", "first_meaning_overlap", "second_meaning_overlap"]

    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_homonym": {"type": "integer"},
            "first_meaning_overlap": {"type": "integer"},
            "second_meaning_overlap": {"type": "integer"},
        },
        "required": output_columns,
    }

    async def apply(row):
        pun_word_fr = row["pun_word_fr"]
        first_meaning_fr = safe_list(row["first_meaning_fr"])
        second_meaning_fr = safe_list(row["second_meaning_fr"])

        prompt = f"""
Question 1: Is the French word "{pun_word_fr}" a homonym? Output 1 for yes or 0 for no.
Question 2: Does the word "{pun_word_fr}" share at least one meaning with any word in this list: {first_meaning_fr}? Output 1 for yes or 0 for no.
Question 3: Does the word "{pun_word_fr}" share at least one meaning with any word in this list: {second_meaning_fr}? Output 1 for yes or 0 for no.

Return only valid JSON.
"""

        log(row.name, pun_word_fr, first_meaning_fr, second_meaning_fr)
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=response_schema,
                required_keys=output_columns,
                routing_preset="fast",
            )
        except Exception as e:
            response = log_and_build_fallback(
                e,
                {
                    "is_homonym": -1,
                    "first_meaning_overlap": -1,
                    "second_meaning_overlap": -1,
                },
            )
        return response

    chunk_size = 10
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    if end == -1:
        end = len(chunks)

    for i in range(start, end):
        chunks[i][output_columns] = await run_async_chunk(chunks[i], apply, output_columns)
        save(chunks[i], f"{homonym_dir}{model}/{i}.tsv")


def generate_french_puns(df):
    return True


async def main():
    task = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    translate_flag = False if len(sys.argv) > 5 else True

    if task == "identify":
        df = load(combined_en_path)
        await identify_pun_meanings(df, model, start, end)

    if task == "translate":
        df = load_all(f"{identify_dir}gemini/")
        save(df, f"{identify_dir}gemini.tsv")
        await translate_pun_meanings(df, model, start, end, translate_flag)

    if task == "lows_similarity":
        llm_verify = "--no-llm" not in sys.argv
        df = load_all(f"{translate_dir}gemini/")
        save(df, f"{translate_dir}gemini.tsv")
        await apply_low_algorithm(df, model, start, end, llm_verify=llm_verify)

    # if task == "similarity":
    #     df = load_all(f"{translate_dir}o4/t/")
    #     save(df, f"{translate_dir}o4.tsv")
    #     get_cosine_similarity(df, model, start, end)

    # if task == "homonym":
    #     df = load_all(f"{similarity_dir}bilingual/")
    #     save(df, f"{similarity_dir}bilingual.tsv")
    #     await check_french_homonyms(df, model, start, end)


if __name__ == "__main__":
    asyncio.run(main())