"""
JOKER French pun generator.

This generator follows the same infrastructure style as preprocessor.py:
  - async chunked execution
  - one OpenRouter/model call per row through get_response_async
  - strict JSON schema
  - chunked TSV output

Input: retrieval-step TSVs, usually data/processed/retrieval/{model}/{chunk}.tsv
Required source columns, inherited from identify/translate:
  text_clean, pun_word, pun_type, first_meaning, second_meaning,
  pun_word_fr, first_meaning_fr, second_meaning_fr
Optional retrieval columns:
  retrieval_affordances_json, retrieval_affordance_count
  retrieval_pack_compact, generator_affordance_pack, bridge_candidates

Usage:
  python generator.py generate google/gemini-3-pro 0 -1

Useful environment variables:
  GENERATOR_DEFAULT_MODEL       default model name
  GENERATOR_MAX_CONCURRENCY     async row concurrency, default 8
  GENERATOR_CHUNK_SIZE          output chunk size, default 100
  GENERATOR_CANDIDATE_COUNT     candidates requested per row, default 12
  GENERATOR_INPUT_DIR           explicit directory containing retrieval TSVs
  GENERATOR_VERBOSE             1/0 logging, default 1
"""

from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

from config import generate_dir
try:
    from config import retrieval_dir  # preferred: add retrieval = ... under [dir]
except Exception:
    retrieval_dir = ""

from data import load_all, save
from utils import get_response_async

pd.options.mode.chained_assignment = None

DEFAULT_MODEL = os.environ.get("GENERATOR_DEFAULT_MODEL", "google/gemini-3-pro")
VERBOSE = os.environ.get("GENERATOR_VERBOSE", "1") == "1"
MAX_CONCURRENCY = int(os.environ.get("GENERATOR_MAX_CONCURRENCY", "8"))
CHUNK_SIZE = int(os.environ.get("GENERATOR_CHUNK_SIZE", "100"))
TARGET_CANDIDATE_COUNT = int(os.environ.get("GENERATOR_CANDIDATE_COUNT", "12"))
MAX_RETRIEVAL_AFFORDANCES_IN_PROMPT = int(os.environ.get("GENERATOR_MAX_AFFORDANCES_IN_PROMPT", "6"))
MAX_FIELD_TERMS = int(os.environ.get("GENERATOR_MAX_FIELD_TERMS", "8"))

OUTPUT_COLUMNS = [
    "candidate_json",
    "candidate_count",
    "generation_error",
]

REQUIRED_INPUT_COLUMNS = [
    "text_clean",
    "pun_word",
    "pun_type",
    "first_meaning",
    "second_meaning",
    "pun_word_fr",
    "first_meaning_fr",
    "second_meaning_fr",
]

CANDIDATE_STRATEGIES = [
    "retrieval_direct",
    "retrieval_loose",
    "mechanism_preserving",
    "semantic_compensation",
    "idiom_or_expression",
    "free_native_french",
]

MECHANISMS = [
    "homophone",
    "near_homophone",
    "homograph",
    "polysemy",
    "idiom",
    "paronymy",
    "morphological",
    "compensation",
    "other",
]


def log(*args: Any) -> None:
    if VERBOSE:
        print(*args)


def norm_space(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def safe_list(x: Any) -> list[str]:
    if x is None:
        return []
    try:
        if isinstance(x, float) and pd.isna(x):
            return []
    except Exception:
        pass
    if isinstance(x, list):
        return [norm_space(v) for v in x if norm_space(v)]
    if isinstance(x, tuple):
        return [norm_space(v) for v in x if norm_space(v)]
    if isinstance(x, str):
        text = x.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            for parser in (ast.literal_eval, json.loads):
                try:
                    value = parser(text)
                    if isinstance(value, list):
                        return [norm_space(v) for v in value if norm_space(v)]
                except Exception:
                    pass
        # Conservative fallback for manually inspected or damaged rows.
        parts = re.split(r"\s*[;,]\s*", text)
        if len(parts) > 1:
            return [norm_space(p) for p in parts if norm_space(p)]
    return []


def safe_json_loads(x: Any) -> Any:
    if x is None:
        return None
    try:
        if isinstance(x, float) and pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (dict, list)):
        return x
    text = str(x).strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return None


def unique_keep_order(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = norm_space(value)
        key = value.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return out


def log_and_build_fallback(error: Exception, payload: dict[str, Any]) -> pd.Series:
    print(f"Error: {error}")
    return pd.Series(payload)


async def run_async_apply(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[Any]],
    result_columns: list[str],
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(index: Any, row: pd.Series):
        async with semaphore:
            result = await apply_async_fn(row)
            return index, result

    tasks = [asyncio.create_task(worker(index, row)) for index, row in chunk_df.iterrows()]
    results: dict[Any, Any] = {}

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


async def run_async_chunk(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[Any]],
    result_columns: list[str],
) -> pd.DataFrame:
    return await run_async_apply(chunk_df, apply_async_fn, result_columns)


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required generator input columns: " + ", ".join(missing))


def compact_affordance(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    left = norm_space(raw.get("left") or raw.get("source_surface") or raw.get("a_surface") or raw.get("source") or raw.get("left_text"))
    right = norm_space(raw.get("right") or raw.get("candidate_surface") or raw.get("b_surface") or raw.get("candidate") or raw.get("right_text"))
    relation = norm_space(raw.get("relation") or raw.get("bridge_type") or raw.get("phonetic_relation") or "sound_or_meaning_bridge")
    bucket = norm_space(raw.get("retrieval_bucket") or raw.get("affordance_bucket") or "")
    lane = norm_space(raw.get("export_lane") or raw.get("strategy") or "")

    scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    phonetic = raw.get("phonetic_score", scores.get("phonetic_match", ""))
    naturalness = raw.get("naturalness_score", scores.get("french_naturalness", ""))
    usability = raw.get("pivotability_score", scores.get("pun_pivot_usability", ""))
    overall = raw.get("llm_priority_score", raw.get("bridge_score", scores.get("overall_score", "")))

    why_bits: list[str] = []
    if left and right:
        why_bits.append(f"{left!r} and {right!r} are a possible French sound/meaning collision")
    if relation:
        why_bits.append(f"relation={relation}")
    if bucket:
        why_bits.append(f"bucket={bucket}")

    return {
        "id": idx,
        "left": left,
        "right": right,
        "relation": relation,
        "retrieval_bucket": bucket,
        "export_lane": lane,
        "phonetic_score": _round_or_blank(phonetic),
        "naturalness_score": _round_or_blank(naturalness),
        "pivot_usability_score": _round_or_blank(usability),
        "overall_score": _round_or_blank(overall),
        "why_interesting": "; ".join(why_bits),
    }


def _round_or_blank(x: Any) -> float | str:
    try:
        return round(float(x), 4)
    except Exception:
        return ""


def parse_retrieval_affordances(row: pd.Series) -> list[dict[str, Any]]:
    """Return compact generator-facing affordances from current or older retrieval outputs."""
    raw: Any = None

    # Current compact retrieval output.
    if "retrieval_affordances_json" in row:
        raw = safe_json_loads(row.get("retrieval_affordances_json"))

    # Older/full retrieval-pack shapes, kept for compatibility.
    if not raw:
        for col in ("generator_affordance_pack", "retrieval_pack_compact", "retrieval_pack"):
            if col not in row:
                continue
            pack = safe_json_loads(row.get(col))
            if not isinstance(pack, dict):
                continue
            if isinstance(pack.get("top_bridge_candidates"), list):
                raw = pack.get("top_bridge_candidates")
                break
            if isinstance(pack.get("bridge_candidates"), list):
                raw = pack.get("bridge_candidates")
                break
            gen = pack.get("generator_affordance_pack")
            if isinstance(gen, dict) and isinstance(gen.get("top_bridge_candidates"), list):
                raw = gen.get("top_bridge_candidates")
                break

    if not isinstance(raw, list):
        return []

    compact: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        c = compact_affordance(item, len(compact) + 1)
        key = (c["left"].lower(), c["right"].lower(), c["relation"].lower())
        if not c["left"] and not c["right"]:
            continue
        if key in seen:
            continue
        seen.add(key)
        compact.append(c)
        if len(compact) >= MAX_RETRIEVAL_AFFORDANCES_IN_PROMPT:
            break
    return compact


def build_humor_card(row: pd.Series) -> dict[str, Any]:
    return {
        "english_sentence": norm_space(row.get("text_clean", "")),
        "pun_word_or_trigger": norm_space(row.get("pun_word", "")),
        "pun_type": norm_space(row.get("pun_type", "")),
        "english_meaning_A": unique_keep_order(safe_list(row.get("first_meaning", [])), MAX_FIELD_TERMS),
        "english_meaning_B": unique_keep_order(safe_list(row.get("second_meaning", [])), MAX_FIELD_TERMS),
        "direct_french_pun_word_hint": norm_space(row.get("pun_word_fr", "")),
        "french_semantic_field_A": unique_keep_order(safe_list(row.get("first_meaning_fr", [])), MAX_FIELD_TERMS),
        "french_semantic_field_B": unique_keep_order(safe_list(row.get("second_meaning_fr", [])), MAX_FIELD_TERMS),
    }


def build_generation_prompt(row: pd.Series, candidate_count: int = TARGET_CANDIDATE_COUNT) -> str:
    humor_card = build_humor_card(row)
    affordances = parse_retrieval_affordances(row)
    payload = {
        "humor_card": humor_card,
        "retrieval_affordances": affordances,
        "candidate_count": candidate_count,
    }

    return f"""
You are an expert native French comedy writer specializing in puns, wordplay, and humorous adaptation.

Your task is NOT literal translation.
Your task is to recreate the humorous effect of the English joke as strong, natural French wordplay.

Primary objective:
Produce genuinely funny French pun candidates that a native French speaker can understand and enjoy.

Priority order:
1. The French sentence is funny and natural.
2. The wordplay is obvious or quickly recoverable.
3. The result is a successful French pun, not merely a paraphrase.
4. Preserve the original wordplay mechanism when it helps.
5. Preserve a related semantic field when it helps.
6. Preserve original imagery, grammar, or literal wording only if it improves the joke.

Important:
- A literal French sentence that is not funny is a failed candidate.
- You may freely change imagery, objects, setting, and exact meaning to make the joke work.
- You may use compensation: the French pun may move elsewhere in the sentence.
- Retrieval affordances are optional creative ingredients, not constraints.
- Use retrieval affordances heavily when they naturally produce a good French pun.
- You may reuse one strong affordance for multiple different jokes.
- You may ignore weak affordances.
- You do not need to cover every affordance.
- Do not make near-duplicates.
- Avoid obscure vocabulary, academic explanations, and English-sounding French.

Generate exactly {candidate_count} distinct French pun candidates.
Use adaptive creative allocation: choose whichever strategies actually produce the best jokes.
Possible strategies include retrieval_direct, retrieval_loose, mechanism_preserving, semantic_compensation, idiom_or_expression, and free_native_french.
These are labels for analysis only; do not force equal numbers of each.

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

For every candidate:
- french: the French joke sentence only
- pun_trigger: the French word or phrase carrying the joke
- mechanism: one of {MECHANISMS}
- strategy: one of {CANDIDATE_STRATEGIES}
- used_retrieval: true if any retrieval affordance materially influenced the candidate
- used_affordance_ids: list of retrieval affordance ids used; empty if none
- semantic_relation: same_field, loose_theme, or free
- risk: low, medium, or high
- why_it_works: brief English explanation of the French wordplay

Return only valid JSON. Do not include markdown. Do not include commentary outside JSON.
""".strip()


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "french": {"type": "string"},
                    "pun_trigger": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "strategy": {"type": "string"},
                    "used_retrieval": {"type": "boolean"},
                    "used_affordance_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "semantic_relation": {"type": "string"},
                    "risk": {"type": "string"},
                    "why_it_works": {"type": "string"},
                },
                "required": [
                    "french",
                    "pun_trigger",
                    "mechanism",
                    "strategy",
                    "used_retrieval",
                    "used_affordance_ids",
                    "semantic_relation",
                    "risk",
                    "why_it_works",
                ],
            },
        }
    },
    "required": ["candidates"],
}


def normalize_candidate(c: dict[str, Any], model: str) -> dict[str, Any]:
    french = norm_space(c.get("french", ""))
    mechanism = norm_space(c.get("mechanism", "other")).lower()
    strategy = norm_space(c.get("strategy", "free_native_french")).lower()
    semantic_relation = norm_space(c.get("semantic_relation", "free")).lower()
    risk = norm_space(c.get("risk", "medium")).lower()

    if mechanism not in MECHANISMS:
        mechanism = "other"
    if strategy not in CANDIDATE_STRATEGIES:
        strategy = "free_native_french"
    if semantic_relation not in {"same_field", "loose_theme", "free"}:
        semantic_relation = "free"
    if risk not in {"low", "medium", "high"}:
        risk = "medium"

    used_ids = c.get("used_affordance_ids", [])
    if not isinstance(used_ids, list):
        used_ids = []
    clean_used_ids: list[int] = []
    for value in used_ids:
        try:
            clean_used_ids.append(int(value))
        except Exception:
            pass

    return {
        "french": french,
        "pun_trigger": norm_space(c.get("pun_trigger", "")),
        "mechanism": mechanism,
        "strategy": strategy,
        "used_retrieval": bool(c.get("used_retrieval", False)) or bool(clean_used_ids),
        "used_affordance_ids": sorted(set(clean_used_ids)),
        "semantic_relation": semantic_relation,
        "risk": risk,
        "why_it_works": norm_space(c.get("why_it_works", "")),
        "generator_model": model,
    }


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in candidates:
        french = norm_space(c.get("french", ""))
        if not french:
            continue
        key = re.sub(r"\s+", " ", french.casefold())
        if key in seen:
            continue
        seen.add(key)
        c["french"] = french
        out.append(c)
    return out


async def generate_row(row: pd.Series, model: str) -> pd.Series:
    row_id = row.get("id_en", row.name)
    try:
        prompt = build_generation_prompt(row, TARGET_CANDIDATE_COUNT)
        response = await get_response_async(
            prompt,
            model,
            response_schema=RESPONSE_SCHEMA,
            required_keys=["candidates"],
            routing_preset="stable",
        )
        raw_candidates = response.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        candidates = dedupe_candidates([
            normalize_candidate(c, model)
            for c in raw_candidates
            if isinstance(c, dict)
        ])
        log(row.name, row_id, f"generated={len(candidates)}")
        return pd.Series({
            "candidate_json": json.dumps(candidates, ensure_ascii=False),
            "candidate_count": len(candidates),
            "generation_error": "",
        })
    except Exception as e:
        return log_and_build_fallback(
            e,
            {
                "candidate_json": "[]",
                "candidate_count": 0,
                "generation_error": str(e),
            },
        )


async def generate_french_puns(df: pd.DataFrame, model: str, start: int = 0, end: int = -1) -> None:
    validate_input(df)
    chunks = [df.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
    end = end if end > 0 else len(chunks)

    for i in range(start, end):
        chunk = chunks[i].copy()
        chunk[OUTPUT_COLUMNS] = await run_async_chunk(
            chunk,
            lambda row: generate_row(row, model),
            OUTPUT_COLUMNS,
        )
        save(chunk, f"{generate_dir}{model}/{i}.tsv")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_input_dirs(model: str) -> list[str]:
    """Candidate retrieval input dirs, ordered from most explicit to most conventional."""
    dirs: list[str] = []
    env_dir = os.environ.get("GENERATOR_INPUT_DIR", "").strip()
    if env_dir:
        dirs.append(env_dir)

    model_variants = unique_keep_order([model, model.replace("/", "__"), model.split("/")[-1]])
    base_dirs: list[str] = []
    if retrieval_dir:
        base_dirs.append(str(retrieval_dir))
    base_dirs.append(str(_repo_root() / "data" / "processed" / "retrieval"))

    for base in base_dirs:
        for variant in model_variants:
            dirs.append(str(Path(base) / variant))

    return unique_keep_order(dirs)


def load_generator_input(model: str) -> pd.DataFrame:
    tried: list[str] = []
    for path in _candidate_input_dirs(model):
        tried.append(path)
        p = Path(path)
        if not p.exists() or not p.is_dir():
            continue
        try:
            df = load_all(str(p) + os.sep)
            if len(df) > 0:
                log("Loading generator input:", p)
                return df
        except Exception as e:
            log("Skipping generator input", p, e)
    raise FileNotFoundError("Could not find generator input retrieval TSVs. Tried: " + " | ".join(tried))


async def main() -> None:
    task = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    if task == "generate":
        df = load_generator_input(model)
        await generate_french_puns(df, model, start, end)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    asyncio.run(main())
