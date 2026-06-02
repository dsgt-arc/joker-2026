"""
JOKER French pun generator v2.

Infrastructure follows preprocessor.py:
  - async chunked execution
  - one OpenRouter/model call per row through get_response_async
  - strict JSON schema
  - chunked TSV output

Input: retrieval-step TSVs, usually data/processed/retrieval/{model}/{chunk}.tsv
Required columns inherited from identify/translate:
  text_clean, pun_word, pun_type, first_meaning, second_meaning,
  pun_word_fr, first_meaning_fr, second_meaning_fr
Optional retrieval columns:
  retrieval_affordances_json, retrieval_affordance_count,
  retrieval_pack_compact, generator_affordance_pack, bridge_candidates

Usage:
  python generator_v2.py generate gemini 0 1
  python generator_v2.py generate google/gemini-3-pro 0 -1

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
    from config import retrieval_dir
except Exception:
    retrieval_dir = ""

from data import load_all, save
from utils import get_response_async

pd.options.mode.chained_assignment = None

GENERATOR_VERSION = "v13"
DEFAULT_MODEL = os.environ.get("GENERATOR_DEFAULT_MODEL", os.environ.get("GENERATOR_MODEL", "google/gemini-3-pro"))
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
    "generator_version",
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

ALLOWED_STRATEGIES = [
    "retrieval_direct",
    "retrieval_loose",
    "mechanism_preserving",
    "semantic_compensation",
    "idiom_or_expression",
    "free_native_french",
]

ALLOWED_MECHANISMS = [
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


def _score_for_sort(raw: dict[str, Any]) -> float:
    scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    for key in ("llm_priority_score", "overall_score", "bridge_score"):
        if key in raw:
            try:
                return float(raw.get(key) or 0.0)
            except Exception:
                pass
    if "overall_score" in scores:
        try:
            return float(scores.get("overall_score") or 0.0)
        except Exception:
            pass
    return 0.0


def compact_affordance(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    """Convert noisy retrieval records into small creative ingredients.

    Deliberately removes retrieval_bucket, retrieval_bucket_rank, and export_lane.
    These are retrieval diagnostics, not useful creative prompt content.
    """
    left = norm_space(
        raw.get("left")
        or raw.get("source_surface")
        or raw.get("a_surface")
        or raw.get("source")
        or raw.get("left_text")
    )
    right = norm_space(
        raw.get("right")
        or raw.get("candidate_surface")
        or raw.get("b_surface")
        or raw.get("candidate")
        or raw.get("right_text")
    )
    relation = norm_space(
        raw.get("relation")
        or raw.get("bridge_type")
        or raw.get("phonetic_relation")
        or "possible sound/meaning collision"
    )

    scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    phonetic = raw.get("phonetic_score", scores.get("phonetic_match", ""))
    usability = raw.get("pivotability_score", scores.get("pun_pivot_usability", ""))

    out: dict[str, Any] = {
        "id": idx,
        "left": left,
        "right": right,
        "relation": relation,
    }
    try:
        if phonetic != "":
            out["phonetic_score"] = round(float(phonetic), 4)
    except Exception:
        pass
    try:
        if usability != "":
            out["pivotability_score"] = round(float(usability), 4)
    except Exception:
        pass

    hint_bits: list[str] = []
    if left and right:
        hint_bits.append(f"{left} ↔ {right}")
    if relation:
        hint_bits.append(relation)
    out["creative_hint"] = "; ".join(hint_bits)

    # Explicitly do not copy retrieval_bucket, retrieval_bucket_rank, export_lane.
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def parse_retrieval_affordances(row: pd.Series) -> list[dict[str, Any]]:
    """Read affordances from retrieval outputs and return prompt-safe records."""
    raw_items: list[dict[str, Any]] = []

    direct = safe_json_loads(row.get("retrieval_affordances_json", ""))
    if isinstance(direct, list):
        raw_items.extend([x for x in direct if isinstance(x, dict)])

    for col in ("generator_affordance_pack", "retrieval_pack_compact"):
        value = safe_json_loads(row.get(col, ""))
        if isinstance(value, dict):
            top = value.get("top_bridge_candidates")
            if isinstance(top, list):
                raw_items.extend([x for x in top if isinstance(x, dict)])
            nested = value.get("generator_affordance_pack")
            if isinstance(nested, dict) and isinstance(nested.get("top_bridge_candidates"), list):
                raw_items.extend([x for x in nested["top_bridge_candidates"] if isinstance(x, dict)])

    bridge_candidates = safe_json_loads(row.get("bridge_candidates", ""))
    if isinstance(bridge_candidates, list):
        raw_items.extend([x for x in bridge_candidates if isinstance(x, dict)])

    # Dedupe by left/right/relation, prefer higher retrieval score.
    raw_items.sort(key=_score_for_sort, reverse=True)
    seen: set[tuple[str, str, str]] = set()
    compact: list[dict[str, Any]] = []
    for raw in raw_items:
        item = compact_affordance(raw, len(compact) + 1)
        left = norm_space(item.get("left", ""))
        right = norm_space(item.get("right", ""))
        relation = norm_space(item.get("relation", ""))
        if not left and not right:
            continue
        key = (left.lower(), right.lower(), relation.lower())
        if key in seen:
            continue
        seen.add(key)
        item["id"] = len(compact) + 1
        compact.append(item)
        if len(compact) >= MAX_RETRIEVAL_AFFORDANCES_IN_PROMPT:
            break
    return compact


def build_humor_card(row: pd.Series) -> dict[str, Any]:
    return {
        "english_sentence": norm_space(row.get("text_clean", "")),
        "english_pun_trigger": norm_space(row.get("pun_word", "")),
        "english_pun_type": norm_space(row.get("pun_type", "")),
        "english_meaning_A": unique_keep_order(safe_list(row.get("first_meaning", [])), MAX_FIELD_TERMS),
        "english_meaning_B": unique_keep_order(safe_list(row.get("second_meaning", [])), MAX_FIELD_TERMS),
        "direct_french_pun_word_hint": norm_space(row.get("pun_word_fr", "")),
        "french_semantic_field_A": unique_keep_order(safe_list(row.get("first_meaning_fr", [])), MAX_FIELD_TERMS),
        "french_semantic_field_B": unique_keep_order(safe_list(row.get("second_meaning_fr", [])), MAX_FIELD_TERMS),
    }


def build_generation_prompt(row: pd.Series, n: int) -> str:
    card = build_humor_card(row)
    affordances = parse_retrieval_affordances(row)
    payload = {
        "humor_card": card,
        "retrieval_affordances": affordances,
    }

    return f"""
You are an expert native French comedy writer specializing in puns, wordplay, idioms, and humorous adaptation.

Your task is NOT literal translation.
Your task is to recreate the humorous effect as strong, natural French wordplay for native French speakers.

Generate exactly 12 different French pun candidates that take genuinely different comedic angles.

Priority order:
1. Genuinely funny and natural in French.
2. Clear recoverable wordplay that creates a joke: lexical ambiguity, homophony, near-homophony, idiom reinterpretation, phrase reinterpretation, morphology, or compensation.
3. Similar comedic mechanism when possible; prefer stronger French wordplay over literal preservation.
4. Related semantic field when possible.
5. Original imagery/wording only when it helps the joke.

Important quality rules:
- A literal translation that is not funny is a failed candidate.
- A poetic metaphor without clear wordplay is a weak candidate. Avoid generic metaphor, symbolism, or dramatic phrasing unless there is an actual pun.
- Prefer obvious, accessible French wordplay, existing French expressions, and natural wording over obscure vocabulary or invented forms.
- Retrieval affordances are optional creative ingredients. Use them heavily if they naturally lead to funny French wordplay. Ignore weak affordances.
- You may reuse one strong affordance for multiple different jokes. You do not need to cover every affordance.
- Do not mention or explain the joke in the French sentence itself.

Input JSON:
{json.dumps(payload, ensure_ascii=False)}

Return ONLY valid minified JSON with this exact shape and no extra text:
{{"candidates":[{{"french":"...","pun_trigger":"...","mechanism":"homophone|near_homophone|homograph|polysemy|idiom|paronymy|morphological|compensation|other","strategy":"retrieval_direct|retrieval_loose|mechanism_preserving|semantic_compensation|idiom_or_expression|free_native_french","used_affordance_ids":[1],"semantic_relation":"same_field|loose_theme|free","risk":"low|medium|high"}}]}}
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
                    "used_affordance_ids": {"type": "array", "items": {"type": "integer"}},
                    "semantic_relation": {"type": "string"},
                    "risk": {"type": "string"},
                },
                "required": [
                    "french",
                    "pun_trigger",
                    "mechanism",
                    "strategy",
                    "used_affordance_ids",
                    "semantic_relation",
                    "risk",
                ],
            },
        }
    },
    "required": ["candidates"],
}


def normalize_candidate(c: dict[str, Any], affordance_count: int, model: str) -> dict[str, Any] | None:
    french = norm_space(c.get("french", ""))
    if not french:
        return None

    mechanism = norm_space(c.get("mechanism", "other")) or "other"
    if mechanism not in ALLOWED_MECHANISMS:
        mechanism = "other"

    strategy = norm_space(c.get("strategy", "free_native_french")) or "free_native_french"
    if strategy not in ALLOWED_STRATEGIES:
        strategy = "free_native_french"

    semantic_relation = norm_space(c.get("semantic_relation", "free")) or "free"
    if semantic_relation not in {"same_field", "loose_theme", "free"}:
        semantic_relation = "free"

    risk = norm_space(c.get("risk", "medium")) or "medium"
    if risk not in {"low", "medium", "high"}:
        risk = "medium"

    ids: list[int] = []
    raw_ids = c.get("used_affordance_ids", [])
    if isinstance(raw_ids, list):
        for value in raw_ids:
            try:
                ivalue = int(value)
                if 1 <= ivalue <= affordance_count and ivalue not in ids:
                    ids.append(ivalue)
            except Exception:
                pass

    return {
        "french": french,
        "pun_trigger": norm_space(c.get("pun_trigger", "")),
        "mechanism": mechanism,
        "strategy": strategy,
        "used_affordance_ids": ids,
        "semantic_relation": semantic_relation,
        "risk": risk,
        "generator_model": model,
        "generator_version": GENERATOR_VERSION,
    }


def dedupe_candidates(candidates: list[dict[str, Any]], limit: int = TARGET_CANDIDATE_COUNT) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in candidates:
        french = norm_space(c.get("french", ""))
        key = french.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        c["french"] = french
        out.append(c)
        if len(out) >= limit:
            break
    return out


async def generate_row(row: pd.Series, model: str) -> pd.Series:
    row_id = row.get("id_en", row.name)
    affordance_count = len(parse_retrieval_affordances(row))
    prompt = build_generation_prompt(row, TARGET_CANDIDATE_COUNT)

    try:
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

        candidates: list[dict[str, Any]] = []
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                continue
            normalized = normalize_candidate(raw, affordance_count, model)
            if normalized is not None:
                candidates.append(normalized)
        candidates = dedupe_candidates(candidates, TARGET_CANDIDATE_COUNT)
        error = ""
    except Exception as e:
        print(f"Error: {e}")
        candidates = []
        error = str(e)

    log(row.name, row_id, f"generated={len(candidates)}", f"affordances={affordance_count}", f"error={bool(error)}")
    return pd.Series({
        "candidate_json": json.dumps(candidates, ensure_ascii=False, separators=(",", ":")),
        "candidate_count": len(candidates),
        "generation_error": error,
        "generator_version": GENERATOR_VERSION,
    })


async def generate_french_puns(df: pd.DataFrame, model: str, start: int = 0, end: int = -1) -> None:
    validate_input(df)
    chunks = [df.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
    end = len(chunks) if end == -1 else end

    for i in range(start, end):
        chunk = chunks[i].copy()
        chunk[OUTPUT_COLUMNS] = await run_async_chunk(
            chunk,
            lambda row: generate_row(row, model),
            OUTPUT_COLUMNS,
        )
        out_path = f"{generate_dir}{model}/candidates_{GENERATOR_VERSION}/{i}.tsv"
        save(chunk, out_path)


def resolve_generator_input_dir(model: str) -> str:
    explicit = os.environ.get("GENERATOR_INPUT_DIR", "").strip()
    if explicit:
        return explicit.rstrip("/") + "/"
    if retrieval_dir:
        return f"{retrieval_dir}{model}/"
    root = Path(__file__).resolve().parents[1]
    return str(root / "data" / "processed" / "retrieval" / model) + "/"


async def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Usage: python generator_v2.py generate <model> <start> <end>")

    task = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    if task == "generate":
        input_dir = resolve_generator_input_dir(model)
        log("Loading generator input:", input_dir.rstrip("/"))
        df = load_all(input_dir)
        save(df, f"{input_dir.rstrip('/')}.tsv")
        await generate_french_puns(df, model, start, end)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    asyncio.run(main())
