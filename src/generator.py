"""
JOKER French pun generator v15.

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
  python generator_v15.py generate gemini 0 1
  python generator_v15.py generate google/gemini-3-pro 0 -1

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

from config import GENERATOR_MODEL_ALIASES, MODEL_ALIASES, generate_dir
try:
    from config import retrieval_dir
except Exception:
    retrieval_dir = ""

from data import load_all, save
from utils import get_response_async

pd.options.mode.chained_assignment = None

GENERATOR_VERSION = "v15"
DEFAULT_MODEL = os.environ.get("GENERATOR_DEFAULT_MODEL", "claude")
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


def _clean_prompt_value(value: Any) -> Any:
    """Keep affordance values useful for generation while avoiding prompt junk."""
    if value is None:
        return None

    if isinstance(value, float):
        return round(value, 4)

    if isinstance(value, (int, bool)):
        return value

    if isinstance(value, str):
        value = norm_space(value)
        return value or None

    if isinstance(value, list):
        cleaned = [_clean_prompt_value(v) for v in value]
        cleaned = [v for v in cleaned if v not in (None, "", [])]
        return cleaned[:8] or None

    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for k, v in value.items():
            if k in AFFORDANCE_DROP_KEYS:
                continue
            cv = _clean_prompt_value(v)
            if cv not in (None, "", []):
                cleaned_dict[str(k)] = cv
        return cleaned_dict or None

    return None


AFFORDANCE_DROP_KEYS = {
    "retrieval_bucket",
    "retrieval_bucket_rank",
    "export_lane",
}


def compact_affordance(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    """Convert retrieval records into compact French pun ingredients.

    Keep lexical pivots and useful scores. Remove only retrieval bookkeeping fields:
    retrieval_bucket, retrieval_bucket_rank, and export_lane.
    """
    left = norm_space(
        raw.get("left")
        or raw.get("pivot_a")
        or raw.get("source_surface")
        or raw.get("a_surface")
        or raw.get("source")
        or raw.get("left_text")
        or raw.get("a")
    )
    right = norm_space(
        raw.get("right")
        or raw.get("pivot_b")
        or raw.get("candidate_surface")
        or raw.get("b_surface")
        or raw.get("candidate")
        or raw.get("right_text")
        or raw.get("b")
    )
    relation = norm_space(
        raw.get("relation")
        or raw.get("bridge_type")
        or raw.get("phonetic_relation")
        or raw.get("match_type")
    )

    out: dict[str, Any] = {"id": idx}

    if left:
        out["left"] = left
    if right:
        out["right"] = right
    if relation:
        out["relation"] = relation

    # Preserve every useful non-bookkeeping field, including nested scores.
    for key, value in raw.items():
        if key in AFFORDANCE_DROP_KEYS:
            continue
        if key in {"left", "right", "pivot_a", "pivot_b", "relation"}:
            continue

        cleaned = _clean_prompt_value(value)
        if cleaned not in (None, "", []):
            out[key] = cleaned

    # Normalize common score aliases at top level when present.
    scores = out.get("scores") if isinstance(out.get("scores"), dict) else {}
    score_aliases = {
        "phonetic_score": raw.get("phonetic_score", scores.get("phonetic_match")),
        "semantic_score": raw.get("semantic_score", scores.get("semantic_domain_similarity")),
        "priority_score": raw.get("priority_score", raw.get("overall_score", scores.get("overall_score"))),
        "pivotability_score": raw.get("pivotability_score", scores.get("pun_pivot_usability")),
    }
    for key, value in score_aliases.items():
        cleaned = _clean_prompt_value(value)
        if cleaned not in (None, "", []):
            out[key] = cleaned

    hint_bits: list[str] = []
    if left and right:
        hint_bits.append(f"{left} ↔ {right}")
    elif left:
        hint_bits.append(left)
    elif right:
        hint_bits.append(right)
    if relation:
        hint_bits.append(relation)
    if hint_bits:
        out["creative_hint"] = "; ".join(hint_bits)

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


def build_generation_prompt(row: pd.Series, n: int) -> str:
    text_clean = norm_space(row.get("text_clean", ""))
    pun_word = norm_space(row.get("pun_word", ""))
    first_meaning_fr = unique_keep_order(safe_list(row.get("first_meaning_fr", [])), MAX_FIELD_TERMS)
    second_meaning_fr = unique_keep_order(safe_list(row.get("second_meaning_fr", [])), MAX_FIELD_TERMS)
    affordances = parse_retrieval_affordances(row)

    schema = """
{"candidates":[{"french":"...","pun_trigger":"...","mechanism":"homophone|near_homophone|homograph|polysemy|idiom|paronymy|morphological|compensation|other","strategy":"retrieval_direct|retrieval_loose|mechanism_preserving|semantic_compensation|idiom_or_expression|free_native_french","used_affordance_ids":[1],"semantic_relation":"same_field|loose_theme|free","risk":"low|medium|high"}]}
""".strip()

    return f"""
You are an expert French comedy writer specializing in puns, wordplay, idioms, and humorous adaptation. Write exactly {n} genuinely funny French puns.

Use this English pun as inspiration: {text_clean}
English pun word: {pun_word}

Relevant French semantic fields:
A. {first_meaning_fr}
B. {second_meaning_fr}

Priority order:
1. Genuinely funny to native French speakers.
2. Clear, obvious wordplay. A native French speaker should immediately identify the pun mechanism without explanation.
3. Original semantic fields only when they help the joke.
4. Similar comedic form to the English only when possible.

Generate candidates using multiple routes:
- direct ambiguity or double meaning
- homophony or near-homophony
- idiom reinterpretation
- collision of distant semantic domains
- a surprising reinterpretation that produces an immediate "aha" moment
- retrieval-affordance-inspired wordplay

Requirements:
- Favor unexpected, memorable pun pivots over safe semantic associations.
- Do not be constrained by the English wording when a stronger French pun is available.
- Every candidate must contain a clear linguistic wordplay mechanism. A joke based only on thematic association is a failed candidate.
- Do not invent fake French words.
- Actively explore each affordance as an alternative search direction. Generate at least one candidate per affordance, as long as the result is recognizable French wordplay.
- Keep the jokes compact and punchy.

French affordances:
{json.dumps(affordances, ensure_ascii=False)}

Return exactly one minified JSON object and nothing else:
{schema}
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


def _allowed_value(value: Any, allowed: list[str], default: str) -> str:
    value = norm_space(value).lower()
    return value if value in allowed else default


def normalize_candidate(c: dict[str, Any], affordance_count: int, model_alias: str, model_id: str) -> dict[str, Any] | None:
    french = norm_space(c.get("french", ""))
    if not french:
        return None

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

    semantic_relation = _allowed_value(
        c.get("semantic_relation", ""),
        ["same_field", "loose_theme", "free"],
        "free",
    )
    risk = _allowed_value(c.get("risk", ""), ["low", "medium", "high"], "medium")

    return {
        "french": french,
        "pun_trigger": norm_space(c.get("pun_trigger", "")),
        "mechanism": _allowed_value(c.get("mechanism", ""), ALLOWED_MECHANISMS, "other"),
        "strategy": _allowed_value(c.get("strategy", ""), ALLOWED_STRATEGIES, "free_native_french"),
        "used_affordance_ids": ids,
        "semantic_relation": semantic_relation,
        "risk": risk,
        "generator_model": model_alias,
        "generator_model_id": model_id,
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


async def generate_row(row: pd.Series, model_alias: str, model_id: str) -> pd.Series:
    row_id = row.get("id_en", row.name)
    affordance_count = len(parse_retrieval_affordances(row))
    prompt = build_generation_prompt(row, TARGET_CANDIDATE_COUNT)

    try:
        response = await get_response_async(
            prompt,
            model_alias,
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
            normalized = normalize_candidate(raw, affordance_count, model_alias, model_id)
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


async def generate_french_puns(df: pd.DataFrame, model_alias: str, model_id: str, start: int = 0, end: int = -1) -> None:
    validate_input(df)
    chunks = [df.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
    end = len(chunks) if end == -1 else end

    for i in range(start, end):
        chunk = chunks[i].copy()
        chunk[OUTPUT_COLUMNS] = await run_async_chunk(
            chunk,
            lambda row: generate_row(row, model_alias, model_id),
            OUTPUT_COLUMNS,
        )
        out_path = f"{generate_dir}{model_alias}/{i}.tsv"
        save(chunk, out_path)



def resolve_generator_input_dir() -> str:
    """Always load generator input from retrieval/gemini/."""
    explicit = os.environ.get("GENERATOR_INPUT_DIR", "").strip()
    if explicit:
        return explicit.rstrip("/") + "/"

    if retrieval_dir:
        return f"{retrieval_dir}gemini/"

    root = Path(__file__).resolve().parents[1]
    return str(root / "data" / "processed" / "retrieval" / "gemini") + "/"


def resolve_model_alias(model_arg: str) -> tuple[str, str]:
    model_arg = norm_space(model_arg)
    if model_arg in MODEL_ALIASES and MODEL_ALIASES.get(model_arg):
        return model_arg, MODEL_ALIASES[model_arg]

    filesystem_alias = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_arg).strip("_")
    return filesystem_alias or "model", model_arg


async def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("""Usage:
  python generator.py generate <model_alias> <start> <end>
  python generator.py ensemble <start> <end>

Examples:
  python generator.py generate gemini 0 1
  python generator.py generate gemini_pro 0 1
  python generator.py generate claude 0 -1
  python generator.py ensemble 0 -1""")

    task = sys.argv[1]

    if task == "generate":
        model_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
        model_alias, model_id = resolve_model_alias(model_arg)

        start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        end = int(sys.argv[4]) if len(sys.argv) > 4 else -1

        input_dir = resolve_generator_input_dir()
        log("Generator input: gemini retrieval")
        log("Generator alias:", model_alias)
        log("OpenRouter model:", model_id)
        log("Loading generator input:", input_dir.rstrip("/"))

        df = load_all(input_dir)
        save(df, f"{input_dir.rstrip('/')}.tsv")

        await generate_french_puns(df, model_alias, model_id, start, end)

    elif task == "ensemble":
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        end = int(sys.argv[3]) if len(sys.argv) > 3 else -1

        input_dir = resolve_generator_input_dir()
        log("Generator input: gemini retrieval")
        log("Loading generator input:", input_dir.rstrip("/"))

        df = load_all(input_dir)
        save(df, f"{input_dir.rstrip('/')}.tsv")

        for model_alias in GENERATOR_MODEL_ALIASES:
            model_id = MODEL_ALIASES.get(model_alias, "")
            if not model_id:
                log("Skipping missing model alias:", model_alias)
                continue

            log("Generating with alias:", model_alias)
            log("OpenRouter model:", model_id)

            await generate_french_puns(df, model_alias, model_id, start, end)

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    asyncio.run(main())
