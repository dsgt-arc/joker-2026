"""
Quality-first French pun generator for JOKER-style wordplay adaptation.

Input: output TSV from preprocessor.py translate step.
Requires columns:
  text_clean, pun_word, pun_type, first_meaning, second_meaning,
  first_meaning_fr, second_meaning_fr
Optional:
  pun_word_fr, id_en, is_pun

Usage:
  python generator.py generate google/gemini-3-pro 0 -1

Saves chunks to: generate_dir/{model}/candidates/{chunk}.tsv
"""

import asyncio
import ast
import json
import os
import re
import sys
from typing import Any, Awaitable, Callable

import pandas as pd

from config import generate_dir, translate_dir
from data import load_all, save
from utils import get_response_async

DEFAULT_MODEL = os.environ.get("GENERATOR_MODEL", "google/gemini-3-pro")
MAX_CONCURRENCY = int(os.environ.get("GENERATOR_MAX_CONCURRENCY", "8"))
CHUNK_SIZE = int(os.environ.get("GENERATOR_CHUNK_SIZE", "100"))
CANDIDATES_PER_MODE = int(os.environ.get("GENERATOR_CANDIDATES_PER_MODE", "3"))
VERBOSE = os.environ.get("GENERATOR_VERBOSE", "1") == "1"

GENERATION_MODES = [
    "free_adaptation",
    "mechanism_preserving",
    "theme_preserving",
    "french_lexical_hints",
]

OUTPUT_COLUMNS = [
    "candidate_json",
    "candidate_count",
    "generation_error",
]


def log(*args: Any) -> None:
    if VERBOSE:
        print(*args)


def safe_list(x: Any) -> list[str]:
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    if isinstance(x, str):
        for parser in (ast.literal_eval, json.loads):
            try:
                v = parser(x)
                if isinstance(v, list):
                    return [str(i) for i in v if str(i).strip()]
            except Exception:
                pass
    return []


def norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in candidates:
        french = norm_space(c.get("french", ""))
        if not french or french.lower() in seen:
            continue
        seen.add(french.lower())
        c["french"] = french
        out.append(c)
    return out


async def run_async_apply(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[pd.Series]],
    result_columns: list[str],
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(index, row):
        async with semaphore:
            result = await apply_async_fn(row)
            return index, result

    tasks = [asyncio.create_task(worker(index, row)) for index, row in chunk_df.iterrows()]
    results: dict[Any, pd.Series] = {}
    try:
        for task in asyncio.as_completed(tasks):
            index, result = await task
            results[index] = result
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    ordered = [results[index] for index in chunk_df.index]
    result_df = pd.DataFrame(ordered, index=chunk_df.index)
    return result_df[result_columns]


def build_generation_prompt(row: pd.Series, mode: str, n: int) -> str:
    text_clean = norm_space(row.get("text_clean", ""))
    pun_word = norm_space(row.get("pun_word", ""))
    pun_type = norm_space(row.get("pun_type", ""))
    first_meaning = safe_list(row.get("first_meaning", []))
    second_meaning = safe_list(row.get("second_meaning", []))
    first_meaning_fr = safe_list(row.get("first_meaning_fr", []))
    second_meaning_fr = safe_list(row.get("second_meaning_fr", []))
    pun_word_fr = norm_space(row.get("pun_word_fr", ""))

    mode_instructions = {
        "free_adaptation": "Make the funniest natural French pun you can. You may change the imagery, setting, objects, and literal meaning freely.",
        "mechanism_preserving": "Try to preserve the source joke's wordplay mechanism: homophone with homophone, ambiguity with ambiguity, idiom twist with idiom twist. Do not force this if it kills the joke.",
        "theme_preserving": "Try to keep the broad theme or semantic field of the English joke, but humor still overrides literal meaning.",
        "french_lexical_hints": f"Use the French lexical hints if they help. The direct French pun-word candidate is {pun_word_fr!r}, but you may ignore it if it produces a weak or literal joke.",
    }[mode]

    return f"""
You are a native French comedy writer specializing in puns.

Task: create {n} different French pun sentences inspired by the English joke.
Do NOT translate literally. Humor is the top priority.

Priority order:
1. Funny to native French speakers.
2. Wordplay is obvious on first read or after a very brief click.
3. Natural, idiomatic, well-edited French.
4. Comparable comedic mechanism if possible.
5. Similar theme or sense only if possible.

Generation mode: {mode}
Mode instruction: {mode_instructions}

English joke:
{text_clean}

Diagnostic hints, not constraints:
- English pun word: {pun_word}
- English pun type: {pun_type}
- English meaning cluster A: {first_meaning}
- English meaning cluster B: {second_meaning}
- French lexical hints A: {first_meaning_fr}
- French lexical hints B: {second_meaning_fr}

Rules:
- The output sentence must be in French only.
- Avoid obscure vocabulary and private explanations inside the joke.
- It is acceptable to change meaning, imagery, or grammar to make the joke work.
- Do not output a literal translation unless it is also a strong French pun.

Return only valid JSON with this exact shape:
{{
  "candidates": [
    {{
      "french": "French pun sentence",
      "pun_word": "main French pun word or phrase",
      "mechanism": "homophone|homograph|polysemy|idiom|paronymy|other",
      "source_relation": "free|mechanism|theme|lexical_hint",
      "why_it_works": "brief English explanation"
    }}
  ]
}}
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
                    "pun_word": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "source_relation": {"type": "string"},
                    "why_it_works": {"type": "string"},
                },
                "required": ["french", "pun_word", "mechanism", "source_relation", "why_it_works"],
            },
        }
    },
    "required": ["candidates"],
}


async def generate_row(row: pd.Series, model: str) -> pd.Series:
    all_candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    row_id = row.get("id_en", row.name)

    for mode in GENERATION_MODES:
        prompt = build_generation_prompt(row, mode, CANDIDATES_PER_MODE)
        try:
            response = await get_response_async(
                prompt,
                model,
                response_schema=RESPONSE_SCHEMA,
                required_keys=["candidates"],
                routing_preset="stable",
            )
            candidates = response.get("candidates", [])
            for c in candidates:
                c["generation_mode"] = mode
                c["generator_model"] = model
            all_candidates.extend(candidates)
        except Exception as e:
            errors.append(f"{mode}: {e}")

    all_candidates = dedupe_candidates(all_candidates)
    log(row.name, row_id, f"generated={len(all_candidates)}", f"errors={len(errors)}")
    return pd.Series({
        "candidate_json": json.dumps(all_candidates, ensure_ascii=False),
        "candidate_count": len(all_candidates),
        "generation_error": " | ".join(errors),
    })


async def generate_french_pun_candidates(df: pd.DataFrame, model: str, start: int = 0, end: int = -1) -> None:
    chunks = [df.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
    end = len(chunks) if end == -1 else end

    for i in range(start, end):
        chunk = chunks[i].copy()
        chunk[OUTPUT_COLUMNS] = await run_async_apply(chunk, lambda row: generate_row(row, model), OUTPUT_COLUMNS)
        save(chunk, f"{generate_dir}{model}/candidates/{i}.tsv")


async def main() -> None:
    task = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    if task == "generate":
        df = load_all(f"{translate_dir}{model}/")
        await generate_french_pun_candidates(df, model, start, end)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    asyncio.run(main())
