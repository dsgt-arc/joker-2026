"""
JOKER 2026 discriminator Run 1: broad four-judge prefilter.

Reads chunked shuffled candidate TSVs and aligned English identify TSVs:
  - French candidates: data/processed/shuffle/{generator_run}/{chunk}.tsv
  - English source:    data/processed/identify/gemini/{chunk}.tsv

For each row, sends one compact prompt asking for ranked top-five candidate IDs
for each judge perspective. The candidate JSON included in the prompt strips the
model-identifying `run` field.

Usage:
  python discriminator_run1_v5.py run claude gpt 0 1
  python discriminator_run1_v5.py run claude gemini_pro 0 -1
  python discriminator_run1_v5.py borda claude gpt 45 30 15 10 0 1

Borda output:
  ../data/processed/discriminate/run1/{generator_run}/{judge}/borda/{comedian}_{pun_expert}_{editor}_{translator}/{chunk}.tsv

Environment variables:
  DISCRIMINATOR_RUN1_OUTPUT_DIR   default ../data/processed/discriminate/run1/
  DISCRIMINATOR_RUN1_MAX_CONCURRENCY default 8
  DISCRIMINATOR_RUN1_VERBOSE      1/0, default 1
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

from config import MODEL_ALIASES, identify_dir, shuffle_dir
from data import load, save
from utils import get_response_async

pd.options.mode.chained_assignment = None

RUN1_VERSION = "discriminator_run1_v5"
DEFAULT_JUDGE_MODEL = os.environ.get("DISCRIMINATOR_RUN1_MODEL", "gpt")
VERBOSE = os.environ.get("DISCRIMINATOR_RUN1_VERBOSE", "1") == "1"
MAX_CONCURRENCY = int(os.environ.get("DISCRIMINATOR_RUN1_MAX_CONCURRENCY", "8"))
OUTPUT_ROOT = os.environ.get(
    "DISCRIMINATOR_RUN1_OUTPUT_DIR",
    "../data/processed/discriminate/run1/",
)
JUDGE_KEYS = ["comedian", "pun_expert", "editor", "translator"]
OUTPUT_COLUMNS = [
    "discriminator_run1_json",
    "discriminator_run1_error",
    "discriminator_run1_model",
    "discriminator_run1_model_id",
    "discriminator_run1_version",
]

BORDA_OUTPUT_COLUMNS = [
    "discriminator_run1_borda_weights_json",
    "discriminator_run1_borda_scores_json",
    "discriminator_run1_borda_ranking_json",
    "discriminator_run1_winner_id",
    "discriminator_run1_winner_pun",
    "discriminator_run1_winner_score",
    "discriminator_run1_borda_error",
    "discriminator_run1_borda_version",
]

DEFAULT_BORDA_WEIGHTS = {
    "comedian": 45.0,
    "pun_expert": 30.0,
    "editor": 15.0,
    "translator": 10.0,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        key: {
            "type": "array",
            "items": {"type": "integer"}
        }
        for key in JUDGE_KEYS
    },
    "required": JUDGE_KEYS,
}


def log(*args: Any) -> None:
    if VERBOSE:
        print(*args)


def norm_space(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def ensure_slash(path: str) -> str:
    return str(path or "").rstrip("/") + "/"


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
    return json.loads(text)


async def run_async_apply(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[pd.Series]],
    result_columns: list[str],
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(index: Any, row: pd.Series):
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

    ordered_rows = [results[index] for index in chunk_df.index]
    result_df = pd.DataFrame(ordered_rows, index=chunk_df.index)
    return result_df[result_columns]


def resolve_model_alias(model_arg: str) -> tuple[str, str]:
    model_arg = norm_space(model_arg)
    if model_arg in MODEL_ALIASES and MODEL_ALIASES.get(model_arg):
        return model_arg, MODEL_ALIASES[model_arg]

    filesystem_alias = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_arg).strip("_")
    return filesystem_alias or "model", model_arg


def validate_shuffle_df(df: pd.DataFrame, path: str) -> None:
    required = ["id_en", "shuffled_candidates_json"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")


def validate_identify_df(df: pd.DataFrame, path: str) -> None:
    required = ["id_en", "text_clean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")


def prompt_candidates_from_json(raw_json: Any) -> list[dict[str, Any]]:
    raw = safe_json_loads(raw_json)
    if not isinstance(raw, list):
        raise ValueError("shuffled_candidates_json must be a JSON array")

    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for item in raw:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "pun" not in item:
            continue
        candidate_id = int(item["id"])
        if candidate_id in seen_ids:
            continue
        pun = norm_space(item.get("pun", ""))
        if not pun:
            continue
        seen_ids.add(candidate_id)
        # Deliberately strip model/source metadata such as `run`.
        candidates.append({"id": candidate_id, "pun": pun})

    if len(candidates) < 5:
        raise ValueError(f"Expected at least 5 usable candidates, got {len(candidates)}")

    return candidates


def build_prompt(candidates: list[dict[str, Any]], english_pun: str) -> str:
    candidates_json = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    english_pun = norm_space(english_pun)

    return f"""French pun candidates:
{candidates_json}

Rank the top five candidates for each judge, from best to worst.
Return exactly five distinct candidate IDs per judge.
For Judges 1–3, judge only the French text.

Judge 1: Comedian
Which candidate is funniest to a native French speaker?

Judge 2: Pun Expert
Could a native French speaker immediately identify the wordplay mechanism?
Prefer clear, intentional wordplay over general cleverness. Reject obscure puns.

Judge 3: Editor
Which candidate sounds most natural, fluent, and publishable in French?
Penalize errors in syntax or awkward word choice.

Judge 4: Translator
Which candidate best preserves the underlying setup and humorous intention of this English joke?
Consider lexical field preservation, sense preservation, wordplay form preservation, style shift, and humorousness shift.
Prefer preservation of humorous intent over literal wording.
{english_pun}

Return JSON only:
{{
  "comedian": [id1, id2, id3, id4, id5],
  "pun_expert": [id1, id2, id3, id4, id5],
  "editor": [id1, id2, id3, id4, id5],
  "translator": [id1, id2, id3, id4, id5]
}}"""


def validate_response(response: pd.Series, valid_ids: set[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for key in JUDGE_KEYS:
        value = response.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Response key {key!r} is not a list")
        ids = [int(v) for v in value]
        if len(ids) != 5:
            raise ValueError(f"Response key {key!r} must contain exactly 5 IDs, got {len(ids)}")
        if len(set(ids)) != 5:
            raise ValueError(f"Response key {key!r} contains duplicate IDs: {ids}")
        invalid = [v for v in ids if v not in valid_ids]
        if invalid:
            raise ValueError(f"Response key {key!r} contains IDs not in input: {invalid}")
        out[key] = ids
    return out


async def discriminate_row(row: pd.Series, judge_alias: str, judge_model_id: str) -> pd.Series:
    row_id = row.get("id_en", row.name)

    try:
        candidates = prompt_candidates_from_json(row.get("shuffled_candidates_json", ""))
        valid_ids = {int(c["id"]) for c in candidates}
        prompt = build_prompt(candidates, row.get("text_clean", ""))

        response = await get_response_async(
            prompt,
            judge_alias,
            response_schema=RESPONSE_SCHEMA,
            required_keys=JUDGE_KEYS,
            routing_preset="stable",
            temperature=0,
        )
        rankings = validate_response(response, valid_ids)
        error = ""
    except Exception as e:
        print(f"Error id_en={row_id}: {e}")
        rankings = {key: [] for key in JUDGE_KEYS}
        error = str(e)

    log(row.name, row_id, f"error={bool(error)}")
    return pd.Series(
        {
            "discriminator_run1_json": json.dumps(rankings, ensure_ascii=False, separators=(",", ":")),
            "discriminator_run1_error": error,
            "discriminator_run1_model": judge_alias,
            "discriminator_run1_model_id": judge_model_id,
            "discriminator_run1_version": RUN1_VERSION,
        }
    )


def chunk_numbers_for_run(generator_run: str) -> list[int]:
    input_dir = ensure_slash(shuffle_dir) + ensure_slash(generator_run)
    files = glob.glob(input_dir + "*.tsv")
    chunks: list[int] = []
    for path in files:
        stem = Path(path).stem
        if stem.isdigit():
            chunks.append(int(stem))
    return sorted(chunks)


def load_chunk(generator_run: str, chunk_num: int) -> pd.DataFrame:
    shuffle_path = f"{ensure_slash(shuffle_dir)}{ensure_slash(generator_run)}{chunk_num}.tsv"
    identify_path = f"{ensure_slash(identify_dir)}gemini/{chunk_num}.tsv"

    if not os.path.exists(shuffle_path):
        raise FileNotFoundError(f"Missing shuffle chunk: {shuffle_path}")
    if not os.path.exists(identify_path):
        raise FileNotFoundError(f"Missing identify chunk: {identify_path}")

    shuffle_df = load(shuffle_path)
    identify_df = load(identify_path)
    validate_shuffle_df(shuffle_df, shuffle_path)
    validate_identify_df(identify_df, identify_path)

    identify_small = identify_df[["id_en", "text_clean"]].copy()
    merged = shuffle_df.merge(identify_small, on="id_en", how="left", validate="one_to_one")
    missing_text = merged["text_clean"].isna() | (merged["text_clean"].astype(str).str.strip() == "")
    if missing_text.any():
        missing_ids = merged.loc[missing_text, "id_en"].head(10).tolist()
        raise ValueError(f"Missing text_clean for {missing_text.sum()} rows; sample id_en={missing_ids}")
    return merged


async def run_discriminator(generator_run: str, judge_arg: str, start: int = 0, end: int = -1) -> None:
    judge_alias, judge_model_id = resolve_model_alias(judge_arg)

    chunks = chunk_numbers_for_run(generator_run)
    if not chunks:
        raise FileNotFoundError(f"No chunk TSV files found under {ensure_slash(shuffle_dir)}{generator_run}/")

    selected = [c for c in chunks if c >= start and (end == -1 or c < end)]
    if not selected:
        raise ValueError(f"No chunks selected for start={start}, end={end}; available={chunks[:10]}...")

    log("Generator run:", generator_run)
    log("Judge alias:", judge_alias)
    log("OpenRouter model:", judge_model_id)
    log("Chunks:", selected)

    for chunk_num in selected:
        chunk = load_chunk(generator_run, chunk_num)
        chunk[OUTPUT_COLUMNS] = await run_async_apply(
            chunk,
            lambda row: discriminate_row(row, judge_alias, judge_model_id),
            OUTPUT_COLUMNS,
        )

        out_path = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(generator_run)}{ensure_slash(judge_alias)}{chunk_num}.tsv"
        save(chunk, out_path)



def candidate_pun_map_from_json(raw_json: Any) -> dict[int, str]:
    candidates = prompt_candidates_from_json(raw_json)
    return {int(c["id"]): str(c["pun"]) for c in candidates}


def calculate_weighted_borda(
    rankings: dict[str, list[int]],
    weights: dict[str, float],
) -> tuple[dict[int, float], list[int]]:
    scores: dict[int, float] = {}
    for judge in JUDGE_KEYS:
        ranking = rankings.get(judge, [])
        weight = float(weights[judge])
        n = len(ranking)
        for pos, candidate_id in enumerate(ranking):
            # Top-five Borda: 1st=5, 2nd=4, ..., 5th=1.
            points = n - pos
            cid = int(candidate_id)
            scores[cid] = scores.get(cid, 0.0) + weight * points

    ranked_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return scores, ranked_ids


def parse_borda_weights(args: list[str]) -> dict[str, float]:
    if len(args) != 4:
        raise ValueError(
            "borda requires four weights: <comedian> <pun_expert> <editor> <translator>"
        )
    values = [float(x) for x in args]
    return dict(zip(JUDGE_KEYS, values))


def calculate_borda_row(row: pd.Series, weights: dict[str, float]) -> pd.Series:
    row_id = row.get("id_en", row.name)
    try:
        if norm_space(row.get("discriminator_run1_error", "")):
            raise ValueError(f"Cannot calculate Borda because Run 1 has error: {row.get('discriminator_run1_error')}")

        rankings_raw = safe_json_loads(row.get("discriminator_run1_json", ""))
        if not isinstance(rankings_raw, dict):
            raise ValueError("discriminator_run1_json must be a JSON object")

        candidates_by_id = candidate_pun_map_from_json(row.get("shuffled_candidates_json", ""))
        valid_ids = set(candidates_by_id)
        rankings = validate_response(pd.Series(rankings_raw), valid_ids)

        scores, ranked_ids = calculate_weighted_borda(rankings, weights)
        scores_sorted = {str(cid): scores[cid] for cid in ranked_ids}
        ranking_json = [
            {
                "id": cid,
                "score": scores[cid],
                "pun": candidates_by_id.get(cid, ""),
            }
            for cid in ranked_ids
        ]

        winner_id = ranked_ids[0] if ranked_ids else ""
        winner_pun = candidates_by_id.get(int(winner_id), "") if winner_id != "" else ""
        winner_score = scores.get(int(winner_id), 0.0) if winner_id != "" else 0.0
        error = ""
    except Exception as e:
        print(f"Borda error id_en={row_id}: {e}")
        scores_sorted = {}
        ranking_json = []
        winner_id = ""
        winner_pun = ""
        winner_score = ""
        error = str(e)

    log(row.name, row_id, f"winner={winner_id} score={winner_score} borda_error={bool(error)}")
    return pd.Series(
        {
            "discriminator_run1_borda_weights_json": json.dumps(weights, ensure_ascii=False, separators=(",", ":")),
            "discriminator_run1_borda_scores_json": json.dumps(scores_sorted, ensure_ascii=False, separators=(",", ":")),
            "discriminator_run1_borda_ranking_json": json.dumps(ranking_json, ensure_ascii=False, separators=(",", ":")),
            "discriminator_run1_winner_id": winner_id,
            "discriminator_run1_winner_pun": winner_pun,
            "discriminator_run1_winner_score": winner_score,
            "discriminator_run1_borda_error": error,
            "discriminator_run1_borda_version": RUN1_VERSION,
        }
    )


def load_run1_chunk(generator_run: str, judge_alias: str, chunk_num: int) -> pd.DataFrame:
    run1_path = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(generator_run)}{ensure_slash(judge_alias)}{chunk_num}.tsv"
    if not os.path.exists(run1_path):
        raise FileNotFoundError(f"Missing Run 1 chunk: {run1_path}")
    df = load(run1_path)
    missing = [c for c in ["id_en", "shuffled_candidates_json", "discriminator_run1_json"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {run1_path}: {', '.join(missing)}")
    return df


def format_weight_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def borda_weight_key(weights: dict[str, float]) -> str:
    return "_".join(format_weight_value(weights[key]) for key in JUDGE_KEYS)


def run_borda(generator_run: str, judge_arg: str, weights: dict[str, float], start: int = 0, end: int = -1) -> None:
    judge_alias, _ = resolve_model_alias(judge_arg)

    input_dir = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(generator_run)}{ensure_slash(judge_alias)}"
    files = glob.glob(input_dir + "*.tsv")
    chunks: list[int] = []
    for path in files:
        stem = Path(path).stem
        if stem.isdigit():
            chunks.append(int(stem))
    chunks = sorted(chunks)
    if not chunks:
        raise FileNotFoundError(f"No Run 1 TSV files found under {input_dir}")

    selected = [c for c in chunks if c >= start and (end == -1 or c < end)]
    if not selected:
        raise ValueError(f"No chunks selected for start={start}, end={end}; available={chunks[:10]}...")

    log("Borda input Run 1:", input_dir.rstrip("/"))
    log("Generator run:", generator_run)
    log("Judge alias:", judge_alias)
    log("Weights:", weights)
    log("Weight key:", borda_weight_key(weights))
    log("Chunks:", selected)

    for chunk_num in selected:
        chunk = load_run1_chunk(generator_run, judge_alias, chunk_num)
        borda_df = pd.DataFrame(
            [calculate_borda_row(row, weights) for _, row in chunk.iterrows()],
            index=chunk.index,
        )
        chunk[BORDA_OUTPUT_COLUMNS] = borda_df[BORDA_OUTPUT_COLUMNS]
        weight_key = borda_weight_key(weights)
        out_path = (
            f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(generator_run)}{ensure_slash(judge_alias)}"
            f"borda/{ensure_slash(weight_key)}{chunk_num}.tsv"
        )
        save(chunk, out_path)


async def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError(
            """Usage:
  python discriminator_run1_v5.py run <generator_run> <judge_model> <start> <end>
  python discriminator_run1_v5.py borda <generator_run> <judge_model> <comedian_w> <pun_expert_w> <editor_w> <translator_w> <start> <end>

Examples:
  python discriminator_run1_v5.py run claude gpt 0 1
  python discriminator_run1_v5.py run claude gemini_pro 0 -1
  python discriminator_run1_v5.py borda claude gpt 45 30 15 10 0 1"""
        )

    task = sys.argv[1]

    if task == "run":
        generator_run = sys.argv[2] if len(sys.argv) > 2 else "claude"
        judge_model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_JUDGE_MODEL
        start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        end = int(sys.argv[5]) if len(sys.argv) > 5 else -1
        await run_discriminator(generator_run, judge_model, start, end)
        return

    if task == "borda":
        generator_run = sys.argv[2] if len(sys.argv) > 2 else "claude"
        judge_model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_JUDGE_MODEL
        weights = parse_borda_weights(sys.argv[4:8]) if len(sys.argv) >= 8 else DEFAULT_BORDA_WEIGHTS
        start = int(sys.argv[8]) if len(sys.argv) > 8 else 0
        end = int(sys.argv[9]) if len(sys.argv) > 9 else -1
        run_borda(generator_run, judge_model, weights, start, end)
        return

    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    asyncio.run(main())
