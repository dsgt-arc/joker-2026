"""
JOKER 2026 discriminator Run 2 v4: ensemble judge over four system winners.

Inputs, chunked as {chunk}.tsv:
  - Claude Run 1 Borda winner:
      ../data/processed/discriminate/run1/claude/claude/borda/25_25_25_25/{chunk}.tsv
  - Gemini Flash Run 1 Borda winner:
      ../data/processed/discriminate/run1/gemini/gemini/borda/25_25_25_25/{chunk}.tsv
  - Gemini Pro single generation:
      ../data/processed/generate_single/gemini_pro/{chunk}.tsv
  - GPT single generation:
      ../data/processed/generate_single/gpt/{chunk}.tsv

For each row, builds four candidates, assigns fresh deterministic random prompt IDs,
shuffles them deterministically, strips source/model metadata from the prompt, and asks
one judge model to rank all four candidates for each of the four locked judge perspectives.

Usage:
  python discriminator_run2_v4.py run ensemble gpt 0 1
  python discriminator_run2_v4.py run ensemble claude 0 -1
  python discriminator_run2_v4.py borda ensemble 25_25_25_25 25_25_25 0 1

Run output:
  ../data/processed/discriminate/run2/{ensemble_run}/{judge}/{chunk}.tsv

Borda output:
  ../data/processed/discriminate/run2/{ensemble_run}/{judge}/borda/{comedian}_{pun_expert}_{editor}_{translator}/{chunk}.tsv

Environment variables:
  DISCRIMINATOR_RUN2_OUTPUT_DIR       default ../data/processed/discriminate/run2/
  DISCRIMINATOR_RUN2_MAX_CONCURRENCY  default 8
  DISCRIMINATOR_RUN2_VERBOSE          1/0, default 1
  DISCRIMINATOR_RUN2_SHUFFLE_SEED     default joker-2026-run2-v4
  DISCRIMINATOR_RUN2_CLAUDE_DIR       override Claude Borda input dir
  DISCRIMINATOR_RUN2_GEMINI_DIR       override Gemini Flash Borda input dir
  DISCRIMINATOR_RUN2_GEMINI_PRO_DIR   override Gemini Pro single input dir
  DISCRIMINATOR_RUN2_GPT_DIR          override GPT single input dir
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

from config import MODEL_ALIASES
from data import load, save
from utils import get_response_async

pd.options.mode.chained_assignment = None

RUN2_VERSION = "discriminator_run2_v4"
DEFAULT_JUDGE_MODEL = os.environ.get("DISCRIMINATOR_RUN2_MODEL", "gpt")
VERBOSE = os.environ.get("DISCRIMINATOR_RUN2_VERBOSE", "1") == "1"
MAX_CONCURRENCY = int(os.environ.get("DISCRIMINATOR_RUN2_MAX_CONCURRENCY", "8"))
OUTPUT_ROOT = os.environ.get("DISCRIMINATOR_RUN2_OUTPUT_DIR", "../data/processed/discriminate/run2/")
SHUFFLE_SEED = os.environ.get("DISCRIMINATOR_RUN2_SHUFFLE_SEED", "joker-2026-run2-v4")

CLAUDE_BORDA_DIR = os.environ.get(
    "DISCRIMINATOR_RUN2_CLAUDE_DIR",
    "../data/processed/discriminate/run1/claude/claude/borda/25_25_25_25/",
)
GEMINI_BORDA_DIR = os.environ.get(
    "DISCRIMINATOR_RUN2_GEMINI_DIR",
    "../data/processed/discriminate/run1/gemini/gemini/borda/25_25_25_25/",
)
GEMINI_PRO_SINGLE_DIR = os.environ.get(
    "DISCRIMINATOR_RUN2_GEMINI_PRO_DIR",
    "../data/processed/generate_single/gemini_pro/",
)
GPT_SINGLE_DIR = os.environ.get(
    "DISCRIMINATOR_RUN2_GPT_DIR",
    "../data/processed/generate_single/gpt/",
)

JUDGE_KEYS = ["comedian", "pun_expert", "editor", "translator"]
SOURCE_KEYS = ["claude", "gemini", "gemini_pro_single", "gpt_single"]

OUTPUT_COLUMNS = [
    "shuffled_candidates_json",
    "discriminator_run2_json",
    "discriminator_run2_error",
    "discriminator_run2_model",
    "discriminator_run2_model_id",
    "discriminator_run2_version",
]

BORDA_OUTPUT_COLUMNS = [
    "discriminator_run2_borda_weights_json",
    "discriminator_run2_borda_scores_json",
    "discriminator_run2_borda_ranking_json",
    "discriminator_run2_winner_id",
    "discriminator_run2_winner_source",
    "discriminator_run2_winner_original_id",
    "discriminator_run2_winner_pun",
    "discriminator_run2_winner_score",
    "discriminator_run2_borda_error",
    "discriminator_run2_borda_version",
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
    "properties": {key: {"type": "array", "items": {"type": "integer"}} for key in JUDGE_KEYS},
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


def chunk_numbers_in_dir(input_dir: str) -> list[int]:
    chunks: list[int] = []
    for path in glob.glob(ensure_slash(input_dir) + "*.tsv"):
        stem = Path(path).stem
        if stem.isdigit():
            chunks.append(int(stem))
    return sorted(chunks)


def available_chunks() -> list[int]:
    chunk_sets = []
    for input_dir in [CLAUDE_BORDA_DIR, GEMINI_BORDA_DIR, GEMINI_PRO_SINGLE_DIR, GPT_SINGLE_DIR]:
        chunks = set(chunk_numbers_in_dir(input_dir))
        if not chunks:
            raise FileNotFoundError(f"No chunk TSV files found under {ensure_slash(input_dir)}")
        chunk_sets.append(chunks)
    return sorted(set.intersection(*chunk_sets))


def validate_has_id(df: pd.DataFrame, path: str) -> None:
    if "id_en" not in df.columns:
        raise ValueError(f"Missing required column in {path}: id_en")


def load_tsv_chunk(input_dir: str, chunk_num: int) -> tuple[pd.DataFrame, str]:
    path = f"{ensure_slash(input_dir)}{chunk_num}.tsv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing chunk: {path}")
    df = load(path)
    validate_has_id(df, path)
    return df, path


def extract_borda_candidate(row: pd.Series, source_name: str) -> dict[str, Any]:
    pun = norm_space(row.get("discriminator_run1_winner_pun", ""))
    if not pun:
        raise ValueError(f"Missing discriminator_run1_winner_pun for {source_name}")
    original_id = row.get("discriminator_run1_winner_id", "")
    try:
        if pd.isna(original_id):
            original_id = ""
    except Exception:
        pass
    return {"source": source_name, "original_id": str(original_id), "pun": pun}


def extract_single_candidate(row: pd.Series, source_name: str) -> dict[str, Any]:
    raw = safe_json_loads(row.get("candidate_json", ""))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"candidate_json for {source_name} must be a non-empty JSON array")
    first = raw[0]
    if not isinstance(first, dict):
        raise ValueError(f"candidate_json[0] for {source_name} must be an object")
    pun = norm_space(first.get("french", ""))
    if not pun:
        raise ValueError(f"Missing candidate_json[0]['french'] for {source_name}")
    original_id = first.get("id", first.get("candidate_id", ""))
    try:
        if pd.isna(original_id):
            original_id = ""
    except Exception:
        pass
    return {"source": source_name, "original_id": str(original_id), "pun": pun}


def random_prompt_ids(row_id: Any) -> list[int]:
    seed_text = f"{SHUFFLE_SEED}|ids|{row_id}"
    seed_int = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed_int)
    return rng.sample(range(10000, 99999), 4)


def shuffle_candidates(candidates: list[dict[str, Any]], row_id: Any) -> list[dict[str, Any]]:
    seed_text = f"{SHUFFLE_SEED}|order|{row_id}"
    seed_int = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed_int)
    out = [dict(c) for c in candidates]
    rng.shuffle(out)
    for prompt_id, item in zip(random_prompt_ids(row_id), out):
        item["id"] = int(prompt_id)
    return out


def prompt_candidates_without_source(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": int(c["id"]), "pun": str(c["pun"])} for c in candidates]


def build_prompt(candidates: list[dict[str, Any]], english_pun: str) -> str:
    candidates_json = json.dumps(prompt_candidates_without_source(candidates), ensure_ascii=False, separators=(",", ":"))
    english_pun = norm_space(english_pun)
    return f"""French pun candidates:
{candidates_json}

Rank all four candidates for each judge, from best to worst.
Return exactly four distinct candidate IDs per judge.
For Judges 1–3, judge only the French text.

Judge 1: Native French Comedian
Which candidate is funnier to a native French speaker?
Ignore faithfulness unless one option completely loses the original joke.

Judge 2: Pun Expert
Which candidate demonstrates stronger wordplay?
Prefer genuine puns, ambiguity, double meanings, homophony, lexical creativity, and elegance.
Do not reward similarity to the English source. Reward the quality of the pun itself.

Judge 3: French Literary Editor
Which candidate sounds most natural, fluent, and publishable in French?
Penalize translationese, awkward syntax, forced constructions, and unnatural word order.

Judge 4: Translation Scholar
Which candidate best preserves the underlying setup and humorous intention of this English joke?
Evaluate preservation of humorous intent, not lexical overlap.
Do not reward literal translation unless it helps preserve the joke.
English joke: {english_pun}

Return JSON only:
{{
  "comedian": [id1, id2, id3, id4],
  "pun_expert": [id1, id2, id3, id4],
  "editor": [id1, id2, id3, id4],
  "translator": [id1, id2, id3, id4]
}}"""


def validate_response(response: pd.Series, valid_ids: set[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for key in JUDGE_KEYS:
        value = response.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Response key {key!r} is not a list")
        ids = [int(v) for v in value]
        if len(ids) != 4:
            raise ValueError(f"Response key {key!r} must contain exactly 4 IDs, got {len(ids)}")
        if len(set(ids)) != 4:
            raise ValueError(f"Response key {key!r} contains duplicate IDs: {ids}")
        invalid = [v for v in ids if v not in valid_ids]
        if invalid:
            raise ValueError(f"Response key {key!r} contains IDs not in input: {invalid}")
        out[key] = ids
    return out


def load_chunk(chunk_num: int) -> pd.DataFrame:
    claude_df, claude_path = load_tsv_chunk(CLAUDE_BORDA_DIR, chunk_num)
    gemini_df, gemini_path = load_tsv_chunk(GEMINI_BORDA_DIR, chunk_num)
    gemini_pro_df, gemini_pro_path = load_tsv_chunk(GEMINI_PRO_SINGLE_DIR, chunk_num)
    gpt_df, gpt_path = load_tsv_chunk(GPT_SINGLE_DIR, chunk_num)

    for path, df in [(claude_path, claude_df), (gemini_path, gemini_df)]:
        missing = [c for c in ["id_en", "discriminator_run1_winner_id", "discriminator_run1_winner_pun"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    for path, df in [(gemini_pro_path, gemini_pro_df), (gpt_path, gpt_df)]:
        if "candidate_json" not in df.columns:
            raise ValueError(f"Missing required column in {path}: candidate_json")

    claude_small = claude_df[["id_en"]].copy()
    claude_small["claude_candidate_json"] = claude_df.apply(
        lambda r: json.dumps(extract_borda_candidate(r, "claude"), ensure_ascii=False, separators=(",", ":")),
        axis=1,
    )

    gemini_small = gemini_df[["id_en"]].copy()
    gemini_small["gemini_candidate_json"] = gemini_df.apply(
        lambda r: json.dumps(extract_borda_candidate(r, "gemini"), ensure_ascii=False, separators=(",", ":")),
        axis=1,
    )

    gemini_pro_small = gemini_pro_df[["id_en"]].copy()
    gemini_pro_small["gemini_pro_single_candidate_json"] = gemini_pro_df.apply(
        lambda r: json.dumps(extract_single_candidate(r, "gemini_pro_single"), ensure_ascii=False, separators=(",", ":")),
        axis=1,
    )

    gpt_small = gpt_df[["id_en"]].copy()
    gpt_small["gpt_single_candidate_json"] = gpt_df.apply(
        lambda r: json.dumps(extract_single_candidate(r, "gpt_single"), ensure_ascii=False, separators=(",", ":")),
        axis=1,
    )

    text_source = None
    for df in [claude_df, gemini_df, gemini_pro_df, gpt_df]:
        if "text_clean" in df.columns:
            text_source = df[["id_en", "text_clean"]].copy()
            break
        if "en" in df.columns:
            text_source = df[["id_en", "en"]].rename(columns={"en": "text_clean"}).copy()
            break
    if text_source is None:
        raise ValueError("Could not find English source column text_clean or en in any input file")

    merged = text_source.merge(claude_small, on="id_en", how="inner", validate="one_to_one")
    merged = merged.merge(gemini_small, on="id_en", how="inner", validate="one_to_one")
    merged = merged.merge(gemini_pro_small, on="id_en", how="inner", validate="one_to_one")
    merged = merged.merge(gpt_small, on="id_en", how="inner", validate="one_to_one")

    expected = len(text_source)
    if len(merged) != expected:
        raise ValueError(f"Merged chunk has {len(merged)} rows but text source has {expected}; check id_en alignment for chunk {chunk_num}")

    def row_candidates(row: pd.Series) -> list[dict[str, Any]]:
        raw = [
            safe_json_loads(row["claude_candidate_json"]),
            safe_json_loads(row["gemini_candidate_json"]),
            safe_json_loads(row["gemini_pro_single_candidate_json"]),
            safe_json_loads(row["gpt_single_candidate_json"]),
        ]
        candidates = [c for c in raw if isinstance(c, dict)]
        if len(candidates) != 4:
            raise ValueError(f"Expected 4 candidates, got {len(candidates)}")
        return shuffle_candidates(candidates, row.get("id_en", ""))

    merged["shuffled_candidates_json"] = merged.apply(
        lambda row: json.dumps(row_candidates(row), ensure_ascii=False, separators=(",", ":")),
        axis=1,
    )
    return merged


def candidates_from_row(row: pd.Series) -> list[dict[str, Any]]:
    raw = safe_json_loads(row.get("shuffled_candidates_json", ""))
    if not isinstance(raw, list):
        raise ValueError("shuffled_candidates_json must be a JSON array")
    candidates: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "source" not in item or "pun" not in item:
            continue
        candidates.append({
            "id": int(item["id"]),
            "source": str(item["source"]),
            "original_id": str(item.get("original_id", "")),
            "pun": norm_space(item["pun"]),
        })
    if len(candidates) != 4:
        raise ValueError(f"Expected exactly 4 shuffled candidates, got {len(candidates)}")
    return candidates


def candidate_map_from_row(row: pd.Series) -> dict[int, dict[str, str]]:
    return {
        int(c["id"]): {"source": str(c["source"]), "original_id": str(c.get("original_id", "")), "pun": str(c["pun"])}
        for c in candidates_from_row(row)
    }


async def discriminate_row(row: pd.Series, judge_alias: str, judge_model_id: str) -> pd.Series:
    row_id = row.get("id_en", row.name)
    try:
        candidates = candidates_from_row(row)
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
    return pd.Series({
        "shuffled_candidates_json": row.get("shuffled_candidates_json", ""),
        "discriminator_run2_json": json.dumps(rankings, ensure_ascii=False, separators=(",", ":")),
        "discriminator_run2_error": error,
        "discriminator_run2_model": judge_alias,
        "discriminator_run2_model_id": judge_model_id,
        "discriminator_run2_version": RUN2_VERSION,
    })


async def run_discriminator(ensemble_run: str, judge_arg: str, start: int = 0, end: int = -1) -> None:
    judge_alias, judge_model_id = resolve_model_alias(judge_arg)
    chunks = available_chunks()
    selected = [c for c in chunks if c >= start and (end == -1 or c < end)]
    if not selected:
        raise ValueError(f"No chunks selected for start={start}, end={end}; available={chunks[:10]}...")

    log("Ensemble run:", ensemble_run)
    log("Judge alias:", judge_alias)
    log("OpenRouter model:", judge_model_id)
    log("Chunks:", selected)

    for chunk_num in selected:
        chunk = load_chunk(chunk_num)
        results = await run_async_apply(
            chunk,
            lambda row: discriminate_row(row, judge_alias, judge_model_id),
            OUTPUT_COLUMNS,
        )
        for col in OUTPUT_COLUMNS:
            chunk[col] = results[col]
        out_path = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}{ensure_slash(judge_alias)}{chunk_num}.tsv"
        save(chunk, out_path)


def calculate_weighted_borda(rankings: dict[str, list[int]], weights: dict[str, float]) -> tuple[dict[int, float], list[int]]:
    scores: dict[int, float] = {}
    for judge in JUDGE_KEYS:
        ranking = rankings.get(judge, [])
        weight = float(weights[judge])
        n = len(ranking)
        for pos, candidate_id in enumerate(ranking):
            points = n - pos
            cid = int(candidate_id)
            scores[cid] = scores.get(cid, 0.0) + weight * points
    ranked_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return scores, ranked_ids


def parse_borda_weights(args: list[str]) -> dict[str, float]:
    if len(args) != 4:
        raise ValueError("borda requires four weights: <comedian> <pun_expert> <editor> <translator>")
    return dict(zip(JUDGE_KEYS, [float(x) for x in args]))


def calculate_borda_row(row: pd.Series, weights: dict[str, float]) -> pd.Series:
    row_id = row.get("id_en", row.name)
    try:
        if norm_space(row.get("discriminator_run2_error", "")):
            raise ValueError(f"Cannot calculate Borda because Run 2 has error: {row.get('discriminator_run2_error')}")
        rankings_raw = safe_json_loads(row.get("discriminator_run2_json", ""))
        if not isinstance(rankings_raw, dict):
            raise ValueError("discriminator_run2_json must be a JSON object")
        candidates_by_id = candidate_map_from_row(row)
        rankings = validate_response(pd.Series(rankings_raw), set(candidates_by_id))
        scores, ranked_ids = calculate_weighted_borda(rankings, weights)

        scores_sorted = {str(cid): scores[cid] for cid in ranked_ids}
        ranking_json = [{
            "id": cid,
            "source": candidates_by_id[cid].get("source", ""),
            "original_id": candidates_by_id[cid].get("original_id", ""),
            "score": scores[cid],
            "pun": candidates_by_id[cid].get("pun", ""),
        } for cid in ranked_ids]

        winner_id = ranked_ids[0] if ranked_ids else ""
        winner_meta = candidates_by_id[int(winner_id)] if winner_id != "" else {}
        winner_source = winner_meta.get("source", "")
        winner_original_id = winner_meta.get("original_id", "")
        winner_pun = winner_meta.get("pun", "")
        winner_score = scores.get(int(winner_id), 0.0) if winner_id != "" else 0.0
        error = ""
    except Exception as e:
        print(f"Borda error id_en={row_id}: {e}")
        scores_sorted = {}
        ranking_json = []
        winner_id = ""
        winner_source = ""
        winner_original_id = ""
        winner_pun = ""
        winner_score = ""
        error = str(e)

    log(row.name, row_id, f"winner={winner_id} source={winner_source} score={winner_score} borda_error={bool(error)}")
    return pd.Series({
        "discriminator_run2_borda_weights_json": json.dumps(weights, ensure_ascii=False, separators=(",", ":")),
        "discriminator_run2_borda_scores_json": json.dumps(scores_sorted, ensure_ascii=False, separators=(",", ":")),
        "discriminator_run2_borda_ranking_json": json.dumps(ranking_json, ensure_ascii=False, separators=(",", ":")),
        "discriminator_run2_winner_id": winner_id,
        "discriminator_run2_winner_source": winner_source,
        "discriminator_run2_winner_original_id": winner_original_id,
        "discriminator_run2_winner_pun": winner_pun,
        "discriminator_run2_winner_score": winner_score,
        "discriminator_run2_borda_error": error,
        "discriminator_run2_borda_version": RUN2_VERSION,
    })


def load_run2_chunk(ensemble_run: str, judge_alias: str, chunk_num: int) -> pd.DataFrame:
    run2_path = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}{ensure_slash(judge_alias)}{chunk_num}.tsv"
    if not os.path.exists(run2_path):
        raise FileNotFoundError(f"Missing Run 2 chunk: {run2_path}")
    df = load(run2_path)
    missing = [c for c in ["id_en", "shuffled_candidates_json", "discriminator_run2_json"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {run2_path}: {', '.join(missing)}")
    return df


def format_weight_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def borda_weight_key(weights: dict[str, float]) -> str:
    return "_".join(format_weight_value(weights[key]) for key in JUDGE_KEYS)


def run_borda(ensemble_run: str, judge_arg: str, weights: dict[str, float], start: int = 0, end: int = -1) -> None:
    judge_alias, _ = resolve_model_alias(judge_arg)
    input_dir = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}{ensure_slash(judge_alias)}"
    chunks = chunk_numbers_in_dir(input_dir)
    if not chunks:
        raise FileNotFoundError(f"No Run 2 TSV files found under {input_dir}")
    selected = [c for c in chunks if c >= start and (end == -1 or c < end)]
    if not selected:
        raise ValueError(f"No chunks selected for start={start}, end={end}; available={chunks[:10]}...")

    log("Borda input Run 2:", input_dir.rstrip("/"))
    log("Ensemble run:", ensemble_run)
    log("Judge alias:", judge_alias)
    log("Weights:", weights)
    log("Weight key:", borda_weight_key(weights))
    log("Chunks:", selected)

    for chunk_num in selected:
        chunk = load_run2_chunk(ensemble_run, judge_alias, chunk_num)
        borda_df = pd.DataFrame([calculate_borda_row(row, weights) for _, row in chunk.iterrows()], index=chunk.index)
        for col in BORDA_OUTPUT_COLUMNS:
            chunk[col] = borda_df[col]
        weight_key = borda_weight_key(weights)
        out_path = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}{ensure_slash(judge_alias)}borda/{ensure_slash(weight_key)}{chunk_num}.tsv"
        save(chunk, out_path)



# -----------------------------------------------------------------------------
# Run 2 v3 cross-model Borda flow.
# This is intentionally separate from the older per-judge Run 2 Borda helper above.
# It consumes raw Run 2 outputs from multiple judge-model directories and writes
# three aggregation methods to separate directories.
# -----------------------------------------------------------------------------

RUN2_BORDA_METHODS = ["judges_then_models", "models_then_judges", "pooled_rankings"]

CROSS_BORDA_OUTPUT_COLUMNS = [
    "discriminator_run2_borda_method",
    "discriminator_run2_internal_weights_json",
    "discriminator_run2_model_weights_json",
    "discriminator_run2_borda_scores_json",
    "discriminator_run2_borda_ranking_json",
    "discriminator_run2_winner_id",
    "discriminator_run2_winner_source",
    "discriminator_run2_winner_original_id",
    "discriminator_run2_winner_pun",
    "discriminator_run2_winner_score",
    "discriminator_run2_borda_error",
    "discriminator_run2_borda_version",
]


def discover_judge_models(ensemble_run: str) -> list[str]:
    base = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}"
    dirs = [p for p in glob.glob(base + "*/") if os.path.isdir(p)]
    judges: list[str] = []
    for path in dirs:
        name = Path(path.rstrip("/")).name
        if name in {"borda", "metrics", "reports", "analysis"}:
            continue
        has_numeric_chunk = any(Path(tsv).stem.isdigit() for tsv in glob.glob(ensure_slash(path) + "*.tsv"))
        if has_numeric_chunk:
            judges.append(name)
    return sorted(judges)


def selected_run2_chunks(ensemble_run: str, judges: list[str], start: int, end: int) -> list[int]:
    chunk_sets: list[set[int]] = []
    for judge in judges:
        input_dir = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}{ensure_slash(judge)}"
        chunks = set(chunk_numbers_in_dir(input_dir))
        if chunks:
            chunk_sets.append(chunks)
    if not chunk_sets:
        return []
    available = sorted(set.union(*chunk_sets))
    return [c for c in available if c >= start and (end == -1 or c < end)]


def parse_weight_string(weight_string: str, names: list[str], label: str, allow_extra: bool = False) -> dict[str, float]:
    parts = [p for p in str(weight_string).split("_") if p != ""]
    try:
        values = [float(p.replace("p", ".")) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid {label} weight string {weight_string!r}; expected numbers joined by underscores") from exc

    if len(values) < len(names):
        raise ValueError(
            f"{label} weight string {weight_string!r} has {len(values)} values but needs at least {len(names)} "
            f"for: {', '.join(names)}"
        )
    if len(values) > len(names) and not allow_extra:
        raise ValueError(
            f"{label} weight string {weight_string!r} has {len(values)} values but needs exactly {len(names)} "
            f"for: {', '.join(names)}"
        )
    if len(values) > len(names) and allow_extra:
        print(
            f"Note: {label} weight string has {len(values)} values but only {len(names)} models are active; "
            f"using the first {len(names)} values and ignoring extras."
        )
        values = values[: len(names)]
    return dict(zip(names, values))


def format_weight_key(weights: dict[str, float], names: list[str]) -> str:
    return "_".join(format_weight_value(weights[name]) for name in names)


def validate_run2_rankings(rankings_raw: Any, valid_ids: set[int]) -> dict[str, list[int]]:
    if not isinstance(rankings_raw, dict):
        raise ValueError("discriminator_run2_json must be a JSON object")
    return validate_response(pd.Series(rankings_raw), valid_ids)


def borda_rank(
    rankings: list[list[int]],
    weights: list[float] | None = None,
) -> tuple[dict[int, float], list[int]]:
    if not rankings:
        return {}, []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")

    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        n = len(ranking)
        for pos, cid in enumerate(ranking):
            cid = int(cid)
            scores[cid] = scores.get(cid, 0.0) + float(weight) * (n - pos)
    ranked_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return scores, ranked_ids


def load_run2_chunk_for_cross_borda(ensemble_run: str, judge_model: str, chunk_num: int) -> pd.DataFrame:
    path = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}{ensure_slash(judge_model)}{chunk_num}.tsv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Run 2 chunk: {path}")
    df = load(path)
    missing = [c for c in ["id_en", "shuffled_candidates_json", "discriminator_run2_json"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    return df


def row_output_base(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id_en": row.get("id_en", ""),
        "text_clean": row.get("text_clean", ""),
        "shuffled_candidates_json": row.get("shuffled_candidates_json", ""),
    }
    return out


def ranking_json_from_scores(ranked_ids: list[int], scores: dict[int, float], candidates_by_id: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": cid,
            "source": candidates_by_id[cid].get("source", ""),
            "original_id": candidates_by_id[cid].get("original_id", ""),
            "score": scores[cid],
            "pun": candidates_by_id[cid].get("pun", ""),
        }
        for cid in ranked_ids
    ]


def cross_borda_output_row(
    method: str,
    internal_weights: dict[str, float],
    model_weights: dict[str, float],
    candidates_by_id: dict[int, dict[str, str]],
    scores: dict[int, float],
    ranked_ids: list[int],
    error: str = "",
) -> pd.Series:
    scores_sorted = {str(cid): scores[cid] for cid in ranked_ids}
    ranking_json = ranking_json_from_scores(ranked_ids, scores, candidates_by_id) if not error else []
    winner_id = ranked_ids[0] if ranked_ids else ""
    winner_meta = candidates_by_id[int(winner_id)] if winner_id != "" else {}
    winner_score = scores.get(int(winner_id), 0.0) if winner_id != "" else ""
    return pd.Series({
        "discriminator_run2_borda_method": method,
        "discriminator_run2_internal_weights_json": json.dumps(internal_weights, ensure_ascii=False, separators=(",", ":")),
        "discriminator_run2_model_weights_json": json.dumps(model_weights, ensure_ascii=False, separators=(",", ":")),
        "discriminator_run2_borda_scores_json": json.dumps(scores_sorted, ensure_ascii=False, separators=(",", ":")),
        "discriminator_run2_borda_ranking_json": json.dumps(ranking_json, ensure_ascii=False, separators=(",", ":")),
        "discriminator_run2_winner_id": winner_id,
        "discriminator_run2_winner_source": winner_meta.get("source", ""),
        "discriminator_run2_winner_original_id": winner_meta.get("original_id", ""),
        "discriminator_run2_winner_pun": winner_meta.get("pun", ""),
        "discriminator_run2_winner_score": winner_score,
        "discriminator_run2_borda_error": error,
        "discriminator_run2_borda_version": RUN2_VERSION,
    })



def fallback_random_ranking(valid_ids: set[int], row_id: Any, judge_model: str, judge_key: str) -> list[int]:
    """Deterministic random ranking used when one judge-model row is missing or invalid."""
    ids = sorted(int(x) for x in valid_ids)
    seed_text = f"{SHUFFLE_SEED}|fallback_random|{row_id}|{judge_model}|{judge_key}"
    seed_int = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed_int)
    rng.shuffle(ids)
    return ids


def fallback_rankings_for_model(valid_ids: set[int], row_id: Any, judge_model: str) -> dict[str, list[int]]:
    return {
        judge_key: fallback_random_ranking(valid_ids, row_id, judge_model, judge_key)
        for judge_key in JUDGE_KEYS
    }


def calculate_cross_borda_rows(
    base_row: pd.Series,
    rows_by_model: dict[str, pd.Series],
    judges: list[str],
    internal_weights: dict[str, float],
    model_weights: dict[str, float],
) -> dict[str, pd.Series]:
    row_id = base_row.get("id_en", base_row.name)
    fallback_notes: list[str] = []
    try:
        candidates_by_id = candidate_map_from_row(base_row)
        valid_ids = set(candidates_by_id)

        rankings_by_model: dict[str, dict[str, list[int]]] = {}
        for judge_model in judges:
            row = rows_by_model.get(judge_model)
            if row is None:
                rankings_by_model[judge_model] = fallback_rankings_for_model(valid_ids, row_id, judge_model)
                fallback_notes.append(f"{judge_model}:missing_row_random_fallback")
                continue

            try:
                if norm_space(row.get("discriminator_run2_error", "")):
                    raise ValueError(f"Run 2 error: {row.get('discriminator_run2_error')}")
                rankings_raw = safe_json_loads(row.get("discriminator_run2_json", ""))
                rankings_by_model[judge_model] = validate_run2_rankings(rankings_raw, valid_ids)
            except Exception as e:
                rankings_by_model[judge_model] = fallback_rankings_for_model(valid_ids, row_id, judge_model)
                fallback_notes.append(f"{judge_model}:invalid_or_error_random_fallback:{e}")

        # Method 1: first aggregate internal judge personas within each model judge,
        # then aggregate the resulting model-level rankings.
        model_level_rankings: list[list[int]] = []
        model_level_weights: list[float] = []
        for judge_model in judges:
            internal_rankings = [rankings_by_model[judge_model][judge_key] for judge_key in JUDGE_KEYS]
            internal_weight_values = [internal_weights[judge_key] for judge_key in JUDGE_KEYS]
            _, ranked_ids = borda_rank(internal_rankings, internal_weight_values)
            model_level_rankings.append(ranked_ids)
            model_level_weights.append(model_weights[judge_model])
        judges_then_scores, judges_then_ranked = borda_rank(model_level_rankings, model_level_weights)

        # Method 2: first aggregate model judges within each internal judge persona,
        # then aggregate the resulting persona-level rankings.
        persona_level_rankings: list[list[int]] = []
        persona_level_weights: list[float] = []
        for judge_key in JUDGE_KEYS:
            model_rankings = [rankings_by_model[judge_model][judge_key] for judge_model in judges]
            model_weight_values = [model_weights[judge_model] for judge_model in judges]
            _, ranked_ids = borda_rank(model_rankings, model_weight_values)
            persona_level_rankings.append(ranked_ids)
            persona_level_weights.append(internal_weights[judge_key])
        models_then_scores, models_then_ranked = borda_rank(persona_level_rankings, persona_level_weights)

        # Method 3: pool every judge-model × internal-judge ranking at once.
        pooled_rankings: list[list[int]] = []
        pooled_weights: list[float] = []
        for judge_model in judges:
            for judge_key in JUDGE_KEYS:
                pooled_rankings.append(rankings_by_model[judge_model][judge_key])
                pooled_weights.append(model_weights[judge_model] * internal_weights[judge_key])
        pooled_scores, pooled_ranked = borda_rank(pooled_rankings, pooled_weights)

        # This is an informational note, not a fatal Borda error. The row still gets a winner.
        error_note = ";".join(fallback_notes)

        return {
            "judges_then_models": cross_borda_output_row(
                "judges_then_models", internal_weights, model_weights, candidates_by_id, judges_then_scores, judges_then_ranked, error_note
            ),
            "models_then_judges": cross_borda_output_row(
                "models_then_judges", internal_weights, model_weights, candidates_by_id, models_then_scores, models_then_ranked, error_note
            ),
            "pooled_rankings": cross_borda_output_row(
                "pooled_rankings", internal_weights, model_weights, candidates_by_id, pooled_scores, pooled_ranked, error_note
            ),
        }
    except Exception as e:
        print(f"Cross-model Borda error id_en={row_id}: {e}")
        try:
            candidates_by_id = candidate_map_from_row(base_row)
        except Exception:
            candidates_by_id = {}
        return {
            method: cross_borda_output_row(method, internal_weights, model_weights, candidates_by_id, {}, [], str(e))
            for method in RUN2_BORDA_METHODS
        }


def run_borda(
    ensemble_run: str,
    judges: list[str],
    internal_weights: dict[str, float],
    model_weights: dict[str, float],
    start: int = 0,
    end: int = -1,
) -> None:
    chunks = selected_run2_chunks(ensemble_run, judges, start, end)
    if not chunks:
        raise FileNotFoundError(
            f"No raw Run 2 chunks found for ensemble_run={ensemble_run}, judges={judges}, start={start}, end={end}"
        )

    internal_weight_key = format_weight_key(internal_weights, JUDGE_KEYS)
    model_weight_key = format_weight_key(model_weights, judges)

    log("Cross-model Borda input Run 2:", f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}")
    log("Ensemble run:", ensemble_run)
    log("Judge models:", ", ".join(judges))
    log("Internal weights:", internal_weights)
    log("Model weights:", model_weights)
    log("Internal weight key:", internal_weight_key)
    log("Model weight key:", model_weight_key)
    log("Output methods:", ", ".join(RUN2_BORDA_METHODS))
    log("Missing/invalid judge-model row policy:", "deterministic random ranking fallback")
    log("Chunks:", chunks)

    for chunk_num in chunks:
        dfs_by_model: dict[str, pd.DataFrame] = {}
        for judge_model in judges:
            try:
                dfs_by_model[judge_model] = load_run2_chunk_for_cross_borda(ensemble_run, judge_model, chunk_num)
            except FileNotFoundError as e:
                print(f"Missing Run 2 chunk for judge_model={judge_model}, chunk={chunk_num}; using random fallback for all rows from this model. {e}")
                dfs_by_model[judge_model] = pd.DataFrame()

        non_empty = [df for df in dfs_by_model.values() if not df.empty]
        if not non_empty:
            print(f"Skipping chunk {chunk_num}: no judge-model inputs found")
            continue

        # Use the union of id_en values across present judge-model files, so a row can be
        # produced even when one or more judge models are missing that id_en.
        id_order: list[str] = []
        seen_ids: set[str] = set()
        rows_lookup: dict[str, dict[str, pd.Series]] = {}
        base_lookup: dict[str, pd.Series] = {}

        for judge_model in judges:
            df = dfs_by_model[judge_model]
            if df.empty:
                continue
            for _, row in df.iterrows():
                id_en = norm_space(row.get("id_en", ""))
                if not id_en:
                    continue
                if id_en not in seen_ids:
                    seen_ids.add(id_en)
                    id_order.append(id_en)
                    base_lookup[id_en] = row
                rows_lookup.setdefault(id_en, {})[judge_model] = row

        method_rows: dict[str, list[pd.Series]] = {method: [] for method in RUN2_BORDA_METHODS}
        out_base_rows: list[dict[str, Any]] = []

        for out_index, id_en in enumerate(id_order):
            base_row = base_lookup[id_en]
            rows_by_model = rows_lookup.get(id_en, {})
            outputs = calculate_cross_borda_rows(base_row, rows_by_model, judges, internal_weights, model_weights)
            out_base_rows.append(row_output_base(base_row))
            for method in RUN2_BORDA_METHODS:
                method_rows[method].append(outputs[method])

            if VERBOSE:
                winners = {method: outputs[method].get("discriminator_run2_winner_source", "") for method in RUN2_BORDA_METHODS}
                notes = outputs["pooled_rankings"].get("discriminator_run2_borda_error", "")
                print(out_index, id_en, "winners=", winners, "notes=", notes)

        base_out_df = pd.DataFrame(out_base_rows)
        for method in RUN2_BORDA_METHODS:
            out_df = base_out_df.copy()
            method_df = pd.DataFrame(method_rows[method])
            for col in CROSS_BORDA_OUTPUT_COLUMNS:
                out_df[col] = method_df[col]
            out_path = (
                f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}borda/"
                f"{ensure_slash(method)}{ensure_slash(internal_weight_key)}{ensure_slash(model_weight_key)}{chunk_num}.tsv"
            )
            save(out_df, out_path)


async def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("""Usage:
  python discriminator_run2_v4.py run <ensemble_run> <judge_model> <start> <end>
  python discriminator_run2_v4.py borda <ensemble_run> <internal_weights> <model_weights> <start> <end>
  python discriminator_run2_v4.py borda <ensemble_run> <judge_models_csv> <internal_weights> <model_weights> <start> <end>

Examples:
  python discriminator_run2_v4.py run ensemble gpt 0 1
  python discriminator_run2_v4.py run ensemble claude 0 -1
  python discriminator_run2_v4.py borda ensemble 25_25_25_25 25_25_25 33 34
  python discriminator_run2_v4.py borda ensemble claude,gemini_pro,gpt 45_30_15_10 33_33_34 0 -1

Borda outputs, one directory per aggregation method:
  ../data/processed/discriminate/run2/{ensemble_run}/borda/judges_then_models/{internal_weights}/{model_weights}/{chunk}.tsv
  ../data/processed/discriminate/run2/{ensemble_run}/borda/models_then_judges/{internal_weights}/{model_weights}/{chunk}.tsv
  ../data/processed/discriminate/run2/{ensemble_run}/borda/pooled_rankings/{internal_weights}/{model_weights}/{chunk}.tsv

Weight order:
  internal_weights: comedian_pun_expert_editor_translator
  model_weights: discovered/provided judge model order printed at runtime
  missing/invalid judge-model rows use deterministic random-ranking fallback
""")

    task = sys.argv[1]
    if task == "run":
        ensemble_run = sys.argv[2] if len(sys.argv) > 2 else "ensemble"
        judge_model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_JUDGE_MODEL
        start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        end = int(sys.argv[5]) if len(sys.argv) > 5 else -1
        await run_discriminator(ensemble_run, judge_model, start, end)
        return

    if task == "borda":
        ensemble_run = sys.argv[2] if len(sys.argv) > 2 else "ensemble"

        # Forms:
        #   borda ensemble internal_weights model_weights start end
        #   borda ensemble judges_csv internal_weights model_weights start end
        if len(sys.argv) >= 8 and "," in sys.argv[3]:
            judges = [resolve_model_alias(x.strip())[0] for x in sys.argv[3].split(",") if x.strip()]
            internal_weight_string = sys.argv[4]
            model_weight_string = sys.argv[5]
            start = int(sys.argv[6])
            end = int(sys.argv[7])
        elif len(sys.argv) >= 7:
            judges = discover_judge_models(ensemble_run)
            internal_weight_string = sys.argv[3]
            model_weight_string = sys.argv[4]
            start = int(sys.argv[5])
            end = int(sys.argv[6])
        else:
            raise ValueError(
                "borda usage: python discriminator_run2_v4.py borda <ensemble_run> <internal_weights> <model_weights> <start> <end>\n"
                "or:          python discriminator_run2_v4.py borda <ensemble_run> <judge_models_csv> <internal_weights> <model_weights> <start> <end>"
            )

        if not judges:
            raise FileNotFoundError(f"No judge model directories found under {ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}")

        internal_weights = parse_weight_string(internal_weight_string, JUDGE_KEYS, "internal judge", allow_extra=False)
        model_weights = parse_weight_string(model_weight_string, judges, "judge model", allow_extra=True)
        run_borda(ensemble_run, judges, internal_weights, model_weights, start, end)
        return

    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    asyncio.run(main())
