"""
JOKER shuffle builder with per-candidate IDs and target selection.

This step does NOT call an LLM and does NOT evaluate candidates.
It loads generated candidate TSVs from configured generator output runs,
extracts only generated French pun strings, attaches the run label, adds a
random-looking deterministic 5-digit candidate id, shuffles candidates per id_en,
and writes chunked TSVs.

Targets:
  ensemble   use all configured runs and save to data/processed/shuffle/ensemble/
  <run>      use exactly one configured run and save to data/processed/shuffle/<run>/

Configured runs come from config.py / config.ini via SHUFFLE_ENSEMBLE_RUNS.
Default configured runs are usually:
  data/processed/generate/claude            -> claude
  data/processed/generate/gemini_pro        -> gemini_pro
  data/processed/generate_single/gpt        -> gpt_single
  data/processed/generate_single/gemini_pro -> gemini_pro_single

Output columns:
  id_en
  shuffled_candidates_json

Each shuffled candidate is exactly:
  {"id": 28334, "pun": "...", "run": "..."}

IDs are:
  - integers
  - 5 digits, 10000 through 99999
  - unique within each id_en row
  - deterministic for the same target, row, seed, and candidate pool

Usage:
  python shuffle_targeted_with_ids.py shuffle ensemble 0 -1
  python shuffle_targeted_with_ids.py shuffle claude 0 1
  python shuffle_targeted_with_ids.py shuffle gemini_pro 0 -1
  python shuffle_targeted_with_ids.py shuffle gpt_single 0 -1

Environment variables:
  SHUFFLE_ENSEMBLE_CHUNK_SIZE     output chunk size, default 100
  SHUFFLE_ENSEMBLE_RANDOM_SEED    base shuffle seed, default 20260529
  SHUFFLE_ENSEMBLE_VERBOSE        1/0 logging, default 1
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
from typing import Any

import pandas as pd

from config import SHUFFLE_ENSEMBLE_RUNS, shuffle_dir, shuffle_ensemble_dir
from data import load_all, save

pd.options.mode.chained_assignment = None

SHUFFLE_VERSION = "shuffle_targeted_with_ids_v1"
VERBOSE = os.environ.get("SHUFFLE_ENSEMBLE_VERBOSE", "1") == "1"
CHUNK_SIZE = int(os.environ.get("SHUFFLE_ENSEMBLE_CHUNK_SIZE", "100"))
RANDOM_SEED = int(os.environ.get("SHUFFLE_ENSEMBLE_RANDOM_SEED", "20260529"))

OUTPUT_COLUMNS = ["id_en", "shuffled_candidates_json"]
MIN_CANDIDATE_ID = 10000
MAX_CANDIDATE_ID = 99999
ENSEMBLE_TARGET = "ensemble"


def log(*args: Any) -> None:
    if VERBOSE:
        print(*args)


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def ensure_slash(path: str) -> str:
    path = norm_space(path)
    return path.rstrip("/") + "/" if path else path


def safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (dict, list)):
        return value

    text = str(value).strip()
    if not text:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return None


def extract_candidate_puns(candidate_json: Any) -> list[str]:
    """Extract only generated pun strings from candidate_json.

    Generator outputs normally store a JSON list of candidate objects in
    candidate_json, with the pun text in the "french" field. This function
    intentionally ignores all candidate metadata.
    """
    parsed = safe_json_loads(candidate_json)
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        candidates = parsed.get("candidates") or parsed.get("candidate") or []
    else:
        candidates = parsed

    if not isinstance(candidates, list):
        candidates = [candidates]

    puns: list[str] = []
    for candidate in candidates:
        pun = ""
        if isinstance(candidate, dict):
            for key in ("french", "pun", "candidate", "text", "sentence", "output"):
                pun = norm_space(candidate.get(key, ""))
                if pun:
                    break
        else:
            pun = norm_space(candidate)

        if pun:
            puns.append(pun)

    return puns


def stable_row_seed(target: str, id_en: Any) -> int:
    """Use target in seed so ensemble and single-run shuffles are independent."""
    text = norm_space(id_en)
    target_offset = sum(ord(ch) for ch in norm_space(target))
    try:
        return RANDOM_SEED + target_offset + int(text)
    except Exception:
        return RANDOM_SEED + target_offset + sum(ord(ch) for ch in text)


def next_unique_five_digit_id(rng: random.Random, used_ids: set[int]) -> int:
    """Return a random 5-digit integer unique within the current row."""
    capacity = MAX_CANDIDATE_ID - MIN_CANDIDATE_ID + 1
    if len(used_ids) >= capacity:
        raise ValueError("Cannot assign unique 5-digit IDs: row has more than 90000 candidates")

    while True:
        candidate_id = rng.randint(MIN_CANDIDATE_ID, MAX_CANDIDATE_ID)
        if candidate_id not in used_ids:
            used_ids.add(candidate_id)
            return candidate_id


def shuffle_candidate_objects(target: str, id_en: Any, candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Clean, shuffle, and assign deterministic per-row 5-digit IDs.

    No deduplication is performed. If two runs produce the same pun, both
    candidates are preserved as separate objects with different IDs.
    """
    cleaned: list[dict[str, str]] = []
    for candidate in candidates:
        pun = norm_space(candidate.get("pun", ""))
        run = norm_space(candidate.get("run", ""))
        if pun and run:
            cleaned.append({"pun": pun, "run": run})

    rng = random.Random(stable_row_seed(target, id_en))
    rng.shuffle(cleaned)

    used_ids: set[int] = set()
    with_ids: list[dict[str, Any]] = []
    for candidate in cleaned:
        with_ids.append({
            "id": next_unique_five_digit_id(rng, used_ids),
            "pun": candidate["pun"],
            "run": candidate["run"],
        })

    return with_ids


def load_run_candidates(run_label: str, source_dir: str) -> pd.DataFrame:
    source_dir = ensure_slash(source_dir)
    log("Loading shuffle source:", run_label, source_dir.rstrip("/"))

    df = load_all(source_dir)

    if "id_en" not in df.columns:
        raise ValueError(f"Missing id_en column in {source_dir}")
    if "candidate_json" not in df.columns:
        raise ValueError(f"Missing candidate_json column in {source_dir}")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        id_en = norm_space(row.get("id_en", ""))
        if not id_en:
            continue

        puns = extract_candidate_puns(row.get("candidate_json", ""))
        candidate_objects = [{"pun": pun, "run": run_label} for pun in puns]
        rows.append({
            "id_en": id_en,
            "candidates": candidate_objects,
            "candidate_count_extracted": len(candidate_objects),
        })

    out = pd.DataFrame(rows)
    total = int(out["candidate_count_extracted"].sum()) if len(out) else 0
    log("Loaded source rows:", len(out), "candidates:", total)
    return out


def build_shuffled_rows(target: str, runs: list[tuple[str, str]]) -> pd.DataFrame:
    if not runs:
        raise ValueError("No shuffle runs configured")

    run_frames = [load_run_candidates(run_label, source_dir) for run_label, source_dir in runs]
    all_rows = pd.concat(run_frames, ignore_index=True)

    grouped: dict[str, list[dict[str, str]]] = {}
    for _, row in all_rows.iterrows():
        id_en = norm_space(row.get("id_en", ""))
        if not id_en:
            continue
        grouped.setdefault(id_en, []).extend(row.get("candidates", []) or [])

    output_rows: list[dict[str, str]] = []
    for id_en in sorted(grouped.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        shuffled = shuffle_candidate_objects(target, id_en, grouped[id_en])
        output_rows.append({
            "id_en": id_en,
            "shuffled_candidates_json": json.dumps(shuffled, ensure_ascii=False, separators=(",", ":")),
        })

    out = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    total_candidates = sum(len(safe_json_loads(v) or []) for v in out["shuffled_candidates_json"])
    log("Built shuffled rows:", len(out))
    log("Total shuffled candidates:", total_candidates)
    return out


def save_chunked(df: pd.DataFrame, output_dir: str, start: int = 0, end: int = -1) -> None:
    output_dir = ensure_slash(output_dir)
    chunks = [df.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
    end = len(chunks) if end == -1 else end

    for i in range(start, end):
        if i < 0 or i >= len(chunks):
            raise IndexError(f"Chunk index {i} out of range 0:{len(chunks)}")
        save(chunks[i], f"{output_dir}{i}.tsv")


def normalize_target(target: str) -> str:
    target = norm_space(target)
    if not target:
        raise ValueError("Missing shuffle target. Use 'ensemble' or a configured run label, e.g. claude.")
    return target


def resolve_runs_for_target(target: str) -> list[tuple[str, str]]:
    if target == ENSEMBLE_TARGET:
        return list(SHUFFLE_ENSEMBLE_RUNS)

    matches = [(run_label, source_dir) for run_label, source_dir in SHUFFLE_ENSEMBLE_RUNS if run_label == target]
    if matches:
        return matches

    available = ", ".join([ENSEMBLE_TARGET] + [run_label for run_label, _ in SHUFFLE_ENSEMBLE_RUNS])
    raise ValueError(f"Unknown shuffle target '{target}'. Available targets: {available}")


def resolve_output_dir_for_target(target: str) -> str:
    if target == ENSEMBLE_TARGET:
        return shuffle_ensemble_dir
    return f"{ensure_slash(shuffle_dir)}{target}/"


def shuffle(target: str, start: int = 0, end: int = -1) -> None:
    target = normalize_target(target)
    runs = resolve_runs_for_target(target)
    output_dir = resolve_output_dir_for_target(target)

    log("Shuffle step:", SHUFFLE_VERSION)
    log("Target:", target)
    log("Configured runs:")
    for run_label, source_dir in runs:
        log(" -", run_label, ensure_slash(source_dir).rstrip("/"))
    log("Output dir:", ensure_slash(output_dir).rstrip("/"))
    log("Chunk size:", CHUNK_SIZE)
    log("Random seed:", RANDOM_SEED)
    log("Dedupe: disabled")
    log("Candidate IDs: enabled, unique 5-digit IDs within each row")

    df = build_shuffled_rows(target, runs)
    save_chunked(df, output_dir, start, end)


def main() -> None:
    if len(sys.argv) < 3:
        raise ValueError("""Usage:
  python shuffle_target.py shuffle <ensemble|run_label> <start> <end>

Examples:
  python shuffle_target.py shuffle ensemble 0 -1
  python shuffle_target.py shuffle claude 0 1
  python shuffle_target.py shuffle gemini_pro 0 -1
  python shuffle_target.py shuffle gpt_single 0 -1
""")

    task = sys.argv[1]
    if task != "shuffle":
        raise ValueError(f"Unknown task: {task}")

    target = sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    shuffle(target, start, end)


if __name__ == "__main__":
    main()
