"""
JOKER discriminator ensemble-prep step.

This first discriminator step does NOT call an LLM and does NOT use candidate
metadata. It loads generated candidate TSVs from multiple generator output
sources, extracts only the generated French pun strings, shuffles the pooled
candidates per id_en, and writes chunked TSVs.

Default input sources:
  data/processed/generate/claude
  data/processed/generate/gemini_pro
  data/processed/generate_single/gpt
  data/processed/generate_single/gemini_pro

Default output:
  data/processed/generator/ensemble/shuffled/{chunk}.tsv

Output columns:
  id_en
  shuffled_candidates_json

Usage:
  python discriminator.py shuffle 0 1
  python discriminator.py shuffle 0 -1

Configurable environment variables:
  DISCRIMINATOR_INPUT_SOURCES   comma-separated list of input directories
  DISCRIMINATOR_OUTPUT_DIR      output directory for shuffled chunks
  DISCRIMINATOR_CHUNK_SIZE      output chunk size, default 100
  DISCRIMINATOR_RANDOM_SEED     base shuffle seed, default 20260529
  DISCRIMINATOR_DEDUPE          1/0 dedupe candidates within each row, default 1
  DISCRIMINATOR_VERBOSE         1/0 logging, default 1
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from config import generate_dir, generate_single_dir
except Exception:
    generate_dir = "../data/processed/generate/"
    generate_single_dir = "../data/processed/generate_single/"

from data import load_all, save

pd.options.mode.chained_assignment = None

DISCRIMINATOR_VERSION = "shuffle_v1"
VERBOSE = os.environ.get("DISCRIMINATOR_VERBOSE", "1") == "1"
CHUNK_SIZE = int(os.environ.get("DISCRIMINATOR_CHUNK_SIZE", "100"))
RANDOM_SEED = int(os.environ.get("DISCRIMINATOR_RANDOM_SEED", "20260529"))
DEDUPE = os.environ.get("DISCRIMINATOR_DEDUPE", "1") == "1"

OUTPUT_COLUMNS = ["id_en", "shuffled_candidates_json"]


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


def extract_candidate_texts(candidate_json: Any) -> list[str]:
    """Extract only generated pun strings from candidate_json.

    Regular generator runs store a JSON list of candidate objects, usually with
    a "french" field. Single runs should do the same but normally contain one
    candidate. This function intentionally discards all metadata fields.
    """
    parsed = safe_json_loads(candidate_json)
    if parsed is None:
        return []

    # Some files may wrap candidates in an object.
    if isinstance(parsed, dict):
        candidates = parsed.get("candidates") or parsed.get("candidate") or []
    else:
        candidates = parsed

    if not isinstance(candidates, list):
        candidates = [candidates]

    out: list[str] = []
    for candidate in candidates:
        text = ""
        if isinstance(candidate, dict):
            # Keep this order conservative: current generator uses "french".
            for key in ("french", "pun", "candidate", "text", "sentence", "output"):
                text = norm_space(candidate.get(key, ""))
                if text:
                    break
        else:
            text = norm_space(candidate)

        if text:
            out.append(text)

    return out


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = norm_space(value).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(norm_space(value))
    return out


def stable_row_seed(id_en: Any) -> int:
    text = norm_space(id_en)
    try:
        return RANDOM_SEED + int(text)
    except Exception:
        return RANDOM_SEED + sum(ord(ch) for ch in text)


def shuffle_candidates_for_row(id_en: Any, candidates: list[str]) -> list[str]:
    candidates = dedupe_keep_order(candidates) if DEDUPE else [norm_space(c) for c in candidates if norm_space(c)]
    rng = random.Random(stable_row_seed(id_en))
    rng.shuffle(candidates)
    return candidates


def default_input_sources() -> list[str]:
    return [
        f"{ensure_slash(generate_dir)}claude/",
        f"{ensure_slash(generate_dir)}gemini_pro/",
        f"{ensure_slash(generate_single_dir)}gpt/",
        f"{ensure_slash(generate_single_dir)}gemini_pro/",
    ]


def resolve_input_sources() -> list[str]:
    explicit = os.environ.get("DISCRIMINATOR_INPUT_SOURCES", "").strip()
    if explicit:
        return [ensure_slash(p) for p in explicit.split(",") if norm_space(p)]
    return default_input_sources()


def resolve_output_dir() -> str:
    explicit = os.environ.get("DISCRIMINATOR_OUTPUT_DIR", "").strip()
    if explicit:
        return ensure_slash(explicit)

    # User requested generator/ensemble/shuffled, not generate/ensemble/shuffled.
    root = Path(__file__).resolve().parents[1]
    return str(root / "data" / "processed" / "generator" / "ensemble" / "shuffled") + "/"


def load_source_candidates(source_dir: str) -> pd.DataFrame:
    log("Loading discriminator source:", source_dir.rstrip("/"))
    df = load_all(source_dir)

    if "id_en" not in df.columns:
        raise ValueError(f"Missing id_en column in {source_dir}")
    if "candidate_json" not in df.columns:
        raise ValueError(f"Missing candidate_json column in {source_dir}")

    rows: list[dict[str, Any]] = []
    source_name = Path(source_dir.rstrip("/")).name

    for _, row in df.iterrows():
        id_en = norm_space(row.get("id_en", ""))
        if not id_en:
            continue
        candidates = extract_candidate_texts(row.get("candidate_json", ""))
        rows.append({
            "id_en": id_en,
            "source": source_name,
            "candidates": candidates,
            "candidate_count_extracted": len(candidates),
        })

    out = pd.DataFrame(rows)
    log(
        "Loaded source rows:", len(out),
        "candidates:", int(out["candidate_count_extracted"].sum()) if len(out) else 0,
    )
    return out


def build_shuffled_ensemble(input_sources: list[str]) -> pd.DataFrame:
    source_frames = [load_source_candidates(source_dir) for source_dir in input_sources]
    if not source_frames:
        raise ValueError("No discriminator input sources configured")

    all_rows = pd.concat(source_frames, ignore_index=True)
    grouped: dict[str, list[str]] = {}

    for _, row in all_rows.iterrows():
        id_en = norm_space(row.get("id_en", ""))
        if not id_en:
            continue
        grouped.setdefault(id_en, []).extend(row.get("candidates", []) or [])

    output_rows: list[dict[str, str]] = []
    for id_en in sorted(grouped.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        shuffled = shuffle_candidates_for_row(id_en, grouped[id_en])
        output_rows.append({
            "id_en": id_en,
            "shuffled_candidates_json": json.dumps(shuffled, ensure_ascii=False, separators=(",", ":")),
        })

    out = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    log("Built shuffled ensemble rows:", len(out))
    log("Total shuffled candidates:", sum(len(safe_json_loads(v) or []) for v in out["shuffled_candidates_json"]))
    return out


def save_chunked(df: pd.DataFrame, output_dir: str, start: int = 0, end: int = -1) -> None:
    chunks = [df.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
    end = len(chunks) if end == -1 else end

    for i in range(start, end):
        if i < 0 or i >= len(chunks):
            raise IndexError(f"Chunk index {i} out of range 0:{len(chunks)}")
        out_path = f"{ensure_slash(output_dir)}{i}.tsv"
        save(chunks[i], out_path)


def shuffle(start: int = 0, end: int = -1) -> None:
    input_sources = resolve_input_sources()
    output_dir = resolve_output_dir()

    log("Discriminator step:", DISCRIMINATOR_VERSION)
    log("Input sources:")
    for source in input_sources:
        log(" -", source.rstrip("/"))
    log("Output dir:", output_dir.rstrip("/"))
    log("Chunk size:", CHUNK_SIZE)
    log("Random seed:", RANDOM_SEED)
    log("Dedupe:", DEDUPE)

    df = build_shuffled_ensemble(input_sources)
    save_chunked(df, output_dir, start, end)


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("""Usage:
  python discriminator.py shuffle <start> <end>

Examples:
  python discriminator.py shuffle 0 1
  python discriminator.py shuffle 0 -1

Environment configuration:
  DISCRIMINATOR_INPUT_SOURCES=../data/processed/generate/claude/,../data/processed/generate/gemini_pro/,../data/processed/generate_single/gpt/,../data/processed/generate_single/gemini_pro/
  DISCRIMINATOR_OUTPUT_DIR=../data/processed/generator/ensemble/shuffled/
""")

    task = sys.argv[1]
    if task != "shuffle":
        raise ValueError(f"Unknown task: {task}")

    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    shuffle(start, end)


if __name__ == "__main__":
    main()
