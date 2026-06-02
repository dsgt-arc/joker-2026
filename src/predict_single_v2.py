"""
Build JOKER task 2 prediction.json from single-candidate generator outputs.

Usage:
  python predict_single_v2.py <run_id> <model>

Example:
  python predict_single_v2.py UBO_task_3_claude claude

Inputs:
  - data/2026/joker_task2_en_fr_2026_test.json via config.translation_path
  - data/processed/generate_single/<model>/*.tsv via config.generate_single_dir

Outputs:
  - data/processed/generate_single/<model>.tsv         combined chunk file
  - data/processed/predict/<run_id>/prediction.json   submission file
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config import generate_single_dir, translation_path
from data import load, load_all, save


MANUAL_FLAG = int(os.environ.get("PREDICTION_MANUAL", "0"))


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def ensure_slash(path: str) -> str:
    return str(path).rstrip("/") + "/"


def resolve_generate_single_model_dir(model: str) -> str:
    return f"{ensure_slash(generate_single_dir)}{model}/"


def resolve_combined_output_path(model: str) -> str:
    return f"{ensure_slash(generate_single_dir).rstrip('/')}/{model}.tsv"


def resolve_predict_dir() -> str:
    """
    Derive ../data/processed/predict from config.generate_single_dir.

    This keeps the same project path infrastructure without requiring a new
    config.py variable. For the current project layout:
      ../data/processed/generate_single -> ../data/processed/predict
    """
    processed_dir = Path(generate_single_dir).parent
    return str(processed_dir / "predict")


def resolve_prediction_path(run_id: str) -> str:
    return str(Path(resolve_predict_dir()) / run_id / "prediction.json")


def parse_maybe_json(value: Any) -> Any:
    if value is None:
        return None
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

    return text


def extract_fr_from_candidate_json(value: Any) -> str:
    """
    Supports generator_single output formats:
      - [{"french": "..."}]
      - {"candidates": [{"french": "..."}]}
      - {"french": "..."}
      - {"fr": "..."}
      - direct string fallback
    """
    parsed = parse_maybe_json(value)

    if isinstance(parsed, dict):
        if norm_space(parsed.get("french")):
            return norm_space(parsed.get("french"))
        if norm_space(parsed.get("fr")):
            return norm_space(parsed.get("fr"))
        candidates = parsed.get("candidates")
        if isinstance(candidates, list) and candidates:
            return extract_fr_from_candidate_json(candidates[0])
        return ""

    if isinstance(parsed, list):
        for item in parsed:
            fr = extract_fr_from_candidate_json(item)
            if fr:
                return fr
        return ""

    return norm_space(parsed)


def pick_fr(row: pd.Series) -> str:
    if "candidate_json" in row.index:
        fr = extract_fr_from_candidate_json(row.get("candidate_json"))
        if fr:
            return fr

    for col in ("fr", "french", "translation", "prediction"):
        if col in row.index and norm_space(row.get(col)):
            return norm_space(row.get(col))

    return ""


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def build_prediction(run_id: str, model: str) -> list[dict[str, Any]]:
    model_dir = resolve_generate_single_model_dir(model)
    combined_path = resolve_combined_output_path(model)

    print(f"Loading single-generator chunks: {model_dir}")
    generated_df = load_all(model_dir)
    save(generated_df, combined_path)
    print(f"Saved combined generator output: {combined_path}")

    print(f"Loading source input: {translation_path}")
    source_df = load(translation_path)

    validate_columns(source_df, ["id_en", "en"], "source input")
    validate_columns(generated_df, ["id_en"], "generator output")

    source_df = source_df[["id_en", "en"]].copy()
    generated_df = generated_df.copy()

    source_df["id_en"] = source_df["id_en"].astype(str)
    generated_df["id_en"] = generated_df["id_en"].astype(str)

    generated_df["fr"] = generated_df.apply(pick_fr, axis=1)
    generated_df = generated_df[generated_df["fr"].astype(str).str.strip() != ""]
    generated_df = generated_df.drop_duplicates(subset=["id_en"], keep="first")

    merged = source_df.merge(generated_df[["id_en", "fr"]], on="id_en", how="left")
    merged["fr"] = merged["fr"].fillna("")

    missing = int((merged["fr"].astype(str).str.strip() == "").sum())
    if missing:
        print(f"WARNING: {missing} rows have empty fr predictions.")

    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        records.append({
            "run_id": run_id,
            "manual": MANUAL_FLAG,
            "id_en": norm_space(row.get("id_en")),
            "en": norm_space(row.get("en")),
            "fr": norm_space(row.get("fr")),
        })

    return records


def write_prediction(records: list[dict[str, Any]], run_id: str) -> None:
    out_path = Path(resolve_prediction_path(run_id))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved {out_path}")
    print(f"Row count: {len(records)}")


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("Usage: python predict_single_v2.py <run_id> <model>")

    run_id = norm_space(sys.argv[1])
    model = norm_space(sys.argv[2])

    if not run_id:
        raise ValueError("run_id cannot be empty")
    if not model:
        raise ValueError("model cannot be empty")

    records = build_prediction(run_id, model)
    write_prediction(records, run_id)


if __name__ == "__main__":
    main()
