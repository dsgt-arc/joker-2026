"""
Build JOKER task 2 prediction.json from Borda discriminator outputs.

Usage:
  python predict_borda_v4_correct.py <run_id> <discriminator_model> <generator_model> <weights_string>

Example:
  python predict_borda_v4_correct.py dsgt_task2_claude_sonnet_4.6_25_25_25_25 claude claude 25_25_25_25

Inputs:
  - data/2026/joker_task2_en_fr_2026_test.json via config.translation_path
  - data/processed/discriminate/run1/<discriminator_model>/<generator_model>/borda/<weights_string>/*.tsv

Outputs:
  - data/processed/discriminate/run1/<discriminator_model>/<generator_model>/borda/<weights_string>.tsv
  - data/processed/predict/<run_id>/prediction.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config import translation_path
from data import load, load_all, save


MANUAL_FLAG = 0
DISCRIMINATOR_RUN_FOLDER = "run1"


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def resolve_borda_dir(discriminator_model: str, generator_model: str, weights_string: str) -> str:
    return (
        f"../data/processed/discriminate/"
        f"{DISCRIMINATOR_RUN_FOLDER}/"
        f"{discriminator_model}/"
        f"{generator_model}/"
        f"borda/"
        f"{weights_string}/"
    )


def resolve_combined_output_path(discriminator_model: str, generator_model: str, weights_string: str) -> str:
    return (
        f"../data/processed/discriminate/"
        f"{DISCRIMINATOR_RUN_FOLDER}/"
        f"{discriminator_model}/"
        f"{generator_model}/"
        f"borda/"
        f"{weights_string}.tsv"
    )


def resolve_prediction_path(run_id: str) -> str:
    return f"../data/processed/predict/{run_id}/prediction.json"


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def pick_fr(row: pd.Series) -> str:
    # Borda discriminator output stores the resolved winning French pun here.
    # Keep fallbacks so older/renamed discriminator outputs still work.
    for col in (
        "discriminator_run1_winner_pun",
        "winner_pun",
        "selected_pun",
        "final_translation",
        "borda_translation",
        "fr",
        "french",
        "translation",
        "prediction",
    ):
        if col in row.index and norm_space(row.get(col)):
            return norm_space(row.get(col))
    return ""


def build_prediction(
    run_id: str,
    discriminator_model: str,
    generator_model: str,
    weights_string: str,
) -> list[dict[str, Any]]:
    borda_dir = resolve_borda_dir(discriminator_model, generator_model, weights_string)
    combined_path = resolve_combined_output_path(discriminator_model, generator_model, weights_string)

    print(f"Discriminator run folder: {DISCRIMINATOR_RUN_FOLDER}")
    print(f"Loading Borda discriminator chunks: {borda_dir}")

    generated_df = load_all(borda_dir)
    save(generated_df, combined_path)
    print(f"Saved combined Borda output: {combined_path}")

    print(f"Loading source input: {translation_path}")
    source_df = load(translation_path)

    validate_columns(source_df, ["id_en", "en"], "source input")
    validate_columns(generated_df, ["id_en"], "borda output")

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
    if len(sys.argv) != 5:
        raise ValueError(
            "Usage: python predict_borda_v4_correct.py "
            "<run_id> <discriminator_model> <generator_model> <weights_string>"
        )

    run_id = norm_space(sys.argv[1])
    discriminator_model = norm_space(sys.argv[2])
    generator_model = norm_space(sys.argv[3])
    weights_string = norm_space(sys.argv[4])

    if not run_id:
        raise ValueError("run_id cannot be empty")
    if not discriminator_model:
        raise ValueError("discriminator_model cannot be empty")
    if not generator_model:
        raise ValueError("generator_model cannot be empty")
    if not weights_string:
        raise ValueError("weights_string cannot be empty")

    records = build_prediction(run_id, discriminator_model, generator_model, weights_string)
    write_prediction(records, run_id)


if __name__ == "__main__":
    main()
