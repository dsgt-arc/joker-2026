"""
Build JOKER task 2 prediction.json from Run 2 ensemble Borda outputs.

Reads one of the discriminator_run2_v3 cross-model Borda output directories:
  ../data/processed/discriminate/run2/{ensemble_run}/borda/{borda_method}/{internal_weights}/{model_weights}/*.tsv

Usage from src:
  python predict_ensemble.py <run_id> <ensemble_run> <borda_method> <internal_weights> <model_weights>

Examples:
  python predict_ensemble.py dsgt_task2_ensemble_judges_first_25_25_25_25_25_25_25 ensemble judges_then_models 25_25_25_25 25_25_25
  python predict_ensemble.py dsgt_task2_ensemble_models_first_25_25_25_25_25_25_25 ensemble models_then_judges 25_25_25_25 25_25_25
  python predict_ensemble.py dsgt_task2_ensemble_pooled_25_25_25_25_25_25_25 ensemble pooled_rankings 25_25_25_25 25_25_25

Accepted borda_method aliases:
  judges_then_models: judges_then_models, judge_first, judges_first, judges_then_model, judge_then_model
  models_then_judges: models_then_judges, model_first, models_first, models_then_judge, model_then_judge
  pooled_rankings: pooled_rankings, pooled, pool

Inputs:
  - data/2026/joker_task2_en_fr_2026_test.json via config.translation_path
  - data/processed/discriminate/run2/{ensemble_run}/borda/{borda_method}/{internal_weights}/{model_weights}/*.tsv

Outputs:
  - data/processed/discriminate/run2/{ensemble_run}/borda/{borda_method}/{internal_weights}/{model_weights}.tsv
  - data/processed/predict/{run_id}/prediction.json
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
DISCRIMINATOR_RUN_FOLDER = "run2"
VALID_BORDA_METHODS = {
    "judges_then_models": "judges_then_models",
    "judge_first": "judges_then_models",
    "judges_first": "judges_then_models",
    "judges_then_model": "judges_then_models",
    "judge_then_model": "judges_then_models",
    "models_then_judges": "models_then_judges",
    "model_first": "models_then_judges",
    "models_first": "models_then_judges",
    "models_then_judge": "models_then_judges",
    "model_then_judge": "models_then_judges",
    "pooled_rankings": "pooled_rankings",
    "pooled": "pooled_rankings",
    "pool": "pooled_rankings",
}


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def resolve_borda_method(value: str) -> str:
    key = norm_space(value).lower()
    if key not in VALID_BORDA_METHODS:
        valid = ", ".join(sorted(set(VALID_BORDA_METHODS.values())))
        aliases = ", ".join(sorted(VALID_BORDA_METHODS))
        raise ValueError(f"Invalid borda_method {value!r}. Valid methods: {valid}. Accepted aliases: {aliases}")
    return VALID_BORDA_METHODS[key]


def resolve_borda_dir(
    ensemble_run: str,
    borda_method: str,
    internal_weights: str,
    model_weights: str,
) -> str:
    method = resolve_borda_method(borda_method)
    return (
        f"../data/processed/discriminate/"
        f"{DISCRIMINATOR_RUN_FOLDER}/"
        f"{ensemble_run}/"
        f"borda/"
        f"{method}/"
        f"{internal_weights}/"
        f"{model_weights}/"
    )


def resolve_combined_output_path(
    ensemble_run: str,
    borda_method: str,
    internal_weights: str,
    model_weights: str,
) -> str:
    method = resolve_borda_method(borda_method)
    return (
        f"../data/processed/discriminate/"
        f"{DISCRIMINATOR_RUN_FOLDER}/"
        f"{ensemble_run}/"
        f"borda/"
        f"{method}/"
        f"{internal_weights}/"
        f"{model_weights}.tsv"
    )


def resolve_prediction_path(run_id: str) -> str:
    return f"../data/processed/predict/{run_id}/prediction.json"


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def pick_fr(row: pd.Series) -> str:
    # Run 2 cross-model Borda output stores the resolved winning French pun here.
    # Keep fallbacks so older/renamed outputs still work.
    for col in (
        "discriminator_run2_winner_pun",
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
    ensemble_run: str,
    borda_method: str,
    internal_weights: str,
    model_weights: str,
) -> list[dict[str, Any]]:
    method = resolve_borda_method(borda_method)
    borda_dir = resolve_borda_dir(ensemble_run, method, internal_weights, model_weights)
    combined_path = resolve_combined_output_path(ensemble_run, method, internal_weights, model_weights)

    print(f"Discriminator run folder: {DISCRIMINATOR_RUN_FOLDER}")
    print(f"Ensemble run: {ensemble_run}")
    print(f"Borda method: {method}")
    print(f"Internal weights: {internal_weights}")
    print(f"Model weights: {model_weights}")
    print(f"Loading Run 2 Borda chunks: {borda_dir}")

    generated_df = load_all(borda_dir)
    save(generated_df, combined_path)
    print(f"Saved combined Run 2 Borda output: {combined_path}")

    print(f"Loading source input: {translation_path}")
    source_df = load(translation_path)

    validate_columns(source_df, ["id_en", "en"], "source input")
    validate_columns(generated_df, ["id_en"], "Run 2 Borda output")

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


def usage() -> str:
    return """Usage:
  python predict_ensemble.py <run_id> <ensemble_run> <borda_method> <internal_weights> <model_weights>

Examples:
  python predict_ensemble.py dsgt_task2_ensemble_judges_first_25_25_25_25_25_25_25 ensemble judges_then_models 25_25_25_25 25_25_25
  python predict_ensemble.py dsgt_task2_ensemble_models_first_25_25_25_25_25_25_25 ensemble models_then_judges 25_25_25_25 25_25_25
  python predict_ensemble.py dsgt_task2_ensemble_pooled_25_25_25_25_25_25_25 ensemble pooled_rankings 25_25_25_25 25_25_25

borda_method aliases:
  judge_first / judges_first -> judges_then_models
  model_first / models_first -> models_then_judges
  pooled / pool -> pooled_rankings
"""


def main() -> None:
    if len(sys.argv) != 6:
        raise ValueError(usage())

    run_id = norm_space(sys.argv[1])
    ensemble_run = norm_space(sys.argv[2])
    borda_method = norm_space(sys.argv[3])
    internal_weights = norm_space(sys.argv[4])
    model_weights = norm_space(sys.argv[5])

    if not run_id:
        raise ValueError("run_id cannot be empty")
    if not ensemble_run:
        raise ValueError("ensemble_run cannot be empty")
    if not borda_method:
        raise ValueError("borda_method cannot be empty")
    if not internal_weights:
        raise ValueError("internal_weights cannot be empty")
    if not model_weights:
        raise ValueError("model_weights cannot be empty")

    records = build_prediction(run_id, ensemble_run, borda_method, internal_weights, model_weights)
    write_prediction(records, run_id)


if __name__ == "__main__":
    main()
