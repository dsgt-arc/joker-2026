# retrieval_stats_v2.py

"""
Publication-oriented retrieval statistics for CLEF/JOKER working notes.

This script replaces the exploratory retrieval_stats.py report with a compact set
of reporting tables centered on source-pun coverage, retrieval-source coverage,
affordance-count distributions, scoring metrics, and duplicate checks.

Outputs are written as CSV and LaTeX files under --out-dir.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from config import retrieval_dir
except Exception:
    retrieval_dir = ""

from data import load_all, save


INPUT_ENV = "RETRIEVAL_STATS_INPUT_DIR"

BUCKET_LABELS = {
    "A1_B1": r"$A_S$--$B_S$ bridge",
    "A2_B01_DETACHED": r"$A_P$-anchored recovery",
    "B2_A01_DETACHED": r"$B_P$-anchored recovery",
}

BUCKET_ORDER = [
    "A1_B1",
    "A2_B01_DETACHED",
    "B2_A01_DETACHED",
]

SCORE_LABELS = {
    "scores.phonetic_match": "Phonetic",
    "scores.semantic_domain_similarity": "Semantic domain",
    "scores.semantic_surprise": "Semantic surprise",
    "scores.french_naturalness": "Naturalness",
    "scores.pun_pivot_usability": "Pivot usability",
    "scores.overall_score": "Overall",
}


def norm_space(x: Any) -> str:
    return " ".join(str(x or "").split())


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


def resolve_retrieval_input_dir() -> str:
    explicit = os.environ.get(INPUT_ENV, "").strip()
    if explicit:
        return explicit.rstrip("/") + "/"

    if retrieval_dir:
        return f"{retrieval_dir}gemini/"

    root = Path(__file__).resolve().parents[1]
    return str(root / "data" / "processed" / "retrieval" / "gemini") + "/"


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def parse_retrieval_affordances(row: pd.Series) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    direct = safe_json_loads(row.get("retrieval_affordances_json"))
    if isinstance(direct, list):
        items.extend([x for x in direct if isinstance(x, dict)])

    bridge_candidates = safe_json_loads(row.get("bridge_candidates"))
    if isinstance(bridge_candidates, list):
        items.extend([x for x in bridge_candidates if isinstance(x, dict)])

    for col in ("generator_affordance_pack", "retrieval_pack_compact"):
        value = safe_json_loads(row.get(col))
        if isinstance(value, dict):
            top = value.get("top_bridge_candidates")
            if isinstance(top, list):
                items.extend([x for x in top if isinstance(x, dict)])

            nested = value.get("generator_affordance_pack")
            if isinstance(nested, dict):
                nested_top = nested.get("top_bridge_candidates")
                if isinstance(nested_top, list):
                    items.extend([x for x in nested_top if isinstance(x, dict)])

    return items


def bridge_identity(aff: dict[str, Any]) -> tuple[str, str, str]:
    left = norm_space(
        aff.get("left")
        or aff.get("pivot_a")
        or aff.get("source_surface")
        or aff.get("a_surface")
        or aff.get("source")
        or aff.get("left_text")
        or aff.get("a")
    )
    right = norm_space(
        aff.get("right")
        or aff.get("pivot_b")
        or aff.get("candidate_surface")
        or aff.get("b_surface")
        or aff.get("candidate")
        or aff.get("right_text")
        or aff.get("b")
    )
    relation = norm_space(
        aff.get("relation")
        or aff.get("bridge_type")
        or aff.get("phonetic_relation")
        or aff.get("match_type")
    )
    return left, right, relation


def fmt_count_pct(n: int, denom: int) -> str:
    pct = 0.0 if denom == 0 else 100.0 * n / denom
    return f"{n:,} ({pct:.1f}%)"


def fmt_num(x: float | int | None, ndigits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{float(x):.{ndigits}f}"


def write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    latex = df.to_latex(index=False, escape=False, caption=caption, label=label)
    (out_dir / f"{name}.tex").write_text(latex, encoding="utf-8")


def build_flat_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_records: list[dict[str, Any]] = []
    flat_affordances: list[dict[str, Any]] = []

    for row_index, row in df.iterrows():
        affordances = parse_retrieval_affordances(row)
        seen_in_row: set[tuple[str, str, str]] = set()
        duplicate_within_row = 0

        for aff_index, aff0 in enumerate(affordances):
            aff = dict(aff0)
            left, right, relation = bridge_identity(aff)
            identity = (left, right, relation)

            if identity in seen_in_row:
                duplicate_within_row += 1
            seen_in_row.add(identity)

            flat = flatten_dict(aff)
            flat.update(
                {
                    "_row_index": row_index,
                    "_affordance_index": aff_index,
                    "_id_en": row.get("id_en", ""),
                    "_pun_word": row.get("pun_word", ""),
                    "_pun_type": row.get("pun_type", ""),
                    "_text_clean": row.get("text_clean", ""),
                    "_left": left,
                    "_right": right,
                    "_relation": relation,
                    "_identity": json.dumps(identity, ensure_ascii=False),
                }
            )
            flat_affordances.append(flat)

        row_records.append(
            {
                "row_index": row_index,
                "id_en": row.get("id_en", ""),
                "pun_word": row.get("pun_word", ""),
                "pun_type": row.get("pun_type", ""),
                "text_clean": row.get("text_clean", ""),
                "affordance_count": len(affordances),
                "has_affordance": len(affordances) > 0,
                "unique_affordance_count_within_row": len(seen_in_row),
                "duplicate_affordances_within_row": duplicate_within_row,
                "stored_retrieval_affordance_count": row.get("retrieval_affordance_count", pd.NA),
            }
        )

    return pd.DataFrame(row_records), pd.DataFrame(flat_affordances)


def make_dataset_summary(row_df: pd.DataFrame, aff_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    covered = int(row_df["has_affordance"].sum())
    total_aff = int(len(aff_df))
    counts = row_df["affordance_count"]
    duplicate_occ = int(row_df["duplicate_affordances_within_row"].sum())
    rows_with_dup = int((row_df["duplicate_affordances_within_row"] > 0).sum())

    records = [
        ("Source puns", f"{total_rows:,}"),
        ("Source puns with ≥1 affordance", fmt_count_pct(covered, total_rows)),
        ("Source puns with no affordance", fmt_count_pct(total_rows - covered, total_rows)),
        ("Retrieved affordances", f"{total_aff:,}"),
        ("Mean affordances per source pun", fmt_num(total_aff / total_rows, 3)),
        ("Mean affordances per covered source pun", fmt_num(counts[counts > 0].mean(), 3)),
        ("Median affordances per covered source pun", fmt_num(counts[counts > 0].median(), 0)),
        ("95th percentile affordances per covered source pun", fmt_num(counts[counts > 0].quantile(0.95), 0)),
        ("Rows with duplicate affordances", fmt_count_pct(rows_with_dup, total_rows)),
        ("Duplicate affordance occurrences", f"{duplicate_occ:,}"),
    ]
    return pd.DataFrame(records, columns=["Metric", "Value"])


def make_affordances_per_row_table(row_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    counts = row_df["affordance_count"].value_counts().sort_index()
    max_k = int(row_df["affordance_count"].max()) if len(row_df) else 0

    rows = []
    for k in range(0, max_k + 1):
        n = int(counts.get(k, 0))
        # If the loaded dataframe excludes some evaluation rows, assign the missing rows to 0 affordances.
        if k == 0 and total_rows > len(row_df):
            n += total_rows - len(row_df)
        rows.append({"Retrieved affordances per source pun": k, "Source puns": fmt_count_pct(n, total_rows)})

    rows.append({"Retrieved affordances per source pun": "Total", "Source puns": fmt_count_pct(total_rows, total_rows)})
    return pd.DataFrame(rows)


def make_retrieval_source_coverage_table(row_df: pd.DataFrame, aff_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    if aff_df.empty or "retrieval_bucket" not in aff_df.columns:
        return pd.DataFrame()

    total_aff = len(aff_df)
    rows = []

    ordered_buckets = [b for b in BUCKET_ORDER if b in set(aff_df["retrieval_bucket"])]
    ordered_buckets += [b for b in sorted(set(aff_df["retrieval_bucket"])) if b not in ordered_buckets]

    for bucket in ordered_buckets:
        sub = aff_df[aff_df["retrieval_bucket"] == bucket]
        per_row = sub.groupby("_row_index").size()
        covered_rows = int(per_row.shape[0])
        aff_count = int(sub.shape[0])

        # duplicate occurrences within the same source pun and bucket
        dup_occ = int(
            sub.duplicated(subset=["_row_index", "_identity"], keep="first").sum()
        )
        dup_rows = int(
            sub.groupby("_row_index")["_identity"]
            .apply(lambda s: s.duplicated().any())
            .sum()
        )

        rows.append(
            {
                "Retrieval source": BUCKET_LABELS.get(bucket, bucket),
                "Covered source puns": fmt_count_pct(covered_rows, total_rows),
                "Affordances": fmt_count_pct(aff_count, total_aff),
                "Affordances / covered pun": fmt_num(per_row.mean(), 2),
                "Median per covered pun": fmt_num(per_row.median(), 0),
                "Max per covered pun": int(per_row.max()) if covered_rows else 0,
                "Rows with duplicates": fmt_count_pct(dup_rows, total_rows),
                "Duplicate affordances": f"{dup_occ:,}",
            }
        )

    per_row_all = aff_df.groupby("_row_index").size()
    rows.append(
        {
            "Retrieval source": "Overall",
            "Covered source puns": fmt_count_pct(int(per_row_all.shape[0]), total_rows),
            "Affordances": fmt_count_pct(total_aff, total_aff),
            "Affordances / covered pun": fmt_num(per_row_all.mean(), 2),
            "Median per covered pun": fmt_num(per_row_all.median(), 0),
            "Max per covered pun": int(per_row_all.max()) if len(per_row_all) else 0,
            "Rows with duplicates": fmt_count_pct(int((row_df["duplicate_affordances_within_row"] > 0).sum()), total_rows),
            "Duplicate affordances": f"{int(row_df['duplicate_affordances_within_row'].sum()):,}",
        }
    )

    return pd.DataFrame(rows)


def make_source_count_distribution_by_bucket(aff_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    """Rows are retrieval sources; columns show source-pun coverage by number of affordances from that source."""
    if aff_df.empty or "retrieval_bucket" not in aff_df.columns:
        return pd.DataFrame()

    per_row_bucket = (
        aff_df.groupby(["retrieval_bucket", "_row_index"])
        .size()
        .rename("n")
        .reset_index()
    )
    max_k = int(per_row_bucket["n"].max()) if len(per_row_bucket) else 0

    rows = []
    ordered_buckets = [b for b in BUCKET_ORDER if b in set(aff_df["retrieval_bucket"])]
    ordered_buckets += [b for b in sorted(set(aff_df["retrieval_bucket"])) if b not in ordered_buckets]

    for bucket in ordered_buckets:
        sub = per_row_bucket[per_row_bucket["retrieval_bucket"] == bucket]
        counts = sub["n"].value_counts().to_dict()
        out = {"Retrieval source": BUCKET_LABELS.get(bucket, bucket)}
        for k in range(1, max_k + 1):
            out[f"{k} aff."] = fmt_count_pct(int(counts.get(k, 0)), total_rows)
        out["Any aff."] = fmt_count_pct(int(sub["_row_index"].nunique()), total_rows)
        rows.append(out)

    return pd.DataFrame(rows)


def make_score_table(aff_df: pd.DataFrame) -> pd.DataFrame:
    if aff_df.empty or "retrieval_bucket" not in aff_df.columns:
        return pd.DataFrame()

    score_cols = [c for c in SCORE_LABELS if c in aff_df.columns and pd.api.types.is_numeric_dtype(aff_df[c])]
    if not score_cols:
        return pd.DataFrame()

    rows = []
    ordered_buckets = [b for b in BUCKET_ORDER if b in set(aff_df["retrieval_bucket"])]
    ordered_buckets += [b for b in sorted(set(aff_df["retrieval_bucket"])) if b not in ordered_buckets]

    for bucket in ordered_buckets:
        sub = aff_df[aff_df["retrieval_bucket"] == bucket]
        row = {"Retrieval source": BUCKET_LABELS.get(bucket, bucket)}
        for col in score_cols:
            row[SCORE_LABELS[col]] = fmt_num(sub[col].mean(), 3)
        rows.append(row)

    overall = {"Retrieval source": "Overall"}
    for col in score_cols:
        overall[SCORE_LABELS[col]] = fmt_num(aff_df[col].mean(), 3)
    rows.append(overall)

    return pd.DataFrame(rows)


def make_duplicate_identity_table(aff_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    if aff_df.empty:
        return pd.DataFrame()

    identity_counts = (
        aff_df.groupby(["_left", "_right", "_relation"], dropna=False)
        .agg(
            affordance_occurrences=("_identity", "size"),
            distinct_source_puns=("_row_index", "nunique"),
        )
        .reset_index()
    )

    unique_identities = int(len(identity_counts))
    total_aff = int(len(aff_df))
    reused_across_rows = int((identity_counts["distinct_source_puns"] > 1).sum())
    single_use = int((identity_counts["affordance_occurrences"] == 1).sum())

    return pd.DataFrame(
        [
            {"Metric": "Unique affordance identities", "Value": f"{unique_identities:,}"},
            {"Metric": "Affordance occurrences", "Value": f"{total_aff:,}"},
            {"Metric": "Single-use identities", "Value": fmt_count_pct(single_use, unique_identities)},
            {"Metric": "Identities reused across source puns", "Value": fmt_count_pct(reused_across_rows, unique_identities)},
            {
                "Metric": "Mean occurrences per identity",
                "Value": fmt_num(total_aff / unique_identities if unique_identities else 0, 3),
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None, help="Directory containing retrieval TSV/JSON files.")
    parser.add_argument("--total-rows", type=int, default=4061, help="Evaluation-set source-pun count.")
    parser.add_argument("--out-dir", default="retrieval_stats_v2_tables")
    parser.add_argument("--save-flat", action="store_true", help="Also save flattened input and affordance TSVs.")
    args = parser.parse_args()

    input_dir = (args.input_dir.rstrip("/") + "/") if args.input_dir else resolve_retrieval_input_dir()
    out_dir = Path(args.out_dir)

    print(f"Loading retrieval input: {input_dir.rstrip('/')}")
    df = load_all(input_dir)
    print(f"Loaded rows: {len(df):,}")
    print(f"Reporting denominator: {args.total_rows:,}")

    row_df, aff_df = build_flat_tables(df)

    if args.save_flat:
        save(df, f"{input_dir.rstrip('/')}.tsv")
        row_df.to_csv(out_dir / "row_level_retrieval_records.csv", index=False)
        aff_df.to_csv(out_dir / "flattened_retrieval_affordances.csv", index=False)

    tables = {
        "table_01_dataset_summary": (
            make_dataset_summary(row_df, aff_df, args.total_rows),
            "Dataset-level retrieval summary.",
            "tab:retrieval-dataset-summary",
        ),
        "table_02_affordances_per_source_pun": (
            make_affordances_per_row_table(row_df, args.total_rows),
            "Distribution of retrieved affordances per source pun.",
            "tab:retrieval-affordances-per-source-pun",
        ),
        "table_03_retrieval_source_coverage": (
            make_retrieval_source_coverage_table(row_df, aff_df, args.total_rows),
            "Coverage and yield by retrieval source.",
            "tab:retrieval-source-coverage",
        ),
        "table_04_retrieval_source_count_distribution": (
            make_source_count_distribution_by_bucket(aff_df, args.total_rows),
            "Source-pun coverage by retrieval source and number of affordances contributed by that source.",
            "tab:retrieval-source-count-distribution",
        ),
        "table_05_retrieval_scores_by_source": (
            make_score_table(aff_df),
            "Mean retrieval scoring metrics by retrieval source.",
            "tab:retrieval-scores-by-source",
        ),
        "table_06_affordance_identity_reuse": (
            make_duplicate_identity_table(aff_df, args.total_rows),
            "Affordance identity reuse and duplicate diagnostics.",
            "tab:retrieval-identity-reuse",
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (table, caption, label) in tables.items():
        if table.empty:
            print(f"\n{name}: empty; skipped")
            continue
        write_table(table, out_dir, name, caption, label)
        print(f"\n{name}")
        print(table.to_string(index=False))

    print(f"\nWrote tables to: {out_dir}")


if __name__ == "__main__":
    main()
