# retrieval_stats_v3.py
# Console-only retrieval reporting for CLEF/JOKER working-notes result tables.
#
# Design goals:
# - Print candidate publication metrics; do not write files.
# - Use source puns / rows as the primary reporting unit where possible.
# - Keep percentages inside count cells for paper-facing readability.
# - Separate fundamentally different analyses into separate tables.

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

from data import load_all


INPUT_ENV = "RETRIEVAL_STATS_INPUT_DIR"

BUCKET_ORDER = ["A1_B1", "A2_B01_DETACHED", "B2_A01_DETACHED"]
BUCKET_LABELS = {
    "A1_B1": r"$A_S$--$B_S$ bridge",
    "A2_B01_DETACHED": r"$A_P$-anchored recovery",
    "B2_A01_DETACHED": r"$B_P$-anchored recovery",
}

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


def fmt_int(n: int | float) -> str:
    return f"{int(n):,}"


def fmt_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def fmt_count_pct(n: int | float, denom: int | float) -> str:
    n_i = int(n)
    denom_i = int(denom)
    if denom_i <= 0:
        return f"{n_i:,} (0.0%)"
    return f"{n_i:,} ({100.0 * n_i / denom_i:.1f}%)"


def print_section(title: str) -> None:
    print("\n" + title)


def print_table(name: str, table: pd.DataFrame) -> None:
    print_section(name)
    if table.empty:
        print("(empty)")
    else:
        print(table.to_string(index=False))


def quantile_as_observed_value(s: pd.Series, q: float) -> int:
    """Return the smallest observed integer k with cumulative share >= q."""
    if s.empty:
        return 0
    counts = s.astype(int).value_counts().sort_index()
    threshold = q * counts.sum()
    cumulative = 0
    for k, n in counts.items():
        cumulative += int(n)
        if cumulative >= threshold:
            return int(k)
    return int(counts.index.max())


def build_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_records: list[dict[str, Any]] = []
    flat_affordances: list[dict[str, Any]] = []
    raw_affordances: list[dict[str, Any]] = []

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

            base = {
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

            raw_affordances.append({**base, "_raw": aff})
            flat_affordances.append({**flatten_dict(aff), **base})

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
                "stored_retrieval_affordance_count": row.get(
                    "retrieval_affordance_count", pd.NA
                ),
            }
        )

    return pd.DataFrame(row_records), pd.DataFrame(flat_affordances), pd.DataFrame(raw_affordances)


def add_missing_rows_to_distribution(dist: pd.Series, total_rows: int, covered_rows: int) -> pd.Series:
    out = dist.copy()
    out.loc[0] = total_rows - covered_rows
    return out.sort_index()


def table_01_dataset_summary(row_df: pd.DataFrame, aff_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    counts_loaded = row_df["affordance_count"].astype(int)
    covered_loaded = counts_loaded[counts_loaded > 0]
    covered_rows = int((counts_loaded > 0).sum())
    uncovered_rows = total_rows - covered_rows
    total_affordances = len(aff_df)
    duplicate_rows = int((row_df["duplicate_affordances_within_row"] > 0).sum())
    duplicate_occurrences = int(row_df["duplicate_affordances_within_row"].sum())

    # All-source-pun distribution includes missing/uncovered rows implied by total_rows.
    all_counts = pd.concat(
        [counts_loaded, pd.Series([0] * max(0, total_rows - len(row_df)))]
    ).astype(int)

    rows = [
        ("Loaded rows", fmt_int(len(row_df))),
        ("Reporting denominator", fmt_int(total_rows)),
        ("Source puns with ≥1 affordance", fmt_count_pct(covered_rows, total_rows)),
        ("Source puns with no affordance", fmt_count_pct(uncovered_rows, total_rows)),
        ("Retrieved affordances", fmt_int(total_affordances)),
        ("Mean affordances per source pun", fmt_float(total_affordances / total_rows)),
        (
            "Mean affordances per covered source pun",
            fmt_float(total_affordances / covered_rows if covered_rows else 0),
        ),
        ("Median affordances per source pun", fmt_int(quantile_as_observed_value(all_counts, 0.50))),
        ("95th percentile affordances per source pun", fmt_int(quantile_as_observed_value(all_counts, 0.95))),
        (
            "Median affordances per covered source pun",
            fmt_int(quantile_as_observed_value(covered_loaded, 0.50)),
        ),
        (
            "95th percentile affordances per covered source pun",
            fmt_int(quantile_as_observed_value(covered_loaded, 0.95)),
        ),
        ("Rows with duplicate affordances", fmt_count_pct(duplicate_rows, total_rows)),
        ("Duplicate affordance occurrences", fmt_int(duplicate_occurrences)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def table_02_affordances_per_source_pun(row_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    counts = row_df["affordance_count"].astype(int)
    dist = counts.value_counts().sort_index()
    dist = add_missing_rows_to_distribution(dist, total_rows, int((counts > 0).sum()))

    max_k = int(max(dist.index.max(), 0))
    rows = []
    for k in range(0, max_k + 1):
        rows.append((k, fmt_count_pct(int(dist.get(k, 0)), total_rows)))
    rows.append(("Total", fmt_count_pct(total_rows, total_rows)))
    return pd.DataFrame(rows, columns=["Retrieved affordances per source pun", "Source puns"])


def per_row_bucket_counts(aff_df: pd.DataFrame) -> pd.DataFrame:
    if aff_df.empty or "retrieval_bucket" not in aff_df.columns:
        return pd.DataFrame()
    rb = pd.crosstab(aff_df["_row_index"], aff_df["retrieval_bucket"])
    for b in BUCKET_ORDER:
        if b not in rb.columns:
            rb[b] = 0
    return rb[BUCKET_ORDER]


def table_03_retrieval_source_coverage(row_df: pd.DataFrame, aff_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    rb = per_row_bucket_counts(aff_df)
    if rb.empty:
        return pd.DataFrame()

    rows = []
    total_affordances = len(aff_df)
    for b in BUCKET_ORDER:
        per_row = rb[b].astype(int)
        covered = per_row[per_row > 0]
        n_covered = len(covered)
        n_aff = int(per_row.sum())
        dup_rows = 0
        dup_occ = 0
        if n_covered:
            # Duplicates for a bucket are repeated identities within the same row and same bucket.
            subset = aff_df[aff_df["retrieval_bucket"] == b]
            bucket_dups = (
                subset.groupby(["_row_index", "_identity"]).size().reset_index(name="n")
            )
            dup_occ = int((bucket_dups["n"] - 1).clip(lower=0).sum())
            dup_rows = int(bucket_dups.loc[bucket_dups["n"] > 1, "_row_index"].nunique())
        rows.append(
            {
                "Retrieval source": BUCKET_LABELS.get(b, b),
                "Covered source puns": fmt_count_pct(n_covered, total_rows),
                "Affordances": fmt_count_pct(n_aff, total_affordances),
                "Affordances / covered pun": fmt_float(n_aff / n_covered if n_covered else 0, 2),
                "Median per covered pun": fmt_int(quantile_as_observed_value(covered, 0.50)),
                "95th pct. per covered pun": fmt_int(quantile_as_observed_value(covered, 0.95)),
                "Max per covered pun": fmt_int(covered.max() if n_covered else 0),
                "Rows with duplicates": fmt_count_pct(dup_rows, total_rows),
                "Duplicate affordances": fmt_int(dup_occ),
            }
        )

    total_counts = row_df["affordance_count"].astype(int)
    covered_total = total_counts[total_counts > 0]
    rows.append(
        {
            "Retrieval source": "Overall",
            "Covered source puns": fmt_count_pct(len(covered_total), total_rows),
            "Affordances": fmt_count_pct(total_affordances, total_affordances),
            "Affordances / covered pun": fmt_float(total_affordances / len(covered_total), 2),
            "Median per covered pun": fmt_int(quantile_as_observed_value(covered_total, 0.50)),
            "95th pct. per covered pun": fmt_int(quantile_as_observed_value(covered_total, 0.95)),
            "Max per covered pun": fmt_int(covered_total.max()),
            "Rows with duplicates": fmt_count_pct(int((row_df["duplicate_affordances_within_row"] > 0).sum()), total_rows),
            "Duplicate affordances": fmt_int(int(row_df["duplicate_affordances_within_row"].sum())),
        }
    )
    return pd.DataFrame(rows)


def table_04_retrieval_source_count_distribution(aff_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    rb = per_row_bucket_counts(aff_df)
    if rb.empty:
        return pd.DataFrame()

    max_k = int(rb.max().max())
    rows = []
    for b in BUCKET_ORDER:
        per_row = rb[b].astype(int)
        covered = int((per_row > 0).sum())
        rec: dict[str, Any] = {"Retrieval source": BUCKET_LABELS.get(b, b)}
        for k in range(1, max_k + 1):
            rec[f"{k} aff."] = fmt_count_pct(int((per_row == k).sum()), total_rows)
        rec["Any aff."] = fmt_count_pct(covered, total_rows)
        rows.append(rec)
    return pd.DataFrame(rows)


def table_05_bucket_combination_coverage(aff_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    rb = per_row_bucket_counts(aff_df)
    if rb.empty:
        return pd.DataFrame()

    present = rb.gt(0)

    def combo_label(row: pd.Series) -> str:
        active = [BUCKET_LABELS.get(b, b) for b in BUCKET_ORDER if bool(row[b])]
        return " + ".join(active) if active else "No retrieved affordance"

    combos = present.apply(combo_label, axis=1)
    combo_counts = combos.value_counts().to_dict()
    no_aff = total_rows - rb.shape[0]
    if no_aff > 0:
        combo_counts["No retrieved affordance"] = combo_counts.get("No retrieved affordance", 0) + no_aff

    order = [
        "No retrieved affordance",
        BUCKET_LABELS["A1_B1"],
        BUCKET_LABELS["A2_B01_DETACHED"],
        BUCKET_LABELS["B2_A01_DETACHED"],
        f"{BUCKET_LABELS['A1_B1']} + {BUCKET_LABELS['A2_B01_DETACHED']}",
        f"{BUCKET_LABELS['A1_B1']} + {BUCKET_LABELS['B2_A01_DETACHED']}",
        f"{BUCKET_LABELS['A2_B01_DETACHED']} + {BUCKET_LABELS['B2_A01_DETACHED']}",
        f"{BUCKET_LABELS['A1_B1']} + {BUCKET_LABELS['A2_B01_DETACHED']} + {BUCKET_LABELS['B2_A01_DETACHED']}",
    ]

    rows = []
    for label in order:
        n = int(combo_counts.get(label, 0))
        if n:
            rows.append({"Retrieval-source combination": label, "Source puns": fmt_count_pct(n, total_rows)})
    # Include any unexpected labels.
    for label, n in sorted(combo_counts.items()):
        if label not in order and n:
            rows.append({"Retrieval-source combination": label, "Source puns": fmt_count_pct(int(n), total_rows)})
    rows.append({"Retrieval-source combination": "Total", "Source puns": fmt_count_pct(total_rows, total_rows)})
    return pd.DataFrame(rows)


def table_06_total_affordance_count_by_source_mix(aff_df: pd.DataFrame, row_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    """For rows with k total affordances, show which sources produced those affordances.

    This is a candidate analysis table, not necessarily paper-facing: rows are total affordances per source pun,
    and bucket columns are affordance occurrences produced by that bucket among rows in that stratum.
    """
    rb = per_row_bucket_counts(aff_df)
    if rb.empty:
        return pd.DataFrame()

    per_row = rb.copy()
    per_row["total_affordances"] = per_row.sum(axis=1).astype(int)
    max_k = int(max(row_df["affordance_count"].max(), 0))
    rows = []
    for k in range(0, max_k + 1):
        if k == 0:
            rows.append(
                {
                    "Total affordances in source pun": 0,
                    "Source puns": fmt_count_pct(total_rows - rb.shape[0], total_rows),
                    BUCKET_LABELS["A1_B1"]: "0",
                    BUCKET_LABELS["A2_B01_DETACHED"]: "0",
                    BUCKET_LABELS["B2_A01_DETACHED"]: "0",
                }
            )
            continue
        subset = per_row[per_row["total_affordances"] == k]
        n_rows = len(subset)
        denom_aff = n_rows * k
        rec = {
            "Total affordances in source pun": k,
            "Source puns": fmt_count_pct(n_rows, total_rows),
        }
        for b in BUCKET_ORDER:
            rec[BUCKET_LABELS[b]] = fmt_count_pct(int(subset[b].sum()), denom_aff) if denom_aff else "0"
        rows.append(rec)
    rows.append(
        {
            "Total affordances in source pun": "Total",
            "Source puns": fmt_count_pct(total_rows, total_rows),
            BUCKET_LABELS["A1_B1"]: fmt_count_pct(int((aff_df["retrieval_bucket"] == "A1_B1").sum()), len(aff_df)),
            BUCKET_LABELS["A2_B01_DETACHED"]: fmt_count_pct(int((aff_df["retrieval_bucket"] == "A2_B01_DETACHED").sum()), len(aff_df)),
            BUCKET_LABELS["B2_A01_DETACHED"]: fmt_count_pct(int((aff_df["retrieval_bucket"] == "B2_A01_DETACHED").sum()), len(aff_df)),
        }
    )
    return pd.DataFrame(rows)


def table_07_retrieval_scores_by_source(aff_df: pd.DataFrame) -> pd.DataFrame:
    if aff_df.empty or "retrieval_bucket" not in aff_df.columns:
        return pd.DataFrame()

    score_cols = [c for c in SCORE_LABELS if c in aff_df.columns and pd.api.types.is_numeric_dtype(aff_df[c])]
    rows = []
    for b in BUCKET_ORDER:
        subset = aff_df[aff_df["retrieval_bucket"] == b]
        if subset.empty:
            continue
        rec = {"Retrieval source": BUCKET_LABELS.get(b, b)}
        for c in score_cols:
            rec[SCORE_LABELS[c]] = fmt_float(subset[c].mean())
        rows.append(rec)
    rec = {"Retrieval source": "Overall"}
    for c in score_cols:
        rec[SCORE_LABELS[c]] = fmt_float(aff_df[c].mean())
    rows.append(rec)
    return pd.DataFrame(rows)


def table_08_relation_summary(aff_df: pd.DataFrame) -> pd.DataFrame:
    relation_col = "relation" if "relation" in aff_df.columns else "_relation"
    if aff_df.empty or relation_col not in aff_df.columns:
        return pd.DataFrame()
    rows = []
    score_col = "scores.overall_score"
    for rel, subset in aff_df.groupby(relation_col, dropna=False):
        rec = {
            "Relation": str(rel),
            "Affordances": fmt_count_pct(len(subset), len(aff_df)),
        }
        if score_col in aff_df.columns:
            rec["Overall score"] = fmt_float(subset[score_col].mean())
        rows.append(rec)
    rows = sorted(rows, key=lambda r: r["Relation"])
    return pd.DataFrame(rows)


def table_09_source_by_relation(aff_df: pd.DataFrame) -> pd.DataFrame:
    relation_col = "relation" if "relation" in aff_df.columns else "_relation"
    if aff_df.empty or "retrieval_bucket" not in aff_df.columns or relation_col not in aff_df.columns:
        return pd.DataFrame()

    rows = []
    relations = sorted([str(x) for x in aff_df[relation_col].dropna().unique()])
    for b in BUCKET_ORDER:
        subset = aff_df[aff_df["retrieval_bucket"] == b]
        rec = {"Retrieval source": BUCKET_LABELS.get(b, b)}
        for rel in relations:
            rec[rel] = fmt_count_pct(int((subset[relation_col].astype(str) == rel).sum()), len(subset))
        rec["Total"] = fmt_count_pct(len(subset), len(aff_df))
        rows.append(rec)
    rec = {"Retrieval source": "Overall"}
    for rel in relations:
        rec[rel] = fmt_count_pct(int((aff_df[relation_col].astype(str) == rel).sum()), len(aff_df))
    rec["Total"] = fmt_count_pct(len(aff_df), len(aff_df))
    rows.append(rec)
    return pd.DataFrame(rows)


def table_10_duplicate_and_identity_reuse(row_df: pd.DataFrame, raw_df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    identity_counts = (
        raw_df.groupby(["_left", "_right", "_relation"], dropna=False)
        .agg(
            affordance_occurrences=("_identity", "size"),
            distinct_source_puns=("_row_index", "nunique"),
        )
        .reset_index()
    )

    unique_identities = len(identity_counts)
    occurrences = len(raw_df)
    single_use = int((identity_counts["affordance_occurrences"] == 1).sum())
    reused = int((identity_counts["distinct_source_puns"] > 1).sum())
    repeated_within_one_row_only = int(
        (
            (identity_counts["distinct_source_puns"] == 1)
            & (identity_counts["affordance_occurrences"] > 1)
        ).sum()
    )
    dup_rows = int((row_df["duplicate_affordances_within_row"] > 0).sum())
    dup_occ = int(row_df["duplicate_affordances_within_row"].sum())

    rows = [
        ("Rows with duplicate affordances", fmt_count_pct(dup_rows, total_rows)),
        ("Duplicate affordance occurrences", fmt_int(dup_occ)),
        ("Unique affordance identities", fmt_int(unique_identities)),
        ("Affordance occurrences", fmt_int(occurrences)),
        ("Single-use identities", fmt_count_pct(single_use, unique_identities)),
        ("Identities reused across source puns", fmt_count_pct(reused, unique_identities)),
        ("Identities repeated only within one source pun", fmt_count_pct(repeated_within_one_row_only, unique_identities)),
        ("Mean occurrences per identity", fmt_float(occurrences / unique_identities if unique_identities else 0)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def table_11_top_reused_identities(raw_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    out = (
        raw_df.groupby(["_left", "_right", "_relation"], dropna=False)
        .agg(
            occurrences=("_identity", "size"),
            source_puns=("_row_index", "nunique"),
            pun_words=("_pun_word", "nunique"),
        )
        .reset_index()
        .sort_values(["occurrences", "source_puns"], ascending=False)
        .head(top_n)
    )
    return out.rename(
        columns={
            "_left": "Left",
            "_right": "Right",
            "_relation": "Relation",
            "occurrences": "Occurrences",
            "source_puns": "Source puns",
            "pun_words": "Pun words",
        }
    )


def validate_loaded_vs_denominator(loaded_rows: int, total_rows: int) -> None:
    if loaded_rows != total_rows:
        print(
            f"\nNOTE: loaded rows ({loaded_rows:,}) differ from reporting denominator "
            f"({total_rows:,}). Missing rows are treated as zero-affordance rows "
            "in denominator-based tables."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Console-only retrieval statistics v3")
    parser.add_argument("--input-dir", default="", help="Retrieval input directory; defaults to config/env path")
    parser.add_argument("--total-rows", type=int, default=4061, help="Reporting denominator")
    parser.add_argument("--top-n", type=int, default=25, help="Rows for top-reuse diagnostic table")
    args = parser.parse_args()

    input_dir = args.input_dir.rstrip("/") + "/" if args.input_dir else resolve_retrieval_input_dir()

    print("Loading retrieval input:", input_dir.rstrip("/"))
    df = load_all(input_dir)
    print(f"Loaded rows: {len(df):,}")
    print(f"Reporting denominator: {args.total_rows:,}")
    validate_loaded_vs_denominator(len(df), args.total_rows)

    row_df, aff_df, raw_df = build_frames(df)

    print_table("table_01_dataset_summary", table_01_dataset_summary(row_df, aff_df, args.total_rows))
    print_table("table_02_affordances_per_source_pun", table_02_affordances_per_source_pun(row_df, args.total_rows))
    print_table("table_03_retrieval_source_coverage", table_03_retrieval_source_coverage(row_df, aff_df, args.total_rows))
    print_table("table_04_retrieval_source_count_distribution", table_04_retrieval_source_count_distribution(aff_df, args.total_rows))
    print_table("table_05_bucket_combination_coverage", table_05_bucket_combination_coverage(aff_df, args.total_rows))
    print_table("table_06_total_affordance_count_by_source_mix", table_06_total_affordance_count_by_source_mix(aff_df, row_df, args.total_rows))
    print_table("table_07_retrieval_scores_by_source", table_07_retrieval_scores_by_source(aff_df))
    print_table("table_08_relation_summary", table_08_relation_summary(aff_df))
    print_table("table_09_source_by_relation", table_09_source_by_relation(aff_df))
    print_table("table_10_duplicate_and_identity_reuse", table_10_duplicate_and_identity_reuse(row_df, raw_df, args.total_rows))
    print_table("table_11_top_reused_identities", table_11_top_reused_identities(raw_df, args.top_n))


if __name__ == "__main__":
    main()
