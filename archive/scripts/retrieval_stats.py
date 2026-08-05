# retrieval_stats.py

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
TOP_N = int(os.environ.get("RETRIEVAL_STATS_TOP_N", "50"))


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
    out = {}

    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)

        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v

    return out


def value_to_hashable(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v)


def parse_retrieval_affordances(row: pd.Series) -> list[dict[str, Any]]:
    items = []

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


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_subsection(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def print_series_counts(s: pd.Series, title: str, top_n: int = TOP_N) -> None:
    print_subsection(title)

    if s.empty:
        print("(empty)")
        return

    counts = s.fillna("<NA>").map(value_to_hashable).value_counts(dropna=False)
    print(counts.head(top_n).to_string())


def print_numeric_stats(df: pd.DataFrame, title: str) -> None:
    numeric_cols = list(df.select_dtypes(include="number").columns)

    if not numeric_cols:
        return

    print_subsection(title)

    print(
        df[numeric_cols]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .T
        .to_string()
    )


def print_score_extremes(
    aff_df: pd.DataFrame,
    score_cols: list[str],
    top_n: int = TOP_N,
) -> None:
    display_cols = [
        "_row_index",
        "_id_en",
        "_pun_word",
        "_left",
        "_right",
        "_relation",
        "retrieval_bucket",
        "retrieval_bucket_rank",
        "export_lane",
    ]

    display_cols = [c for c in display_cols if c in aff_df.columns]

    for score_col in score_cols:
        print_subsection(f"Highest affordances by {score_col}")
        print(
            aff_df.sort_values(score_col, ascending=False)
            [display_cols + [score_col]]
            .head(top_n)
            .to_string(index=False)
        )

        print_subsection(f"Lowest affordances by {score_col}")
        print(
            aff_df.sort_values(score_col, ascending=True)
            [display_cols + [score_col]]
            .head(top_n)
            .to_string(index=False)
        )


def print_score_group_stats(
    aff_df: pd.DataFrame,
    score_cols: list[str],
    group_cols: list[str],
) -> None:
    for group_col in group_cols:
        if group_col not in aff_df.columns:
            continue

        for score_col in score_cols:
            print_subsection(f"{score_col} by {group_col}")
            print(
                aff_df.groupby(group_col, dropna=False)[score_col]
                .describe(
                    percentiles=[
                        0.01,
                        0.05,
                        0.10,
                        0.25,
                        0.50,
                        0.75,
                        0.90,
                        0.95,
                        0.99,
                    ]
                )
                .to_string()
            )


def main() -> None:
    input_dir = resolve_retrieval_input_dir()

    print("Loading retrieval input:", input_dir.rstrip("/"))

    df = load_all(input_dir)

    print(f"Row count: {len(df):,}")

    save(df, f"{input_dir.rstrip('/')}.tsv")

    row_records = []
    flat_affordances = []
    raw_affordances = []

    for row_index, row in df.iterrows():
        affordances = parse_retrieval_affordances(row)

        seen_in_row = set()
        duplicate_within_row = 0

        for aff_index, aff in enumerate(affordances):
            aff = dict(aff)

            left, right, relation = bridge_identity(aff)
            identity = (left, right, relation)

            if identity in seen_in_row:
                duplicate_within_row += 1

            seen_in_row.add(identity)

            raw_affordances.append(
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
                    "_raw": aff,
                }
            )

            flat = flatten_dict(aff)
            flat["_row_index"] = row_index
            flat["_affordance_index"] = aff_index
            flat["_id_en"] = row.get("id_en", "")
            flat["_pun_word"] = row.get("pun_word", "")
            flat["_pun_type"] = row.get("pun_type", "")
            flat["_text_clean"] = row.get("text_clean", "")
            flat["_left"] = left
            flat["_right"] = right
            flat["_relation"] = relation
            flat["_identity"] = json.dumps(identity, ensure_ascii=False)

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
                "stored_retrieval_affordance_count": row.get(
                    "retrieval_affordance_count",
                    pd.NA,
                ),
            }
        )

    row_df = pd.DataFrame(row_records)
    aff_df = pd.DataFrame(flat_affordances)
    raw_df = pd.DataFrame(raw_affordances)

    print_section("ROW-LEVEL COUNTS")

    counts = row_df["affordance_count"]

    print(f"Rows: {len(row_df):,}")
    print(f"Rows with affordances: {(counts > 0).sum():,}")
    print(f"Rows without affordances: {(counts == 0).sum():,}")
    print(f"Rows coverage pct: {(counts > 0).mean():.4%}")
    print(f"Total affordances: {len(aff_df):,}")
    print(f"Mean affordances per row: {counts.mean():.6f}")

    if (counts > 0).any():
        print(
            f"Mean affordances per covered row: "
            f"{counts[counts > 0].mean():.6f}"
        )

    print_subsection("Affordances per row summary")
    print(
        counts.describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print_series_counts(
        counts,
        "Affordance count distribution",
        top_n=100,
    )

    print_subsection("Coverage by pun_type")
    if "pun_type" in row_df.columns:
        pun_type_stats = (
            row_df.groupby("pun_type", dropna=False)
            .agg(
                rows=("row_index", "size"),
                rows_with_affordances=("has_affordance", "sum"),
                total_affordances=("affordance_count", "sum"),
                mean_affordances_per_row=("affordance_count", "mean"),
                median_affordances_per_row=("affordance_count", "median"),
                max_affordances_in_row=("affordance_count", "max"),
            )
            .reset_index()
        )

        pun_type_stats["coverage_pct"] = (
            pun_type_stats["rows_with_affordances"]
            / pun_type_stats["rows"]
        )

        print(pun_type_stats.to_string(index=False))

        print_subsection("Affordance count distribution by pun_type")
        print(
            pd.crosstab(
                row_df["pun_type"],
                row_df["affordance_count"],
                dropna=False,
            )
            .to_string()
        )

    if "retrieval_affordance_count" in df.columns:
        stored = pd.to_numeric(
            row_df["stored_retrieval_affordance_count"],
            errors="coerce",
        )

        mismatch = row_df[
            stored.fillna(-1).astype(int)
            != row_df["affordance_count"].astype(int)
        ]

        print_subsection("Stored retrieval_affordance_count validation")
        print(f"Mismatched rows: {len(mismatch):,}")

        if len(mismatch):
            print(
                mismatch[
                    [
                        "row_index",
                        "id_en",
                        "pun_word",
                        "pun_type",
                        "affordance_count",
                        "stored_retrieval_affordance_count",
                        "text_clean",
                    ]
                ]
                .head(TOP_N)
                .to_string(index=False)
            )

    print_subsection("Top rows by affordance_count")
    print(
        row_df.sort_values("affordance_count", ascending=False)
        .head(TOP_N)
        .to_string(index=False)
    )

    print_subsection("Rows with duplicate affordance identities within the same row")
    dup_rows = row_df[row_df["duplicate_affordances_within_row"] > 0]
    print(f"Rows with within-row duplicates: {len(dup_rows):,}")

    if len(dup_rows):
        print(
            dup_rows.sort_values(
                "duplicate_affordances_within_row",
                ascending=False,
            )
            .head(TOP_N)
            .to_string(index=False)
        )

    if aff_df.empty:
        print_section("NO AFFORDANCES FOUND")
        return

    score_cols = [
        c for c in aff_df.columns
        if c.startswith("scores.")
        and pd.api.types.is_numeric_dtype(aff_df[c])
    ]

    print_section("AFFORDANCE FIELD INVENTORY")

    field_rows = []

    for col in aff_df.columns:
        present = aff_df[col].notna().sum()

        nonempty = aff_df[col].map(
            lambda x: x not in (None, "", [], {})
        ).sum()

        unique = aff_df[col].map(value_to_hashable).nunique(dropna=True)

        sample_values = (
            aff_df[col]
            .dropna()
            .map(value_to_hashable)
            .replace("", pd.NA)
            .dropna()
            .head(5)
            .tolist()
        )

        field_rows.append(
            {
                "field": col,
                "present_count": present,
                "present_pct": present / len(aff_df),
                "nonempty_count": nonempty,
                "nonempty_pct": nonempty / len(aff_df),
                "unique_count": unique,
                "dtype": str(aff_df[col].dtype),
                "sample_values": " | ".join(sample_values),
            }
        )

    field_df = pd.DataFrame(field_rows).sort_values(["field"])
    print(field_df.to_string(index=False))

    print_section("RAW TOP-LEVEL AFFORDANCE KEYS")

    top_level_key_counts = {}

    for raw in raw_df["_raw"]:
        for key in raw.keys():
            top_level_key_counts[key] = (
                top_level_key_counts.get(key, 0) + 1
            )

    key_df = (
        pd.DataFrame(
            [
                {
                    "key": k,
                    "present_count": v,
                    "present_pct": v / len(raw_df),
                }
                for k, v in top_level_key_counts.items()
            ]
        )
        .sort_values(["present_count", "key"], ascending=[False, True])
    )

    print(key_df.to_string(index=False))

    print_section("AGGREGATIONS FOR EVERY NON-NUMERIC FIELD")

    excluded = {"_raw", "_text_clean"}

    for col in aff_df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(aff_df[col]):
            continue

        print_series_counts(
            aff_df[col],
            f"Top values: {col}",
            TOP_N,
        )

    print_numeric_stats(aff_df, "NUMERIC FIELD STATISTICS")

    print_section("SCORE FIELD REPORTS")

    print_subsection("Detected score fields")
    if score_cols:
        for col in score_cols:
            print(col)
    else:
        print("(none)")

    if score_cols:
        print_numeric_stats(
            aff_df[score_cols],
            "Score field summary statistics",
        )

        print_score_group_stats(
            aff_df,
            score_cols,
            [
                "retrieval_bucket",
                "export_lane",
                "relation",
                "_relation",
                "_pun_type",
            ],
        )

        print_score_extremes(
            aff_df,
            score_cols,
            TOP_N,
        )

    print_section("SCORE CORRELATIONS")

    if score_cols:
        print(
            aff_df[score_cols]
            .corr(numeric_only=True)
            .round(4)
            .to_string()
        )

        if "scores.overall_score" in score_cols:
            print_subsection("Correlation with scores.overall_score")
            print(
                aff_df[score_cols]
                .corr(numeric_only=True)["scores.overall_score"]
                .sort_values(ascending=False)
                .round(4)
                .to_string()
            )


    print_section("BRIDGE DEGREE ANALYSIS")

    left_degree = (
        aff_df.groupby("_left")["_right"]
        .nunique()
        .sort_values(ascending=False)
    )

    right_degree = (
        aff_df.groupby("_right")["_left"]
        .nunique()
        .sort_values(ascending=False)
    )

    print_subsection("Distinct right terms per left term")
    print(
        left_degree.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        .to_string()
    )

    print("\nTop left terms by number of distinct right terms")
    print(left_degree.head(TOP_N).to_string())

    print_subsection("Distinct left terms per right term")
    print(
        right_degree.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        .to_string()
    )

    print("\nTop right terms by number of distinct left terms")
    print(right_degree.head(TOP_N).to_string())

    print_section("BRIDGE IDENTITY REUSE")

    identity_counts = (
        raw_df.groupby(["_left", "_right", "_relation"], dropna=False)
        .agg(
            affordance_occurrences=("_identity", "size"),
            distinct_rows=("_row_index", "nunique"),
            distinct_pun_words=("_pun_word", "nunique"),
            distinct_pun_types=("_pun_type", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["affordance_occurrences", "distinct_rows"],
            ascending=False,
        )
    )

    single_use_count = (
        identity_counts["affordance_occurrences"] == 1
    ).sum()

    multi_use_count = (
        identity_counts["affordance_occurrences"] > 1
    ).sum()

    print(f"Unique bridge identities: {len(identity_counts):,}")
    print(f"Total affordance occurrences: {len(raw_df):,}")
    print(
        f"Mean occurrences per unique bridge identity: "
        f"{len(raw_df) / len(identity_counts):.6f}"
    )
    print(f"Single-use bridge identities: {single_use_count:,}")
    print(f"Multi-use bridge identities: {multi_use_count:,}")
    print(
        f"Single-use bridge identity pct: "
        f"{single_use_count / len(identity_counts):.4%}"
    )
    print(
        f"Multi-use bridge identity pct: "
        f"{multi_use_count / len(identity_counts):.4%}"
    )

    print_subsection("Top reused bridge identities")
    print(identity_counts.head(TOP_N).to_string(index=False))

    print_subsection("Reuse distribution: occurrences per bridge identity")
    print(
        identity_counts["affordance_occurrences"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )
    print(
        identity_counts["affordance_occurrences"]
        .value_counts()
        .sort_index()
        .head(100)
        .to_string()
    )

    print_subsection("Reuse distribution: distinct rows per bridge identity")
    print(
        identity_counts["distinct_rows"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )
    print(
        identity_counts["distinct_rows"]
        .value_counts()
        .sort_index()
        .head(100)
        .to_string()
    )

    print_section("WITHIN-ROW VS ACROSS-ROW REUSE")

    total_duplicate_within_row = int(
        row_df["duplicate_affordances_within_row"].sum()
    )

    reused_across_rows = identity_counts[
        identity_counts["distinct_rows"] > 1
    ]

    reused_only_within_single_row = identity_counts[
        (identity_counts["distinct_rows"] == 1)
        & (identity_counts["affordance_occurrences"] > 1)
    ]

    print(
        f"Duplicate affordance occurrences within same row: "
        f"{total_duplicate_within_row:,}"
    )
    print(
        f"Bridge identities reused across multiple rows: "
        f"{len(reused_across_rows):,}"
    )
    print(
        f"Bridge identities repeated only within one row: "
        f"{len(reused_only_within_single_row):,}"
    )

    print_subsection("Top bridge identities reused across rows")
    print(reused_across_rows.head(TOP_N).to_string(index=False))

    print_section("CROSSTABS")

    categorical_cols = [
        c
        for c in aff_df.columns
        if c not in excluded
        and not pd.api.types.is_numeric_dtype(aff_df[c])
        and aff_df[c].nunique(dropna=True) <= 100
    ]

    preferred_pairs = [
        ("retrieval_bucket", "relation"),
        ("retrieval_bucket", "export_lane"),
        ("export_lane", "relation"),
        ("relation", "_pun_type"),
        ("retrieval_bucket", "_pun_type"),
        ("export_lane", "_pun_type"),
    ]

    seen_pairs = set()

    for a, b in preferred_pairs:
        if a in aff_df.columns and b in aff_df.columns:
            print_subsection(f"Crosstab: {a} × {b}")
            print(pd.crosstab(aff_df[a], aff_df[b]).to_string())
            seen_pairs.add(tuple(sorted((a, b))))

    for i, a in enumerate(categorical_cols):
        for b in categorical_cols[i + 1:]:
            key = tuple(sorted((a, b)))

            if key in seen_pairs:
                continue

            if a.startswith("_") and b.startswith("_"):
                continue

            print_subsection(f"Crosstab: {a} × {b}")
            print(pd.crosstab(aff_df[a], aff_df[b]).to_string())
            seen_pairs.add(key)

    print_section("NUMERIC FIELD AGGREGATIONS BY CATEGORICAL FIELD")

    numeric_cols = [
        c for c in aff_df.columns
        if pd.api.types.is_numeric_dtype(aff_df[c])
    ]

    group_cols = [
        c
        for c in categorical_cols
        if c not in {"_identity", "_text_clean"}
    ]

    for group_col in group_cols:
        for num_col in numeric_cols:
            print_subsection(f"{num_col} by {group_col}")
            print(
                aff_df.groupby(group_col, dropna=False)[num_col]
                .describe(
                    percentiles=[
                        0.25,
                        0.50,
                        0.75,
                        0.90,
                        0.95,
                    ]
                )
                .to_string()
            )

    print_section("RAW EXAMPLES")

    print_subsection("First raw affordances")
    for _, row in raw_df.head(10).iterrows():
        print(
            json.dumps(
                {
                    "row_index": row["_row_index"],
                    "id_en": row["_id_en"],
                    "pun_word": row["_pun_word"],
                    "pun_type": row["_pun_type"],
                    "left": row["_left"],
                    "right": row["_right"],
                    "relation": row["_relation"],
                    "raw": row["_raw"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

    print_subsection("Raw examples for top reused bridge identities")
    for _, ident_row in identity_counts.head(10).iterrows():
        left = ident_row["_left"]
        right = ident_row["_right"]
        relation = ident_row["_relation"]

        subset = raw_df[
            (raw_df["_left"] == left)
            & (raw_df["_right"] == right)
            & (raw_df["_relation"] == relation)
        ].head(5)

        print(f"\nBRIDGE: {left!r} -> {right!r} / {relation!r}")

        for _, row in subset.iterrows():
            print(
                json.dumps(
                    {
                        "row_index": row["_row_index"],
                        "id_en": row["_id_en"],
                        "pun_word": row["_pun_word"],
                        "pun_type": row["_pun_type"],
                        "text_clean": row["_text_clean"],
                        "raw": row["_raw"],
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()