#!/usr/bin/env python3
from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from config import MODEL_ALIASES
try:
    from config import retrieval_dir
except Exception:
    retrieval_dir = ""

from data import load, load_all

pd.options.mode.chained_assignment = None

OUTPUT_ROOT = os.environ.get(
    "DISCRIMINATOR_RUN1_OUTPUT_DIR",
    "../data/processed/discriminate/run1/",
)

RETRIEVAL_STATS_INPUT_DIR = os.environ.get("RETRIEVAL_STATS_INPUT_DIR", "").strip()
JUDGE_KEYS = ["comedian", "pun_expert", "editor", "translator"]


def norm_space(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(ch) != "Mn"
    )


def norm_match_text(x: Any) -> str:
    text = strip_accents(norm_space(x).lower())
    text = text.replace("’", "'").replace("ʼ", "'")
    text = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def surface_present(surface: str, text: str) -> bool:
    surface_n = norm_match_text(surface)
    text_n = norm_match_text(text)
    if not surface_n or not text_n:
        return False

    # Multiword expressions can use simple substring matching.
    if " " in surface_n:
        return surface_n in text_n

    # Single words get a loose boundary check.
    return re.search(rf"(?<![a-z0-9]){re.escape(surface_n)}(?![a-z0-9])", text_n) is not None


def safe_json_loads(x: Any, default=None) -> Any:
    if default is None:
        default = {}
    if x is None:
        return default
    try:
        if isinstance(x, float) and pd.isna(x):
            return default
    except Exception:
        pass
    if isinstance(x, (dict, list)):
        return x

    text = str(x).strip()
    if not text:
        return default

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return default


def ensure_slash(path: str) -> str:
    return str(path or "").rstrip("/") + "/"


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subsection(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def print_kv(rows: list[tuple[str, Any]]) -> None:
    width = max((len(str(k)) for k, _ in rows), default=0)
    for k, v in rows:
        if isinstance(v, float):
            print(f"{k:<{width}} : {v:.6f}")
        else:
            print(f"{k:<{width}} : {v}")


def print_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("(empty)")
        return
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 260,
        "display.max_colwidth", 140,
    ):
        print(df.to_string(index=False))


def count_table(series: pd.Series, name: str) -> pd.DataFrame:
    out = series.value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    out["pct"] = out["count"] / out["count"].sum() * 100
    return out


def describe_series(series: pd.Series, name: str) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame([{"field": name, "count": 0}])
    qs = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return pd.DataFrame([{
        "field": name,
        "count": len(s),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "1%": qs.loc[0.01],
        "5%": qs.loc[0.05],
        "10%": qs.loc[0.10],
        "25%": qs.loc[0.25],
        "50%": qs.loc[0.50],
        "75%": qs.loc[0.75],
        "90%": qs.loc[0.90],
        "95%": qs.loc[0.95],
        "99%": qs.loc[0.99],
        "max": s.max(),
    }])


def resolve_model_alias(model_arg: str) -> tuple[str, str]:
    model_arg = norm_space(model_arg)
    if model_arg in MODEL_ALIASES and MODEL_ALIASES.get(model_arg):
        return model_arg, MODEL_ALIASES[model_arg]
    filesystem_alias = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_arg).strip("_")
    return filesystem_alias or "model", model_arg


def resolve_retrieval_input_dir() -> str:
    if RETRIEVAL_STATS_INPUT_DIR:
        return RETRIEVAL_STATS_INPUT_DIR.rstrip("/") + "/"
    if retrieval_dir:
        return f"{ensure_slash(retrieval_dir)}gemini/"
    root = Path(__file__).resolve().parents[1]
    return str(root / "data" / "processed" / "retrieval" / "gemini") + "/"


def input_dir_for_run(generator_run: str, judge_alias: str) -> str:
    return f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(generator_run)}{ensure_slash(judge_alias)}"


def available_chunks(generator_run: str, judge_alias: str) -> list[int]:
    input_dir = input_dir_for_run(generator_run, judge_alias)
    chunks = []
    for path in glob.glob(input_dir + "*.tsv"):
        stem = Path(path).stem
        if stem.isdigit():
            chunks.append(int(stem))
    return sorted(chunks)


def load_run1_chunks(generator_run: str, judge_alias: str, start: int, end: int) -> pd.DataFrame:
    input_dir = input_dir_for_run(generator_run, judge_alias)
    chunks = available_chunks(generator_run, judge_alias)
    selected = [c for c in chunks if c >= start and (end == -1 or c < end)]

    if not selected:
        raise FileNotFoundError(f"No Run 1 chunks found in {input_dir}")

    frames = []
    for chunk_num in selected:
        path = f"{input_dir}{chunk_num}.tsv"
        df = load(path).fillna("")
        df["chunk"] = chunk_num
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(out):,} Run 1 rows from {input_dir}")
    print(f"Chunks: {selected[0]}..{selected[-1]} ({len(selected)} chunks)")
    return out


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def parse_retrieval_affordances(row: pd.Series) -> list[dict[str, Any]]:
    items = []

    direct = safe_json_loads(row.get("retrieval_affordances_json"), [])
    if isinstance(direct, list):
        items.extend([x for x in direct if isinstance(x, dict)])

    bridge_candidates = safe_json_loads(row.get("bridge_candidates"), [])
    if isinstance(bridge_candidates, list):
        items.extend([x for x in bridge_candidates if isinstance(x, dict)])

    for col in ("generator_affordance_pack", "retrieval_pack_compact"):
        value = safe_json_loads(row.get(col), {})
        if isinstance(value, dict):
            top = value.get("top_bridge_candidates")
            if isinstance(top, list):
                items.extend([x for x in top if isinstance(x, dict)])

            nested = value.get("generator_affordance_pack")
            if isinstance(nested, dict):
                nested_top = nested.get("top_bridge_candidates")
                if isinstance(nested_top, list):
                    items.extend([x for x in nested_top if isinstance(x, dict)])

    # Deduplicate inside row by left/right/relation.
    seen = set()
    out = []
    for item in items:
        left, right, relation = bridge_identity(item)
        key = (left, right, relation)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


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


def load_retrieval() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = resolve_retrieval_input_dir()
    print(f"Loading retrieval rows from {input_dir.rstrip('/')}")
    df = load_all(input_dir).fillna("")

    aff_rows = []
    row_rows = []

    for row_index, row in df.iterrows():
        affs = parse_retrieval_affordances(row)

        row_rows.append({
            "id_en": row.get("id_en"),
            "retrieval_row_index": row_index,
            "pun_word": row.get("pun_word", ""),
            "pun_type": row.get("pun_type", ""),
            "text_clean_retrieval": row.get("text_clean", ""),
            "retrieval_affordance_count_parsed": len(affs),
            "has_retrieval_affordance": len(affs) > 0,
        })

        for aff_index, aff in enumerate(affs):
            left, right, relation = bridge_identity(aff)
            flat = flatten_dict(aff)
            scores = aff.get("scores") if isinstance(aff.get("scores"), dict) else {}

            aff_rows.append({
                "id_en": row.get("id_en"),
                "retrieval_row_index": row_index,
                "affordance_index": aff_index,
                "left": left,
                "right": right,
                "relation": relation,
                "identity": json.dumps([left, right, relation], ensure_ascii=False),
                "retrieval_bucket": aff.get("retrieval_bucket", ""),
                "retrieval_bucket_rank": aff.get("retrieval_bucket_rank", ""),
                "export_lane": aff.get("export_lane", ""),
                "score_overall": scores.get("overall_score", flat.get("scores.overall_score", pd.NA)),
                "score_phonetic": scores.get("phonetic_match", flat.get("scores.phonetic_match", pd.NA)),
                "score_naturalness": scores.get("french_naturalness", flat.get("scores.french_naturalness", pd.NA)),
                "score_semantic_domain": scores.get("semantic_domain_similarity", flat.get("scores.semantic_domain_similarity", pd.NA)),
                "score_pivot": scores.get("pun_pivot_usability", flat.get("scores.pun_pivot_usability", pd.NA)),
                "score_surprise": scores.get("semantic_surprise", flat.get("scores.semantic_surprise", pd.NA)),
            })

    return pd.DataFrame(row_rows), pd.DataFrame(aff_rows)


def parse_candidates(run1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in run1.iterrows():
        candidates = safe_json_loads(row.get("shuffled_candidates_json"), [])
        if not isinstance(candidates, list):
            continue

        for pos, item in enumerate(candidates, start=1):
            if not isinstance(item, dict) or "id" not in item:
                continue
            pun = norm_space(item.get("pun", ""))
            try:
                cid = int(item["id"])
            except Exception:
                continue

            rows.append({
                "id_en": row.get("id_en"),
                "chunk": row.get("chunk"),
                "candidate_id": cid,
                "candidate_pos": pos,
                "candidate_pun": pun,
                "candidate_run": norm_space(item.get("run", "")),
            })
    return pd.DataFrame(rows)


def parse_rankings(run1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in run1.iterrows():
        rankings = safe_json_loads(row.get("discriminator_run1_json"), {})
        if not isinstance(rankings, dict):
            continue

        for judge in JUDGE_KEYS:
            ids = rankings.get(judge, [])
            if not isinstance(ids, list):
                continue
            for rank, cid in enumerate(ids, start=1):
                try:
                    cid = int(cid)
                except Exception:
                    continue
                rows.append({
                    "id_en": row.get("id_en"),
                    "chunk": row.get("chunk"),
                    "judge": judge,
                    "candidate_id": cid,
                    "rank": rank,
                    "points": 6 - rank,
                    "is_top1": rank == 1,
                })
    return pd.DataFrame(rows)


def parse_saved_borda(run1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in run1.iterrows():
        ranking = safe_json_loads(row.get("discriminator_run1_borda_ranking_json"), [])
        if not isinstance(ranking, list):
            continue
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, dict) or "id" not in item:
                continue
            rows.append({
                "id_en": row.get("id_en"),
                "chunk": row.get("chunk"),
                "candidate_id": int(item["id"]),
                "saved_borda_rank": rank,
                "saved_borda_score": float(item.get("score", 0.0) or 0.0),
            })
    return pd.DataFrame(rows)


def calculate_equal_borda(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()

    out = (
        rankings.groupby(["id_en", "candidate_id"], as_index=False)
        .agg(equal_borda_score=("points", "sum"))
    )
    out = out.sort_values(
        ["id_en", "equal_borda_score", "candidate_id"],
        ascending=[True, False, True],
    )
    out["equal_borda_rank"] = out.groupby("id_en").cumcount() + 1
    return out


def attach_affordance_usage(candidates: pd.DataFrame, aff: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if candidates.empty or aff.empty:
        return pd.DataFrame()

    aff_by_id = {id_en: g for id_en, g in aff.groupby("id_en", dropna=False)}

    for _, cand in candidates.iterrows():
        id_en = cand["id_en"]
        pun = cand["candidate_pun"]
        affs = aff_by_id.get(id_en)

        if affs is None or affs.empty:
            continue

        for _, a in affs.iterrows():
            left_used = surface_present(a["left"], pun)
            right_used = surface_present(a["right"], pun)

            if not left_used and not right_used:
                continue

            rows.append({
                "id_en": id_en,
                "candidate_id": cand["candidate_id"],
                "candidate_pos": cand["candidate_pos"],
                "candidate_run": cand["candidate_run"],
                "candidate_pun": pun,
                "affordance_index": a["affordance_index"],
                "left": a["left"],
                "right": a["right"],
                "relation": a["relation"],
                "identity": a["identity"],
                "left_used": left_used,
                "right_used": right_used,
                "both_sides_used": left_used and right_used,
                "usage_type": "both" if left_used and right_used else ("left_only" if left_used else "right_only"),
                "retrieval_bucket": a.get("retrieval_bucket", ""),
                "retrieval_bucket_rank": a.get("retrieval_bucket_rank", ""),
                "export_lane": a.get("export_lane", ""),
                "score_overall": a.get("score_overall", pd.NA),
                "score_phonetic": a.get("score_phonetic", pd.NA),
                "score_naturalness": a.get("score_naturalness", pd.NA),
                "score_semantic_domain": a.get("score_semantic_domain", pd.NA),
                "score_pivot": a.get("score_pivot", pd.NA),
                "score_surprise": a.get("score_surprise", pd.NA),
            })

    return pd.DataFrame(rows)


def selected_sets(candidates: pd.DataFrame, rankings: pd.DataFrame, saved_borda: pd.DataFrame, equal_borda: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}

    out["all_generated"] = candidates.copy()

    if not rankings.empty:
        ranked = candidates.merge(
            rankings[["id_en", "candidate_id", "judge", "rank", "points", "is_top1"]],
            on=["id_en", "candidate_id"],
            how="inner",
        )
        out["ranked_top5_slots"] = ranked

        for judge in JUDGE_KEYS:
            out[f"{judge}_top1"] = ranked[(ranked["judge"] == judge) & (ranked["rank"] == 1)].copy()

    if not saved_borda.empty:
        out["saved_borda_winner"] = candidates.merge(
            saved_borda[saved_borda["saved_borda_rank"] == 1][["id_en", "candidate_id", "saved_borda_score"]],
            on=["id_en", "candidate_id"],
            how="inner",
        )

    if not equal_borda.empty:
        out["equal_borda_winner"] = candidates.merge(
            equal_borda[equal_borda["equal_borda_rank"] == 1][["id_en", "candidate_id", "equal_borda_score"]],
            on=["id_en", "candidate_id"],
            how="inner",
        )

    return out


def print_usage_summary(label: str, subset: pd.DataFrame, usage: pd.DataFrame, aff: pd.DataFrame) -> None:
    subsection(label)

    if subset.empty:
        print("(empty subset)")
        return

    keys = subset[["id_en", "candidate_id"]].drop_duplicates()
    used = usage.merge(keys, on=["id_en", "candidate_id"], how="inner") if not usage.empty else pd.DataFrame()

    rows_total = subset["id_en"].nunique()
    cand_total = keys.shape[0]
    rows_with_any_usage = used["id_en"].nunique() if not used.empty else 0
    candidates_with_any_usage = used[["id_en", "candidate_id"]].drop_duplicates().shape[0] if not used.empty else 0
    distinct_affordances_used = used[["id_en", "identity"]].drop_duplicates().shape[0] if not used.empty else 0

    print_kv([
        ("examples", rows_total),
        ("candidate_instances", cand_total),
        ("examples_with_any_affordance_used", rows_with_any_usage),
        ("examples_with_any_affordance_used_pct", pct(rows_with_any_usage, rows_total)),
        ("candidate_instances_with_any_affordance_used", candidates_with_any_usage),
        ("candidate_instances_with_any_affordance_used_pct", pct(candidates_with_any_usage, cand_total)),
        ("distinct_row_affordances_used", distinct_affordances_used),
        ("total_retrieved_row_affordances", aff[["id_en", "identity"]].drop_duplicates().shape[0] if not aff.empty else 0),
        ("retrieved_row_affordance_usage_pct", pct(distinct_affordances_used, aff[["id_en", "identity"]].drop_duplicates().shape[0] if not aff.empty else 0)),
    ])

    if used.empty:
        return

    per_candidate = (
        used.groupby(["id_en", "candidate_id"])
        .agg(
            matched_affordances=("identity", "nunique"),
            matched_left_terms=("left_used", "sum"),
            matched_right_terms=("right_used", "sum"),
            matched_both_sides=("both_sides_used", "sum"),
        )
        .reset_index()
    )

    print("\nMatched affordances per candidate:")
    print_df(describe_series(per_candidate["matched_affordances"], "matched_affordances_per_candidate"))

    print("\nUsage type:")
    print_df(count_table(used["usage_type"], "usage_type"))

    print("\nUsed affordances by retrieval_bucket:")
    print_df(count_table(used["retrieval_bucket"], "retrieval_bucket"))

    print("\nUsed affordances by relation:")
    print_df(count_table(used["relation"], "relation"))

    score_cols = [
        c for c in [
            "score_overall",
            "score_phonetic",
            "score_naturalness",
            "score_semantic_domain",
            "score_pivot",
            "score_surprise",
        ]
        if c in used.columns
    ]
    if score_cols:
        print("\nUsed affordance score summaries:")
        print_df(pd.concat([describe_series(used[c], c) for c in score_cols], ignore_index=True))

    print("\nTop used affordance identities:")
    top = (
        used.groupby(["left", "right", "relation"], dropna=False)
        .agg(
            uses=("candidate_id", "count"),
            examples=("id_en", "nunique"),
            candidate_instances=("candidate_id", "nunique"),
            mean_score_overall=("score_overall", "mean"),
        )
        .reset_index()
        .sort_values(["uses", "examples"], ascending=False)
        .head(30)
    )
    print_df(top)


def main() -> None:
    if len(sys.argv) < 3:
        raise ValueError(
            """Usage:
  python run1_affordance_usage_metrics.py <generator_run> <judge_model> [start] [end]

Examples:
  python run1_affordance_usage_metrics.py claude claude 0 -1
  python run1_affordance_usage_metrics.py gemini gemini 0 -1"""
        )

    generator_run = sys.argv[1]
    judge_arg = sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    judge_alias, judge_model_id = resolve_model_alias(judge_arg)

    retrieval_rows, aff = load_retrieval()
    run1 = load_run1_chunks(generator_run, judge_alias, start, end)

    candidates = parse_candidates(run1)
    rankings = parse_rankings(run1)
    saved_borda = parse_saved_borda(run1)
    equal_borda = calculate_equal_borda(rankings)

    usage = attach_affordance_usage(candidates, aff)

    section("INPUT SUMMARY")
    print_kv([
        ("generator_run", generator_run),
        ("judge_alias", judge_alias),
        ("judge_model_id", judge_model_id),
        ("retrieval_rows", len(retrieval_rows)),
        ("retrieval_rows_with_affordances", int(retrieval_rows["has_retrieval_affordance"].sum()) if not retrieval_rows.empty else 0),
        ("retrieval_affordances", len(aff)),
        ("run1_rows", len(run1)),
        ("candidate_instances", len(candidates)),
        ("ranking_slots", len(rankings)),
        ("saved_borda_rows", len(saved_borda)),
        ("equal_borda_rows", len(equal_borda)),
        ("usage_matches", len(usage)),
    ])

    section("ROW ALIGNMENT")
    run1_ids = set(run1["id_en"].astype(str))
    ret_ids = set(retrieval_rows["id_en"].astype(str))
    cand_ids = set(candidates["id_en"].astype(str))

    print_kv([
        ("run1_unique_id_en", len(run1_ids)),
        ("retrieval_unique_id_en", len(ret_ids)),
        ("candidate_unique_id_en", len(cand_ids)),
        ("run1_ids_found_in_retrieval", len(run1_ids & ret_ids)),
        ("run1_ids_missing_from_retrieval", len(run1_ids - ret_ids)),
        ("retrieval_ids_not_in_run1", len(ret_ids - run1_ids)),
    ])

    section("RETRIEVAL AVAILABILITY IN RUN1 SET")
    run1_retrieval = retrieval_rows[retrieval_rows["id_en"].astype(str).isin(run1_ids)].copy()

    print_df(describe_series(run1_retrieval["retrieval_affordance_count_parsed"], "retrieval_affordance_count_for_run1_rows"))
    print_df(count_table(run1_retrieval["retrieval_affordance_count_parsed"], "retrieval_affordance_count"))

    section("AFFORDANCE USAGE ACROSS GENERATED AND SELECTED CANDIDATES")
    sets = selected_sets(candidates, rankings, saved_borda, equal_borda)

    for label, subset in sets.items():
        print_usage_summary(label, subset, usage, aff)

    section("COMPARISON: AFFORDANCE USE BY PERSONA TOP1")
    persona_rows = []
    for judge in JUDGE_KEYS:
        label = f"{judge}_top1"
        subset = sets.get(label, pd.DataFrame())
        if subset.empty:
            continue
        keys = subset[["id_en", "candidate_id"]].drop_duplicates()
        used = usage.merge(keys, on=["id_en", "candidate_id"], how="inner") if not usage.empty else pd.DataFrame()
        persona_rows.append({
            "selection_set": label,
            "examples": subset["id_en"].nunique(),
            "candidates": keys.shape[0],
            "candidates_with_affordance_used": used[["id_en", "candidate_id"]].drop_duplicates().shape[0] if not used.empty else 0,
            "candidates_with_affordance_used_pct": pct(
                used[["id_en", "candidate_id"]].drop_duplicates().shape[0] if not used.empty else 0,
                keys.shape[0],
            ),
            "mean_matched_affordances_per_used_candidate": (
                used.groupby(["id_en", "candidate_id"])["identity"].nunique().mean()
                if not used.empty else 0.0
            ),
            "both_side_usage_count": int(used["both_sides_used"].sum()) if not used.empty else 0,
        })

    if "equal_borda_winner" in sets:
        subset = sets["equal_borda_winner"]
        keys = subset[["id_en", "candidate_id"]].drop_duplicates()
        used = usage.merge(keys, on=["id_en", "candidate_id"], how="inner") if not usage.empty else pd.DataFrame()
        persona_rows.append({
            "selection_set": "equal_borda_winner",
            "examples": subset["id_en"].nunique(),
            "candidates": keys.shape[0],
            "candidates_with_affordance_used": used[["id_en", "candidate_id"]].drop_duplicates().shape[0] if not used.empty else 0,
            "candidates_with_affordance_used_pct": pct(
                used[["id_en", "candidate_id"]].drop_duplicates().shape[0] if not used.empty else 0,
                keys.shape[0],
            ),
            "mean_matched_affordances_per_used_candidate": (
                used.groupby(["id_en", "candidate_id"])["identity"].nunique().mean()
                if not used.empty else 0.0
            ),
            "both_side_usage_count": int(used["both_sides_used"].sum()) if not used.empty else 0,
        })

    if "saved_borda_winner" in sets:
        subset = sets["saved_borda_winner"]
        keys = subset[["id_en", "candidate_id"]].drop_duplicates()
        used = usage.merge(keys, on=["id_en", "candidate_id"], how="inner") if not usage.empty else pd.DataFrame()
        persona_rows.append({
            "selection_set": "saved_borda_winner",
            "examples": subset["id_en"].nunique(),
            "candidates": keys.shape[0],
            "candidates_with_affordance_used": used[["id_en", "candidate_id"]].drop_duplicates().shape[0] if not used.empty else 0,
            "candidates_with_affordance_used_pct": pct(
                used[["id_en", "candidate_id"]].drop_duplicates().shape[0] if not used.empty else 0,
                keys.shape[0],
            ),
            "mean_matched_affordances_per_used_candidate": (
                used.groupby(["id_en", "candidate_id"])["identity"].nunique().mean()
                if not used.empty else 0.0
            ),
            "both_side_usage_count": int(used["both_sides_used"].sum()) if not used.empty else 0,
        })

    print_df(pd.DataFrame(persona_rows))

    section("AFFORDANCE USE BY RETRIEVAL SOURCE VS SELECTION")
    if not usage.empty:
        rows = []
        for label, subset in sets.items():
            keys = subset[["id_en", "candidate_id"]].drop_duplicates()
            used = usage.merge(keys, on=["id_en", "candidate_id"], how="inner")
            if used.empty:
                continue
            for group_col in ["retrieval_bucket", "export_lane", "relation", "usage_type"]:
                if group_col not in used:
                    continue
                tmp = used.groupby(group_col).size().reset_index(name="uses")
                tmp["selection_set"] = label
                tmp["group_col"] = group_col
                tmp = tmp.rename(columns={group_col: "group_value"})
                rows.append(tmp[["selection_set", "group_col", "group_value", "uses"]])
        if rows:
            print_df(pd.concat(rows, ignore_index=True))
        else:
            print("(empty)")
    else:
        print("(empty)")

    section("DONE")


if __name__ == "__main__":
    main()