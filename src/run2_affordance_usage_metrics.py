#!/usr/bin/env python3
from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from config import retrieval_dir
except Exception:
    retrieval_dir = ""

from data import load, load_all

pd.options.mode.chained_assignment = None

OUTPUT_ROOT = os.environ.get(
    "DISCRIMINATOR_RUN2_OUTPUT_DIR",
    "../data/processed/discriminate/run2/",
)
ENSEMBLE_RUN = os.environ.get("DISCRIMINATOR_RUN2_ENSEMBLE_RUN", "ensemble")
RETRIEVAL_STATS_INPUT_DIR = os.environ.get("RETRIEVAL_STATS_INPUT_DIR", "").strip()

BORDA_METHODS = ["judges_then_models", "models_then_judges", "pooled_rankings"]
BORDA_ALIASES = {
    "judges_then_models": "judges_then_models",
    "judges_first": "judges_then_models",
    "judge_first": "judges_then_models",
    "models_then_judges": "models_then_judges",
    "models_first": "models_then_judges",
    "model_first": "models_then_judges",
    "pooled_rankings": "pooled_rankings",
    "pooled": "pooled_rankings",
    "pool": "pooled_rankings",
}


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
    return re.sub(r"\s+", " ", text).strip()


def surface_present(surface: str, text: str) -> bool:
    surface_n = norm_match_text(surface)
    text_n = norm_match_text(text)
    if not surface_n or not text_n:
        return False
    if " " in surface_n:
        return surface_n in text_n
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
        print(f"{k:<{width}} : {v:.6f}" if isinstance(v, float) else f"{k:<{width}} : {v}")


def print_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("(empty)")
        return
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 260,
        "display.max_colwidth", 160,
    ):
        print(df.to_string(index=False))


def count_table(series: pd.Series, name: str) -> pd.DataFrame:
    out = series.fillna("").astype(str).value_counts(dropna=False).rename_axis(name).reset_index(name="count")
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


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def resolve_retrieval_input_dir() -> str:
    if RETRIEVAL_STATS_INPUT_DIR:
        return ensure_slash(RETRIEVAL_STATS_INPUT_DIR)
    if retrieval_dir:
        return f"{ensure_slash(retrieval_dir)}gemini/"
    root = Path(__file__).resolve().parents[1]
    return str(root / "data" / "processed" / "retrieval" / "gemini") + "/"


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


def load_retrieval() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = resolve_retrieval_input_dir()
    print(f"Loading retrieval rows from {input_dir.rstrip('/')}")
    df = load_all(input_dir).fillna("")

    row_rows = []
    aff_rows = []

    for row_index, row in df.iterrows():
        affs = parse_retrieval_affordances(row)
        id_en = row.get("id_en")

        row_rows.append({
            "id_en": id_en,
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
                "id_en": id_en,
                "retrieval_row_index": row_index,
                "affordance_index": aff_index,
                "identity": f"{left}|||{right}|||{relation}",
                "left": left,
                "right": right,
                "relation": relation,
                "retrieval_bucket": norm_space(
                    aff.get("retrieval_bucket")
                    or aff.get("bucket")
                    or aff.get("source_bucket")
                    or aff.get("export_lane")
                ),
                "retrieval_bucket_rank": aff.get("retrieval_bucket_rank", ""),
                "export_lane": norm_space(aff.get("export_lane")),
                "score_overall": scores.get("overall_score", flat.get("scores.overall_score", pd.NA)),
                "score_phonetic": scores.get("phonetic_match", flat.get("scores.phonetic_match", pd.NA)),
                "score_naturalness": scores.get("french_naturalness", flat.get("scores.french_naturalness", pd.NA)),
                "score_semantic_domain": scores.get(
                    "semantic_domain_similarity",
                    flat.get("scores.semantic_domain_similarity", pd.NA),
                ),
                "score_pivot": scores.get("pun_pivot_usability", flat.get("scores.pun_pivot_usability", pd.NA)),
                "score_surprise": scores.get("semantic_surprise", flat.get("scores.semantic_surprise", pd.NA)),
            })

    return pd.DataFrame(row_rows), pd.DataFrame(aff_rows)


def chunk_numbers_in_dir(path: str) -> list[int]:
    chunks = []
    for file_path in glob.glob(ensure_slash(path) + "*.tsv"):
        stem = Path(file_path).stem
        if stem.isdigit():
            chunks.append(int(stem))
    return sorted(chunks)


def borda_input_dir(method: str, persona_weights: str, judge_model_weights: str) -> str:
    method = BORDA_ALIASES.get(method, method)
    if method not in BORDA_METHODS:
        raise ValueError(f"Unknown method={method}. Expected one of: {', '.join(BORDA_METHODS)}")

    return (
        f"{ensure_slash(OUTPUT_ROOT)}"
        f"{ensure_slash(ENSEMBLE_RUN)}"
        f"borda/"
        f"{ensure_slash(method)}"
        f"{ensure_slash(persona_weights)}"
        f"{ensure_slash(judge_model_weights)}"
    )


def load_borda_chunks(path: str, start: int = 0, end: int = -1) -> pd.DataFrame:
    chunks = chunk_numbers_in_dir(path)
    selected = [c for c in chunks if c >= start and (end == -1 or c < end)]

    if not selected:
        raise FileNotFoundError(f"No Run 2 Borda chunks found in {path}")

    frames = []
    for chunk in selected:
        file_path = ensure_slash(path) + f"{chunk}.tsv"
        df = load(file_path).fillna("")
        df["chunk"] = chunk
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(out):,} Run 2 Borda rows from {path}")
    print(f"Chunks: {selected[0]}..{selected[-1]} ({len(selected)} chunks)")
    return out


def parse_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        raw = safe_json_loads(row.get("shuffled_candidates_json"), [])
        if not isinstance(raw, list):
            continue

        for pos, item in enumerate(raw, start=1):
            if not isinstance(item, dict) or "id" not in item:
                continue

            try:
                cid = int(item.get("id"))
            except Exception:
                continue

            pun = norm_space(item.get("pun", ""))
            if not pun:
                continue

            rows.append({
                "id_en": norm_space(row.get("id_en")),
                "chunk": row.get("chunk"),
                "candidate_id": cid,
                "candidate_pos": pos,
                "candidate_source": norm_space(item.get("source")),
                "candidate_original_id": norm_space(item.get("original_id")),
                "candidate_pun": pun,
                "pun_len_chars": len(pun),
                "pun_len_words": len(pun.split()),
            })

    return pd.DataFrame(rows)


def parse_borda_ranking(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        ranking = safe_json_loads(row.get("discriminator_run2_borda_ranking_json"), [])
        if not isinstance(ranking, list):
            continue

        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, dict) or "id" not in item:
                continue
            try:
                cid = int(item.get("id"))
            except Exception:
                continue

            rows.append({
                "id_en": norm_space(row.get("id_en")),
                "chunk": row.get("chunk"),
                "candidate_id": cid,
                "borda_rank": rank,
                "borda_score": float(item.get("score", 0.0) or 0.0),
                "candidate_source": norm_space(item.get("source")),
                "candidate_original_id": norm_space(item.get("original_id")),
                "candidate_pun": norm_space(item.get("pun")),
            })

    return pd.DataFrame(rows)


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
            left = norm_space(a.get("left"))
            right = norm_space(a.get("right"))

            left_used = surface_present(left, pun)
            right_used = surface_present(right, pun)

            if not left_used and not right_used:
                continue

            if left_used and right_used:
                usage_type = "both"
            elif left_used:
                usage_type = "left_only"
            else:
                usage_type = "right_only"

            out = cand.to_dict()
            for col in a.index:
                out[col] = a[col]
            out.update({
                "left_used": bool(left_used),
                "right_used": bool(right_used),
                "both_sides_used": bool(left_used and right_used),
                "usage_type": usage_type,
            })
            rows.append(out)

    return pd.DataFrame(rows)


def selected_sets(candidates: pd.DataFrame, borda: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"all_finalists": candidates.copy()}

    if not borda.empty:
        ranked = candidates.merge(
            borda[["id_en", "candidate_id", "borda_rank", "borda_score"]],
            on=["id_en", "candidate_id"],
            how="inner",
        )
        out["borda_ranked_all"] = ranked
        out["borda_winner"] = ranked[ranked["borda_rank"] == 1].copy()

    for source in sorted(candidates["candidate_source"].dropna().astype(str).unique()) if not candidates.empty else []:
        out[f"source_{source}"] = candidates[candidates["candidate_source"] == source].copy()

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
    candidates_with_usage = used[["id_en", "candidate_id"]].drop_duplicates().shape[0] if not used.empty else 0
    examples_with_usage = used["id_en"].nunique() if not used.empty else 0
    distinct_affordances_used = used[["id_en", "identity"]].drop_duplicates().shape[0] if not used.empty else 0
    total_affordances = aff[["id_en", "identity"]].drop_duplicates().shape[0] if not aff.empty else 0

    print_kv([
        ("examples", rows_total),
        ("candidate_instances", cand_total),
        ("examples_with_any_affordance_used", examples_with_usage),
        ("examples_with_any_affordance_used_pct", pct(examples_with_usage, rows_total)),
        ("candidate_instances_with_any_affordance_used", candidates_with_usage),
        ("candidate_instances_with_any_affordance_used_pct", pct(candidates_with_usage, cand_total)),
        ("distinct_row_affordances_used", distinct_affordances_used),
        ("total_retrieved_row_affordances", total_affordances),
        ("retrieved_row_affordance_usage_pct", pct(distinct_affordances_used, total_affordances)),
    ])

    if used.empty:
        return

    per_candidate = (
        used.groupby(["id_en", "candidate_id"])
        .agg(matched_affordances=("identity", "nunique"))
        .reset_index()
    )

    print("\nMatched affordances per candidate:")
    print_df(describe_series(per_candidate["matched_affordances"], "matched_affordances_per_candidate"))

    print("\nUsage type:")
    print_df(count_table(used["usage_type"], "usage_type"))

    print("\nUsed affordances by candidate source:")
    print_df(count_table(used["candidate_source"], "candidate_source"))

    print("\nUsed affordances by retrieval bucket:")
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


def main() -> None:
    if len(sys.argv) < 3:
        raise ValueError(
            """Usage:
  python run2_affordance_usage_metrics_v2.py <persona_weights> <judge_model_weights> [method] [start] [end]

Examples:
  python run2_affordance_usage_metrics_v2.py 25_25_25_25 25_25_25
  python run2_affordance_usage_metrics_v2.py 25_25_25_25 25_25_25 pooled_rankings 0 -1
  python run2_affordance_usage_metrics_v2.py 0_0_0_100 25_25_25 pooled_rankings 0 -1
"""
        )

    persona_weights = sys.argv[1]
    judge_model_weights = sys.argv[2]
    method = BORDA_ALIASES.get(sys.argv[3], sys.argv[3]) if len(sys.argv) > 3 else "pooled_rankings"
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    end = int(sys.argv[5]) if len(sys.argv) > 5 else -1

    input_dir = borda_input_dir(method, persona_weights, judge_model_weights)

    retrieval_rows, aff = load_retrieval()
    run2 = load_borda_chunks(input_dir, start, end)

    candidates = parse_candidates(run2)
    borda = parse_borda_ranking(run2)
    usage = attach_affordance_usage(candidates, aff)
    sets = selected_sets(candidates, borda)

    section("INPUT SUMMARY")
    print_kv([
        ("ensemble_run", ENSEMBLE_RUN),
        ("method", method),
        ("persona_weights", persona_weights),
        ("judge_model_weights", judge_model_weights),
        ("input_dir", input_dir),
        ("retrieval_rows", len(retrieval_rows)),
        ("retrieval_rows_with_affordances", int(retrieval_rows["has_retrieval_affordance"].sum()) if not retrieval_rows.empty else 0),
        ("retrieval_affordances", len(aff)),
        ("run2_rows", len(run2)),
        ("candidate_instances", len(candidates)),
        ("borda_ranking_rows", len(borda)),
        ("usage_matches", len(usage)),
    ])

    section("ROW ALIGNMENT")
    run2_ids = set(run2["id_en"].astype(str))
    ret_ids = set(retrieval_rows["id_en"].astype(str))
    cand_ids = set(candidates["id_en"].astype(str)) if not candidates.empty else set()

    print_kv([
        ("run2_unique_id_en", len(run2_ids)),
        ("retrieval_unique_id_en", len(ret_ids)),
        ("candidate_unique_id_en", len(cand_ids)),
        ("run2_ids_found_in_retrieval", len(run2_ids & ret_ids)),
        ("run2_ids_missing_from_retrieval", len(run2_ids - ret_ids)),
        ("retrieval_ids_not_in_run2", len(ret_ids - run2_ids)),
    ])

    section("CANDIDATE SOURCE DISTRIBUTION")
    if not candidates.empty:
        print_df(count_table(candidates["candidate_source"], "candidate_source"))

    section("AFFORDANCE USAGE ACROSS RUN 2 FINALISTS AND SELECTED WINNERS")
    for label, subset in sets.items():
        print_usage_summary(label, subset, usage, aff)

    section("AFFORDANCE USE BY SOURCE VS SELECTION")
    rows = []
    for label, subset in sets.items():
        keys = subset[["id_en", "candidate_id"]].drop_duplicates()
        used = usage.merge(keys, on=["id_en", "candidate_id"], how="inner") if not usage.empty else pd.DataFrame()
        if used.empty:
            continue

        for source, g in used.groupby("candidate_source", dropna=False):
            rows.append({
                "selection_set": label,
                "candidate_source": source,
                "usage_matches": len(g),
                "examples": g["id_en"].nunique(),
                "candidate_instances": g[["id_en", "candidate_id"]].drop_duplicates().shape[0],
                "mean_score_overall": pd.to_numeric(g["score_overall"], errors="coerce").mean()
                    if "score_overall" in g.columns else pd.NA,
            })

    print_df(pd.DataFrame(rows))


if __name__ == "__main__":
    main()