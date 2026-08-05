#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from config import MODEL_ALIASES
from data import load

pd.options.mode.chained_assignment = None

OUTPUT_ROOT = os.environ.get(
    "DISCRIMINATOR_RUN1_OUTPUT_DIR",
    "../data/processed/discriminate/run1/",
)

JUDGE_KEYS = ["comedian", "pun_expert", "editor", "translator"]


def norm_space(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def ensure_slash(path: str) -> str:
    return str(path or "").rstrip("/") + "/"


def resolve_model_alias(model_arg: str) -> tuple[str, str]:
    model_arg = norm_space(model_arg)
    if model_arg in MODEL_ALIASES and MODEL_ALIASES.get(model_arg):
        return model_arg, MODEL_ALIASES[model_arg]

    filesystem_alias = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_arg).strip("_")
    return filesystem_alias or "model", model_arg


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
    try:
        return json.loads(text)
    except Exception:
        return default


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
        "display.width", 240,
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


def input_dir_for_run(generator_run: str, judge_alias: str) -> str:
    return f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(generator_run)}{ensure_slash(judge_alias)}"


def available_chunks(generator_run: str, judge_alias: str) -> list[int]:
    input_dir = input_dir_for_run(generator_run, judge_alias)
    files = glob.glob(input_dir + "*.tsv")
    chunks: list[int] = []
    for path in files:
        stem = Path(path).stem
        if stem.isdigit():
            chunks.append(int(stem))
    return sorted(chunks)


def load_run1_chunks(generator_run: str, judge_alias: str, start: int = 0, end: int = -1) -> pd.DataFrame:
    input_dir = input_dir_for_run(generator_run, judge_alias)
    chunks = available_chunks(generator_run, judge_alias)
    selected = [c for c in chunks if c >= start and (end == -1 or c < end)]

    if not selected:
        raise FileNotFoundError(
            f"No Run 1 TSV chunks found for generator_run={generator_run}, judge={judge_alias}, "
            f"start={start}, end={end}, input_dir={input_dir}"
        )

    frames = []
    for chunk_num in selected:
        path = f"{input_dir}{chunk_num}.tsv"
        df = load(path).fillna("")
        df["chunk"] = chunk_num
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(out):,} rows from {input_dir}")
    print(f"Chunks: {selected[0]}..{selected[-1]} ({len(selected)} chunks)")
    return out


def parse_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
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
                "pun_len_chars": len(pun),
                "pun_len_words": len(pun.split()),
            })

    return pd.DataFrame(rows)


def parse_rankings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
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


def parse_saved_borda(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        ranking = safe_json_loads(row.get("discriminator_run1_borda_ranking_json"), [])
        if not isinstance(ranking, list):
            continue

        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, dict) or "id" not in item:
                continue

            try:
                cid = int(item["id"])
            except Exception:
                continue

            rows.append({
                "id_en": row.get("id_en"),
                "chunk": row.get("chunk"),
                "candidate_id": cid,
                "saved_borda_rank": rank,
                "saved_borda_score": float(item.get("score", 0.0) or 0.0),
                "saved_borda_pun": norm_space(item.get("pun", "")),
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


def pairwise_order_agreement(a: list[int], b: list[int]) -> float | None:
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return None

    pos_a = {cid: i for i, cid in enumerate(a)}
    pos_b = {cid: i for i, cid in enumerate(b)}

    total = 0
    agree = 0
    for x, y in combinations(common, 2):
        total += 1
        agree += int((pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y]) > 0)

    return agree / total if total else None


def main() -> None:
    if len(sys.argv) < 3:
        raise ValueError(
            """Usage:
  python metrics_run1_candidate_selection.py <generator_run> <judge_model> [start] [end]

Examples:
  python metrics_run1_candidate_selection.py claude gpt 0 41
  python metrics_run1_candidate_selection.py gemini gemini_pro 0 -1"""
        )

    generator_run = sys.argv[1]
    judge_arg = sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    judge_alias, judge_model_id = resolve_model_alias(judge_arg)

    df = load_run1_chunks(generator_run, judge_alias, start, end)
    candidates = parse_candidates(df)
    rankings = parse_rankings(df)
    saved_borda = parse_saved_borda(df)
    equal_borda = calculate_equal_borda(rankings)

    candidate_meta = candidates.drop_duplicates(["id_en", "candidate_id"]).copy()

    top1 = rankings[rankings["rank"] == 1].merge(
        candidate_meta[
            [
                "id_en",
                "candidate_id",
                "candidate_pos",
                "candidate_run",
                "candidate_pun",
                "pun_len_chars",
                "pun_len_words",
            ]
        ],
        on=["id_en", "candidate_id"],
        how="left",
    )

    saved_winners = pd.DataFrame()
    if not saved_borda.empty:
        saved_winners = saved_borda[saved_borda["saved_borda_rank"] == 1].merge(
            candidate_meta[
                [
                    "id_en",
                    "candidate_id",
                    "candidate_pos",
                    "candidate_run",
                    "candidate_pun",
                    "pun_len_chars",
                    "pun_len_words",
                ]
            ],
            on=["id_en", "candidate_id"],
            how="left",
        )

    section("INPUT SUMMARY")
    print_kv([
        ("generator_run", generator_run),
        ("judge_alias", judge_alias),
        ("judge_model_id", judge_model_id),
        ("input_dir", input_dir_for_run(generator_run, judge_alias)),
        ("start", start),
        ("end", end),
        ("raw_rows", len(df)),
        ("unique_id_en", df["id_en"].nunique() if "id_en" in df.columns else ""),
        ("columns", len(df.columns)),
    ])

    section("ERRORS AND VALIDITY")
    rows = []
    for col in ["discriminator_run1_error", "discriminator_run1_borda_error"]:
        if col in df.columns:
            s = df[col].fillna("").astype(str).str.strip()
            rows.append((f"{col}_nonempty", int(s.ne("").sum())))
            rows.append((f"{col}_nonempty_pct", pct(int(s.ne("").sum()), len(df))))

    rows.extend([
        ("parsed_candidate_rows", len(candidates)),
        ("unique_candidate_pairs", candidate_meta.shape[0]),
        ("parsed_ranking_rows", len(rankings)),
        ("parsed_saved_borda_rows", len(saved_borda)),
        ("rows_with_all_4_judge_rankings", int(rankings.groupby("id_en")["judge"].nunique().eq(4).sum()) if not rankings.empty else 0),
        ("rows_with_20_ranked_slots", int(rankings.groupby("id_en").size().eq(20).sum()) if not rankings.empty else 0),
        ("rows_with_saved_borda_winner", len(saved_winners)),
    ])
    print_kv(rows)

    section("CANDIDATE POOL METRICS")
    pool = candidate_meta.groupby("id_en").agg(
        candidate_count=("candidate_id", "count"),
        unique_candidate_puns=("candidate_pun", "nunique"),
        unique_candidate_runs=("candidate_run", "nunique"),
        mean_pun_len_chars=("pun_len_chars", "mean"),
        mean_pun_len_words=("pun_len_words", "mean"),
    ).reset_index()

    print_df(describe_series(pool["candidate_count"], "candidate_count"))
    print_df(count_table(pool["candidate_count"], "candidate_count"))

    subsection("Candidate source run counts")
    print_df(count_table(candidate_meta["candidate_run"], "candidate_run"))

    subsection("Candidate length")
    print_df(describe_series(candidate_meta["pun_len_chars"], "candidate_pun_length_chars"))
    print_df(describe_series(candidate_meta["pun_len_words"], "candidate_pun_length_words"))

    section("RANKING SLOT METRICS")
    print_df(rankings.groupby(["judge", "rank"]).size().reset_index(name="count"))

    subsection("Ranked candidate prompt-position distribution")
    ranked_with_meta = rankings.merge(
        candidate_meta[["id_en", "candidate_id", "candidate_pos", "candidate_run"]],
        on=["id_en", "candidate_id"],
        how="left",
    )
    print_df(ranked_with_meta.groupby(["judge", "rank", "candidate_pos"]).size().reset_index(name="count"))

    section("TOP-1 METRICS BY PERSONA")
    print_df(top1.groupby("judge").agg(
        examples=("id_en", "count"),
        unique_top1_candidates=("candidate_id", "nunique"),
        mean_candidate_pos=("candidate_pos", "mean"),
        median_candidate_pos=("candidate_pos", "median"),
        mean_len_chars=("pun_len_chars", "mean"),
        mean_len_words=("pun_len_words", "mean"),
    ).reset_index())

    subsection("Top-1 by judge and prompt position")
    print_df(top1.groupby(["judge", "candidate_pos"]).size().reset_index(name="count"))

    subsection("Top-1 by judge and candidate source run")
    print_df(top1.groupby(["judge", "candidate_run"]).size().reset_index(name="count"))

    section("PERSONA TOP-1 AGREEMENT")
    top1_wide = top1.pivot_table(
        index="id_en",
        columns="judge",
        values="candidate_id",
        aggfunc="first",
    ).reset_index()

    agree_rows = []
    for j1, j2 in combinations(JUDGE_KEYS, 2):
        valid = top1_wide[[j1, j2]].dropna()
        n = len(valid)
        same = int((valid[j1] == valid[j2]).sum()) if n else 0
        agree_rows.append({
            "judge_1": j1,
            "judge_2": j2,
            "rows": n,
            "same_top1": same,
            "top1_agreement_pct": pct(same, n),
        })
    print_df(pd.DataFrame(agree_rows))

    distinct_top1 = (
        top1.groupby("id_en")["candidate_id"]
        .nunique()
        .reset_index(name="distinct_persona_top1_candidates")
    )

    subsection("Distinct persona top-1 winners per example")
    print_df(count_table(distinct_top1["distinct_persona_top1_candidates"], "distinct_persona_top1_candidates"))
    print_df(describe_series(distinct_top1["distinct_persona_top1_candidates"], "distinct_persona_top1_candidates"))

    section("PERSONA TOP-5 OVERLAP")
    overlap_rows = []
    order_rows = []

    for id_en, g in rankings.groupby("id_en"):
        by_judge_lists = {}
        by_judge_sets = {}
        for judge in JUDGE_KEYS:
            ids = g[g["judge"] == judge].sort_values("rank")["candidate_id"].astype(int).tolist()
            by_judge_lists[judge] = ids
            by_judge_sets[judge] = set(ids)

        for j1, j2 in combinations(JUDGE_KEYS, 2):
            a = by_judge_sets[j1]
            b = by_judge_sets[j2]
            inter = len(a & b)
            union = len(a | b)

            overlap_rows.append({
                "id_en": id_en,
                "judge_1": j1,
                "judge_2": j2,
                "top5_intersection": inter,
                "top5_union": union,
                "top5_jaccard": inter / union if union else 0.0,
            })

            order = pairwise_order_agreement(by_judge_lists[j1], by_judge_lists[j2])
            if order is not None:
                order_rows.append({
                    "id_en": id_en,
                    "judge_1": j1,
                    "judge_2": j2,
                    "common_ranked_candidates": inter,
                    "pairwise_order_agreement": order,
                })

    overlap = pd.DataFrame(overlap_rows)
    order = pd.DataFrame(order_rows)

    print_df(overlap.groupby(["judge_1", "judge_2"]).agg(
        rows=("id_en", "count"),
        mean_intersection=("top5_intersection", "mean"),
        median_intersection=("top5_intersection", "median"),
        mean_jaccard=("top5_jaccard", "mean"),
        median_jaccard=("top5_jaccard", "median"),
    ).reset_index())

    subsection("Top-5 intersection distribution")
    print_df(overlap.groupby(["judge_1", "judge_2", "top5_intersection"]).size().reset_index(name="count"))

    subsection("Pairwise order agreement over shared top-5 candidates")
    if not order.empty:
        print_df(order.groupby(["judge_1", "judge_2"]).agg(
            rows=("id_en", "count"),
            mean_common_candidates=("common_ranked_candidates", "mean"),
            mean_pairwise_order_agreement=("pairwise_order_agreement", "mean"),
            median_pairwise_order_agreement=("pairwise_order_agreement", "median"),
        ).reset_index())
    else:
        print("(empty)")

    section("SAVED WEIGHTED BORDA METRICS")
    if saved_winners.empty:
        print("No saved Borda output found in loaded chunks.")
    else:
        print_kv([
            ("saved_borda_winners", len(saved_winners)),
            ("mean_winner_score", saved_winners["saved_borda_score"].mean()),
            ("median_winner_score", saved_winners["saved_borda_score"].median()),
            ("min_winner_score", saved_winners["saved_borda_score"].min()),
            ("max_winner_score", saved_winners["saved_borda_score"].max()),
        ])

        subsection("Borda winner score distribution")
        print_df(describe_series(saved_winners["saved_borda_score"], "saved_borda_winner_score"))

        subsection("Borda winner by prompt position")
        print_df(count_table(saved_winners["candidate_pos"], "candidate_pos"))

        subsection("Borda winner by candidate source run")
        print_df(count_table(saved_winners["candidate_run"], "candidate_run"))

        subsection("Full Borda ranking score by rank")
        print_df(saved_borda.groupby("saved_borda_rank").agg(
            count=("candidate_id", "count"),
            mean_score=("saved_borda_score", "mean"),
            median_score=("saved_borda_score", "median"),
            min_score=("saved_borda_score", "min"),
            max_score=("saved_borda_score", "max"),
        ).reset_index())

        piv = saved_borda.pivot_table(
            index="id_en",
            columns="saved_borda_rank",
            values="saved_borda_score",
            aggfunc="first",
        )
        if 1 in piv.columns and 2 in piv.columns:
            margin = (piv[1] - piv[2]).dropna()
            subsection("Borda winner margin over second place")
            print_df(describe_series(margin, "borda_margin_rank1_minus_rank2"))

        top1_lookup = top1.set_index(["id_en", "judge"])["candidate_id"].to_dict()

        match_rows = []
        for _, w in saved_winners.iterrows():
            rec = {"id_en": w["id_en"], "winner_id": w["candidate_id"]}
            match_count = 0
            for judge in JUDGE_KEYS:
                matched = top1_lookup.get((w["id_en"], judge)) == w["candidate_id"]
                rec[f"matches_{judge}"] = matched
                match_count += int(bool(matched))
            rec["num_persona_top1_matches"] = match_count
            match_rows.append(rec)

        matches = pd.DataFrame(match_rows)

        subsection("Borda winner matches persona top-1")
        print_df(pd.DataFrame([
            {
                "judge": judge,
                "matches": int(matches[f"matches_{judge}"].sum()),
                "rows": len(matches),
                "match_pct": pct(int(matches[f"matches_{judge}"].sum()), len(matches)),
            }
            for judge in JUDGE_KEYS
        ]))

        subsection("Number of persona top-1 choices matching Borda winner")
        print_df(count_table(matches["num_persona_top1_matches"], "num_persona_top1_matches"))

        top5_sets = rankings.groupby(["id_en", "judge"])["candidate_id"].apply(lambda s: set(s.astype(int))).to_dict()

        top5_rows = []
        for _, w in saved_winners.iterrows():
            rec = {"id_en": w["id_en"], "winner_id": w["candidate_id"]}
            count = 0
            for judge in JUDGE_KEYS:
                present = int(w["candidate_id"]) in top5_sets.get((w["id_en"], judge), set())
                rec[f"in_{judge}_top5"] = present
                count += int(bool(present))
            rec["num_persona_top5_inclusions"] = count
            top5_rows.append(rec)

        top5_matches = pd.DataFrame(top5_rows)

        subsection("Borda winner included in persona top-5")
        print_df(pd.DataFrame([
            {
                "judge": judge,
                "included": int(top5_matches[f"in_{judge}_top5"].sum()),
                "rows": len(top5_matches),
                "included_pct": pct(int(top5_matches[f"in_{judge}_top5"].sum()), len(top5_matches)),
            }
            for judge in JUDGE_KEYS
        ]))

        subsection("Number of persona top-5 lists containing Borda winner")
        print_df(count_table(top5_matches["num_persona_top5_inclusions"], "num_persona_top5_inclusions"))

    section("EQUAL-WEIGHT BORDA METRICS")
    if equal_borda.empty:
        print("No equal-weight Borda could be computed.")
    else:
        equal_winners = equal_borda[equal_borda["equal_borda_rank"] == 1].merge(
            candidate_meta[["id_en", "candidate_id", "candidate_pos", "candidate_run", "candidate_pun"]],
            on=["id_en", "candidate_id"],
            how="left",
        )

        print_kv([
            ("equal_borda_winners", len(equal_winners)),
            ("mean_equal_borda_winner_score", equal_winners["equal_borda_score"].mean()),
            ("median_equal_borda_winner_score", equal_winners["equal_borda_score"].median()),
        ])

        subsection("Equal Borda winner by prompt position")
        print_df(count_table(equal_winners["candidate_pos"], "candidate_pos"))

        subsection("Equal Borda winner by candidate source run")
        print_df(count_table(equal_winners["candidate_run"], "candidate_run"))

        if not saved_winners.empty:
            comp = saved_winners[["id_en", "candidate_id"]].rename(columns={"candidate_id": "saved_winner"}).merge(
                equal_winners[["id_en", "candidate_id"]].rename(columns={"candidate_id": "equal_winner"}),
                on="id_en",
                how="inner",
            )
            same = int((comp["saved_winner"] == comp["equal_winner"]).sum())

            subsection("Saved weighted Borda vs equal-weight Borda")
            print_kv([
                ("compared_rows", len(comp)),
                ("same_winner", same),
                ("same_winner_pct", pct(same, len(comp))),
                ("different_winner", len(comp) - same),
                ("different_winner_pct", pct(len(comp) - same, len(comp))),
            ])

    section("POSITION BIAS DIAGNOSTICS")
    total_by_pos = candidate_meta.groupby("candidate_pos").size().reset_index(name="available_count")
    top1_by_pos = top1.groupby("candidate_pos").size().reset_index(name="top1_count")
    pos = total_by_pos.merge(top1_by_pos, on="candidate_pos", how="left").fillna(0)
    pos["top1_per_available_pct"] = pos["top1_count"] / pos["available_count"] * 100
    print_df(pos)

    if not saved_winners.empty:
        borda_pos = saved_winners.groupby("candidate_pos").size().reset_index(name="borda_winner_count")
        pos2 = total_by_pos.merge(borda_pos, on="candidate_pos", how="left").fillna(0)
        pos2["borda_winner_per_available_pct"] = pos2["borda_winner_count"] / pos2["available_count"] * 100

        subsection("Borda winner rate by prompt position")
        print_df(pos2)

    section("SOURCE-RUN BIAS DIAGNOSTICS")
    total_by_run = candidate_meta.groupby("candidate_run").size().reset_index(name="available_count")
    top1_by_run = top1.groupby("candidate_run").size().reset_index(name="top1_count")
    run = total_by_run.merge(top1_by_run, on="candidate_run", how="left").fillna(0)
    run["top1_per_available_pct"] = run["top1_count"] / run["available_count"] * 100
    print_df(run)

    if not saved_winners.empty:
        borda_run = saved_winners.groupby("candidate_run").size().reset_index(name="borda_winner_count")
        run2 = total_by_run.merge(borda_run, on="candidate_run", how="left").fillna(0)
        run2["borda_winner_per_available_pct"] = run2["borda_winner_count"] / run2["available_count"] * 100

        subsection("Borda winner rate by source run")
        print_df(run2)

    section("MOST COMMON SELECTED PUN TEXTS")
    subsection("Persona top-1 pun text counts")
    print_df(
        top1.groupby("candidate_pun")
        .size()
        .reset_index(name="top1_count")
        .sort_values("top1_count", ascending=False)
        .head(50)
    )

    if not saved_winners.empty:
        subsection("Borda winner pun text counts")
        print_df(
            saved_winners.groupby("candidate_pun")
            .size()
            .reset_index(name="winner_count")
            .sort_values("winner_count", ascending=False)
            .head(50)
        )

    section("DONE")


if __name__ == "__main__":
    main()