#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from data import load

pd.options.mode.chained_assignment = None

OUTPUT_ROOT = os.environ.get(
    "DISCRIMINATOR_RUN2_OUTPUT_DIR",
    "../data/processed/discriminate/run2/",
)

ENSEMBLE_RUN = os.environ.get("DISCRIMINATOR_RUN2_ENSEMBLE_RUN", "ensemble")

JUDGE_KEYS = ["comedian", "pun_expert", "editor", "translator"]
BORDA_METHODS = ["judges_then_models", "models_then_judges", "pooled_rankings"]

BORDA_ALIASES = {
    "judges_then_models": "judges_then_models",
    "judge_first": "judges_then_models",
    "judges_first": "judges_then_models",
    "models_then_judges": "models_then_judges",
    "model_first": "models_then_judges",
    "models_first": "models_then_judges",
    "pooled_rankings": "pooled_rankings",
    "pooled": "pooled_rankings",
    "pool": "pooled_rankings",
}


def norm_space(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def ensure_slash(path: str) -> str:
    return str(path or "").rstrip("/") + "/"


def safe_json_loads(x: Any, default: Any = None) -> Any:
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


def chunk_numbers_in_dir(path: str) -> list[int]:
    chunks: list[int] = []
    for file_path in glob.glob(ensure_slash(path) + "*.tsv"):
        stem = Path(file_path).stem
        if stem.isdigit():
            chunks.append(int(stem))
    return sorted(chunks)


def discover_raw_judges() -> list[str]:
    base = ensure_slash(OUTPUT_ROOT) + ensure_slash(ENSEMBLE_RUN)
    out = []
    for path in glob.glob(base + "*/"):
        name = Path(path.rstrip("/")).name
        if name in {"borda", "metrics", "reports", "analysis"}:
            continue
        if chunk_numbers_in_dir(path):
            out.append(name)
    return sorted(out)


def discover_borda_dirs() -> list[dict[str, str]]:
    base = ensure_slash(OUTPUT_ROOT) + ensure_slash(ENSEMBLE_RUN) + "borda/"
    rows: list[dict[str, str]] = []

    for method_dir in sorted(glob.glob(base + "*/")):
        method = Path(method_dir.rstrip("/")).name
        method = BORDA_ALIASES.get(method, method)
        if method not in BORDA_METHODS:
            continue

        for internal_dir in sorted(glob.glob(ensure_slash(method_dir) + "*/")):
            internal_weights = Path(internal_dir.rstrip("/")).name

            for model_dir in sorted(glob.glob(ensure_slash(internal_dir) + "*/")):
                model_weights = Path(model_dir.rstrip("/")).name
                if chunk_numbers_in_dir(model_dir):
                    rows.append({
                        "method": method,
                        "internal_weights": internal_weights,
                        "model_weights": model_weights,
                        "path": ensure_slash(model_dir),
                    })

    return rows


def load_chunks(path: str) -> pd.DataFrame:
    chunks = chunk_numbers_in_dir(path)
    if not chunks:
        raise FileNotFoundError(f"No TSV chunks found under {path}")

    frames = []
    for chunk in chunks:
        file_path = ensure_slash(path) + f"{chunk}.tsv"
        df = load(file_path).fillna("")
        df["chunk"] = chunk
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        raw = safe_json_loads(row.get("shuffled_candidates_json"), [])
        if not isinstance(raw, list):
            continue

        for pos, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            if "id" not in item:
                continue

            rows.append({
                "id_en": norm_space(row.get("id_en")),
                "chunk": row.get("chunk"),
                "candidate_id": int(item.get("id")),
                "candidate_pos": pos,
                "candidate_source": norm_space(item.get("source")),
                "candidate_original_id": norm_space(item.get("original_id")),
                "candidate_pun": norm_space(item.get("pun")),
                "pun_len_chars": len(norm_space(item.get("pun"))),
                "pun_len_words": len(norm_space(item.get("pun")).split()),
            })

    return pd.DataFrame(rows)


def ranking_rows(df: pd.DataFrame, json_col: str = "discriminator_run2_json") -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        raw = safe_json_loads(row.get(json_col), {})
        if not isinstance(raw, dict):
            continue

        for judge in JUDGE_KEYS:
            ids = raw.get(judge, [])
            if not isinstance(ids, list):
                continue

            for rank, cid in enumerate(ids, start=1):
                try:
                    cid = int(cid)
                except Exception:
                    continue

                rows.append({
                    "id_en": norm_space(row.get("id_en")),
                    "chunk": row.get("chunk"),
                    "judge": judge,
                    "candidate_id": cid,
                    "rank": rank,
                    "points": 5 - rank,
                    "is_top1": rank == 1,
                })

    return pd.DataFrame(rows)


def borda_ranking_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        raw = safe_json_loads(row.get("discriminator_run2_borda_ranking_json"), [])
        if not isinstance(raw, list):
            continue

        for rank, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            try:
                cid = int(item.get("id"))
            except Exception:
                continue

            rows.append({
                "id_en": norm_space(row.get("id_en")),
                "chunk": row.get("chunk"),
                "rank": rank,
                "candidate_id": cid,
                "source": norm_space(item.get("source")),
                "original_id": norm_space(item.get("original_id")),
                "pun": norm_space(item.get("pun")),
                "score": float(item.get("score") or 0.0),
                "pun_len_chars": len(norm_space(item.get("pun"))),
                "pun_len_words": len(norm_space(item.get("pun")).split()),
            })

    return pd.DataFrame(rows)


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


def raw_metrics(judge_model: str, path: str) -> None:
    df = load_chunks(path)
    candidates = candidate_rows(df)
    rankings = ranking_rows(df)

    section(f"RAW RUN 2 METRICS: {judge_model}")

    print_kv([
        ("judge_model", judge_model),
        ("input_dir", path),
        ("rows", len(df)),
        ("chunks", df["chunk"].nunique() if "chunk" in df.columns else ""),
        ("unique_id_en", df["id_en"].nunique() if "id_en" in df.columns else ""),
        ("candidate_rows", len(candidates)),
        ("ranking_rows", len(rankings)),
    ])

    subsection("Errors")
    if "discriminator_run2_error" in df.columns:
        s = df["discriminator_run2_error"].fillna("").astype(str).str.strip()
        print_kv([
            ("error_rows", int(s.ne("").sum())),
            ("error_pct", pct(int(s.ne("").sum()), len(df))),
        ])
        if s.ne("").any():
            print_df(count_table(s[s.ne("")], "error").head(20))
    else:
        print("(no discriminator_run2_error column)")

    subsection("Candidate sources")
    if not candidates.empty:
        print_df(count_table(candidates["candidate_source"], "candidate_source"))

    subsection("Candidate prompt positions")
    if not candidates.empty:
        print_df(count_table(candidates["candidate_pos"], "candidate_pos"))

    subsection("Candidate length")
    if not candidates.empty:
        print_df(describe_series(candidates["pun_len_chars"], "candidate_pun_len_chars"))
        print_df(describe_series(candidates["pun_len_words"], "candidate_pun_len_words"))

    subsection("Ranking slot counts")
    if not rankings.empty:
        print_df(rankings.groupby(["judge", "rank"]).size().reset_index(name="count"))

    top1 = pd.DataFrame()
    if not rankings.empty and not candidates.empty:
        candidate_meta = candidates.drop_duplicates(["id_en", "candidate_id"])
        top1 = rankings[rankings["rank"] == 1].merge(
            candidate_meta[
                [
                    "id_en",
                    "candidate_id",
                    "candidate_pos",
                    "candidate_source",
                    "candidate_pun",
                    "pun_len_chars",
                    "pun_len_words",
                ]
            ],
            on=["id_en", "candidate_id"],
            how="left",
        )

    subsection("Top-1 by persona")
    if not top1.empty:
        print_df(top1.groupby("judge").agg(
            examples=("id_en", "count"),
            unique_top1_candidates=("candidate_id", "nunique"),
            mean_candidate_pos=("candidate_pos", "mean"),
            median_candidate_pos=("candidate_pos", "median"),
            mean_len_chars=("pun_len_chars", "mean"),
            mean_len_words=("pun_len_words", "mean"),
        ).reset_index())

    subsection("Top-1 source by persona")
    if not top1.empty:
        print_df(top1.groupby(["judge", "candidate_source"]).size().reset_index(name="count"))

    subsection("Persona top-1 agreement")
    if not top1.empty:
        top1_wide = top1.pivot_table(
            index="id_en",
            columns="judge",
            values="candidate_id",
            aggfunc="first",
        ).reset_index()

        rows = []
        for j1, j2 in combinations(JUDGE_KEYS, 2):
            if j1 not in top1_wide.columns or j2 not in top1_wide.columns:
                continue
            valid = top1_wide[[j1, j2]].dropna()
            n = len(valid)
            same = int((valid[j1] == valid[j2]).sum()) if n else 0
            rows.append({
                "judge_1": j1,
                "judge_2": j2,
                "rows": n,
                "same_top1": same,
                "top1_agreement_pct": pct(same, n),
            })
        print_df(pd.DataFrame(rows))

        distinct = top1.groupby("id_en")["candidate_id"].nunique().reset_index(name="distinct_persona_top1_candidates")
        subsection("Distinct persona top-1 winners per example")
        print_df(count_table(distinct["distinct_persona_top1_candidates"], "distinct_persona_top1_candidates"))

    subsection("Persona top-4 overlap/order agreement")
    if not rankings.empty:
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
                    "intersection": inter,
                    "union": union,
                    "jaccard": inter / union if union else 0.0,
                })

                order = pairwise_order_agreement(by_judge_lists[j1], by_judge_lists[j2])
                if order is not None:
                    order_rows.append({
                        "id_en": id_en,
                        "judge_1": j1,
                        "judge_2": j2,
                        "common_candidates": inter,
                        "pairwise_order_agreement": order,
                    })

        overlap = pd.DataFrame(overlap_rows)
        order = pd.DataFrame(order_rows)

        if not overlap.empty:
            print_df(overlap.groupby(["judge_1", "judge_2"]).agg(
                rows=("id_en", "count"),
                mean_intersection=("intersection", "mean"),
                mean_jaccard=("jaccard", "mean"),
            ).reset_index())

        if not order.empty:
            print_df(order.groupby(["judge_1", "judge_2"]).agg(
                rows=("id_en", "count"),
                mean_common_candidates=("common_candidates", "mean"),
                mean_pairwise_order_agreement=("pairwise_order_agreement", "mean"),
                median_pairwise_order_agreement=("pairwise_order_agreement", "median"),
            ).reset_index())


def borda_metrics(info: dict[str, str]) -> pd.DataFrame:
    df = load_chunks(info["path"])
    rankings = borda_ranking_rows(df)
    winners = rankings[rankings["rank"] == 1].copy() if not rankings.empty else pd.DataFrame()

    title = (
        f"BORDA METRICS: method={info['method']} "
        f"internal={info['internal_weights']} model={info['model_weights']}"
    )
    section(title)

    print_kv([
        ("method", info["method"]),
        ("internal_weights", info["internal_weights"]),
        ("model_weights", info["model_weights"]),
        ("input_dir", info["path"]),
        ("rows", len(df)),
        ("chunks", df["chunk"].nunique() if "chunk" in df.columns else ""),
        ("unique_id_en", df["id_en"].nunique() if "id_en" in df.columns else ""),
        ("ranking_rows", len(rankings)),
        ("winner_rows", len(winners)),
    ])

    subsection("Borda errors")
    if "discriminator_run2_borda_error" in df.columns:
        s = df["discriminator_run2_borda_error"].fillna("").astype(str).str.strip()
        print_kv([
            ("borda_error_rows", int(s.ne("").sum())),
            ("borda_error_pct", pct(int(s.ne("").sum()), len(df))),
        ])
        if s.ne("").any():
            print_df(count_table(s[s.ne("")], "borda_error").head(20))
    else:
        print("(no discriminator_run2_borda_error column)")

    if winners.empty:
        print("No winners found.")
        return pd.DataFrame()

    subsection("Winner source distribution")
    print_df(count_table(winners["source"], "winner_source"))

    subsection("Winner score")
    print_df(describe_series(winners["score"], "winner_score"))

    subsection("Winner length")
    print_df(describe_series(winners["pun_len_chars"], "winner_pun_len_chars"))
    print_df(describe_series(winners["pun_len_words"], "winner_pun_len_words"))

    subsection("Full ranking score by rank")
    print_df(rankings.groupby("rank").agg(
        count=("id_en", "count"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index())

    piv = rankings.pivot_table(index="id_en", columns="rank", values="score", aggfunc="first")
    if 1 in piv.columns and 2 in piv.columns:
        margin = (piv[1] - piv[2]).dropna()
        subsection("Winner margin over second place")
        print_df(describe_series(margin, "winner_margin_rank1_minus_rank2"))

    out = winners[["id_en", "candidate_id", "source", "pun", "score"]].copy()
    out["method"] = info["method"]
    out["internal_weights"] = info["internal_weights"]
    out["model_weights"] = info["model_weights"]
    return out


def compare_borda(all_winners: pd.DataFrame) -> None:
    if all_winners.empty:
        return

    section("COMPARE BORDA METHODS")

    key_cols = ["method", "internal_weights", "model_weights"]

    print_kv([
        ("winner_rows", len(all_winners)),
        ("unique_configs", all_winners[key_cols].drop_duplicates().shape[0]),
        ("unique_id_en", all_winners["id_en"].nunique()),
    ])

    subsection("Config coverage")
    print_df(all_winners.groupby(key_cols).agg(
        rows=("id_en", "count"),
        unique_ids=("id_en", "nunique"),
        unique_winners=("candidate_id", "nunique"),
        mean_score=("score", "mean"),
    ).reset_index())

    configs = []
    for key, g in all_winners.groupby(key_cols):
        label = "|".join(str(x) for x in key)
        configs.append((label, g.set_index("id_en")["candidate_id"].to_dict()))

    rows = []
    for (label_a, map_a), (label_b, map_b) in combinations(configs, 2):
        ids = sorted(set(map_a) & set(map_b))
        same = sum(1 for id_en in ids if map_a[id_en] == map_b[id_en])
        rows.append({
            "config_1": label_a,
            "config_2": label_b,
            "shared_ids": len(ids),
            "same_winner": same,
            "same_winner_pct": pct(same, len(ids)),
        })

    subsection("Pairwise winner agreement")
    print_df(pd.DataFrame(rows))

    subsection("Winner source by config")
    print_df(all_winners.groupby(key_cols + ["source"]).size().reset_index(name="count"))


def main() -> None:
    section("RUN 2 ENSEMBLE METRICS AUTO-DISCOVERY")
    print_kv([
        ("OUTPUT_ROOT", OUTPUT_ROOT),
        ("ENSEMBLE_RUN", ENSEMBLE_RUN),
    ])

    raw_judges = discover_raw_judges()
    borda_dirs = discover_borda_dirs()

    print_kv([
        ("raw_judge_dirs_found", ", ".join(raw_judges) if raw_judges else "(none)"),
        ("borda_dirs_found", len(borda_dirs)),
    ])

    for judge in raw_judges:
        path = ensure_slash(OUTPUT_ROOT) + ensure_slash(ENSEMBLE_RUN) + ensure_slash(judge)
        raw_metrics(judge, path)

    all_winners = []
    for info in borda_dirs:
        winners = borda_metrics(info)
        if not winners.empty:
            all_winners.append(winners)

    if all_winners:
        compare_borda(pd.concat(all_winners, ignore_index=True))
    else:
        section("COMPARE BORDA METHODS")
        print("No Borda winner outputs found.")


if __name__ == "__main__":
    main()