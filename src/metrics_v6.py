"""
JOKER 2026 Run 2 metrics v6: pooled-only batch runner.

Runs the selected Run 2 ensemble discriminator metric configurations and prints only
"Pooled Borda over all judge-model x internal-judge rankings" counts.

Default behavior matches the first v5 self-preference block:
  self_policy = no_self_correction, self_factor = 1.0

Expected input path:
  ../data/processed/discriminate/run2/{ensemble_run}/{judge_model}/{chunk}.tsv

Typical usage from src:
  python metrics_v6.py
  python metrics_v6.py run2 ensemble 0 -1
  python metrics_v6.py run2 ensemble claude,gemini_pro,gpt 0 -1

Optional:
  python metrics_v6.py --all-policies
  python metrics_v6.py run2 ensemble 0 -1 --all-policies

Output columns are ordered as requested:
  gpt, gemini_pro, claude, gemini_flash

Notes:
  - Internal judge weights are ordered: comedian_pun_expert_editor_translator.
  - Judge model weights are ordered by the discovered/provided judge model list.
    With the usual discovered order, this is: claude, gemini_pro, gpt.
  - The model-weight strings below are therefore interpreted the same way as v5.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from config import MODEL_ALIASES
except Exception:
    MODEL_ALIASES = {}

try:
    from data import load
except Exception:

    def load(path: str) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t")


OUTPUT_ROOT = os.environ.get(
    "DISCRIMINATOR_RUN2_OUTPUT_DIR",
    "../data/processed/discriminate/run2/",
)

JUDGE_KEYS = ["comedian", "pun_expert", "editor", "translator"]
SOURCE_ORDER = ["claude", "gemini", "gemini_pro_single", "gpt_single"]

SELF_POLICIES = [
    ("no_self_correction", 1.0),
    ("half_self_weight", 0.5),
    ("remove_self_votes", 0.0),
]

DEFAULT_RUNS = [
    ("25_25_25_25", "1_1_1"),
    ("45_30_25_10", "1_1_1"),
    ("100_0_0_0", "1_1_1"),
    ("0_100_0_0", "1_1_1"),
    ("0_0_100_0", "1_1_1"),
    ("0_0_0_100", "1_1_1"),
    ("25_25_25_25", "1_0_0"),
    ("45_30_25_10", "1_0_0"),
    ("100_0_0_0", "1_0_0"),
    ("0_100_0_0", "1_0_0"),
    ("0_0_100_0", "1_0_0"),
    ("0_0_0_100", "1_0_0"),
    ("25_25_25_25", "0_1_0"),
    ("45_30_25_10", "0_1_0"),
    ("100_0_0_0", "0_1_0"),
    ("0_100_0_0", "0_1_0"),
    ("0_0_100_0", "0_1_0"),
    ("0_0_0_100", "0_1_0"),
    ("25_25_25_25", "0_0_1"),
    ("45_30_25_10", "0_0_1"),
    ("100_0_0_0", "0_0_1"),
    ("0_100_0_0", "0_0_1"),
    ("0_0_100_0", "0_0_1"),
    ("0_0_0_100", "0_0_1"),
]


@dataclass
class RowEval:
    judge_model: str
    chunk: int
    id_en: str
    source_by_id: dict[int, str]
    rankings: dict[str, list[int]]
    positions_by_source: dict[str, int]


def norm_space(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def ensure_slash(path: str) -> str:
    return str(path or "").rstrip("/") + "/"


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
    return json.loads(text)


def resolve_model_alias(model_arg: str) -> str:
    model_arg = norm_space(model_arg)
    if model_arg in MODEL_ALIASES and MODEL_ALIASES.get(model_arg):
        return model_arg
    filesystem_alias = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_arg).strip("_")
    return filesystem_alias or "model"


def discover_judges(ensemble_run: str) -> list[str]:
    base = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}"
    dirs = [p for p in glob.glob(base + "*/") if os.path.isdir(p)]
    judges: list[str] = []
    for path in dirs:
        name = Path(path.rstrip("/")).name
        if name in {"borda", "metrics", "reports", "analysis"}:
            continue
        has_numeric_chunk = any(
            Path(tsv).stem.isdigit()
            for tsv in glob.glob(ensure_slash(path) + "*.tsv")
        )
        if has_numeric_chunk:
            judges.append(name)
    return sorted(judges)


def chunk_numbers_for_judge(ensemble_run: str, judge: str) -> list[int]:
    input_dir = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}{ensure_slash(judge)}"
    chunks: list[int] = []
    for path in glob.glob(input_dir + "*.tsv"):
        stem = Path(path).stem
        if stem.isdigit():
            chunks.append(int(stem))
    return sorted(chunks)


def selected_chunks(ensemble_run: str, judges: list[str], start: int, end: int) -> list[int]:
    chunk_sets = []
    for judge in judges:
        chunks = set(chunk_numbers_for_judge(ensemble_run, judge))
        if chunks:
            chunk_sets.append(chunks)
    if not chunk_sets:
        return []
    available = sorted(set.union(*chunk_sets))
    return [c for c in available if c >= start and (end == -1 or c < end)]


def parse_weight_string(
    weight_string: str,
    names: list[str],
    label: str,
    allow_extra: bool = False,
) -> dict[str, float]:
    parts = [p for p in str(weight_string).split("_") if p != ""]
    try:
        values = [float(p.replace("p", ".")) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"Invalid {label} weight string {weight_string!r}; "
            "expected numbers joined by underscores"
        ) from exc

    if len(values) < len(names):
        raise ValueError(
            f"{label} weight string {weight_string!r} has {len(values)} values "
            f"but needs at least {len(names)} for: {', '.join(names)}"
        )
    if len(values) > len(names) and not allow_extra:
        raise ValueError(
            f"{label} weight string {weight_string!r} has {len(values)} values "
            f"but needs exactly {len(names)} for: {', '.join(names)}"
        )
    if len(values) > len(names) and allow_extra:
        values = values[: len(names)]

    return dict(zip(names, values))


def candidate_maps_from_row(row: pd.Series) -> tuple[dict[int, str], dict[str, int]]:
    raw = safe_json_loads(row.get("shuffled_candidates_json", ""))
    if not isinstance(raw, list):
        raise ValueError("shuffled_candidates_json must be a JSON array")

    source_by_id: dict[int, str] = {}
    positions_by_source: dict[str, int] = {}
    for pos, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        if "id" not in item or "source" not in item:
            continue
        cid = int(item["id"])
        source = str(item["source"])
        source_by_id[cid] = source
        positions_by_source[source] = pos
    if len(source_by_id) != 4:
        raise ValueError(f"Expected 4 candidate ids/sources, got {len(source_by_id)}")
    return source_by_id, positions_by_source


def rankings_from_row(row: pd.Series) -> dict[str, list[int]]:
    raw = safe_json_loads(row.get("discriminator_run2_json", ""))
    if not isinstance(raw, dict):
        raise ValueError("discriminator_run2_json must be a JSON object")

    out: dict[str, list[int]] = {}
    for key in JUDGE_KEYS:
        vals = raw.get(key)
        if not isinstance(vals, list) or len(vals) != 4:
            raise ValueError(f"Missing or invalid ranking for {key}: {vals}")
        ids = [int(v) for v in vals]
        if len(set(ids)) != 4:
            raise ValueError(f"Ranking for {key} contains duplicates: {ids}")
        out[key] = ids
    return out


def same_model_family(judge_model: str, candidate_source: str) -> bool:
    j = judge_model.lower()
    s = candidate_source.lower()
    if j.startswith("claude") and s == "claude":
        return True
    if j.startswith("gpt") and s == "gpt_single":
        return True
    if j.startswith("gemini") and s in {"gemini", "gemini_pro_single"}:
        return True
    return False


def borda_rank(
    rankings: list[list[int]],
    weights: list[float] | None = None,
    *,
    source_by_id: dict[int, str] | None = None,
    ranking_judge_models: list[str] | None = None,
    self_factor: float = 1.0,
) -> tuple[dict[int, float], list[int]]:
    if not rankings:
        return {}, []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")
    if ranking_judge_models is not None and len(ranking_judge_models) != len(rankings):
        raise ValueError("ranking_judge_models length must match rankings length")

    scores: dict[int, float] = defaultdict(float)
    for i, (ranking, weight) in enumerate(zip(rankings, weights)):
        n = len(ranking)
        judge_model = ranking_judge_models[i] if ranking_judge_models else ""
        for pos, cid in enumerate(ranking):
            cid = int(cid)
            factor = 1.0
            if source_by_id is not None and ranking_judge_models is not None:
                if same_model_family(judge_model, source_by_id[cid]):
                    factor = self_factor
            scores[cid] += float(weight) * factor * (n - pos)
    ranked_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return dict(scores), ranked_ids


def weighted_internal_borda_for_eval(
    row_eval: RowEval,
    internal_weights: dict[str, float],
    *,
    self_factor: float = 1.0,
) -> tuple[dict[int, float], list[int]]:
    rankings = [row_eval.rankings[judge] for judge in JUDGE_KEYS]
    weights = [internal_weights[judge] for judge in JUDGE_KEYS]
    judge_models = [row_eval.judge_model for _ in JUDGE_KEYS]
    return borda_rank(
        rankings,
        weights,
        source_by_id=row_eval.source_by_id,
        ranking_judge_models=judge_models,
        self_factor=self_factor,
    )


def winner_source(ranked_ids: list[int], source_by_id: dict[int, str]) -> str:
    if not ranked_ids:
        return ""
    return source_by_id[int(ranked_ids[0])]


def load_records(
    ensemble_run: str,
    judges: list[str],
    chunks: list[int],
) -> tuple[dict[tuple[int, str], dict[str, RowEval]], Counter[str], Counter[str]]:
    records: dict[tuple[int, str], dict[str, RowEval]] = defaultdict(dict)
    errors_by_judge_model: Counter[str] = Counter()
    rows_by_judge_model: Counter[str] = Counter()

    for judge_model in judges:
        for chunk_num in chunks:
            path = f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}{ensure_slash(judge_model)}{chunk_num}.tsv"
            if not os.path.exists(path):
                continue
            df = load(path)
            rows_by_judge_model[judge_model] += len(df)
            for _, row in df.iterrows():
                try:
                    if norm_space(row.get("discriminator_run2_error", "")):
                        errors_by_judge_model[judge_model] += 1
                        continue
                    source_by_id, positions_by_source = candidate_maps_from_row(row)
                    rankings = rankings_from_row(row)
                    valid_ids = set(source_by_id)
                    for judge_key, ranking in rankings.items():
                        invalid = [cid for cid in ranking if cid not in valid_ids]
                        if invalid:
                            raise ValueError(f"Invalid ids in {judge_key}: {invalid}")
                    id_en = norm_space(row.get("id_en", ""))
                    records[(chunk_num, id_en)][judge_model] = RowEval(
                        judge_model=judge_model,
                        chunk=chunk_num,
                        id_en=id_en,
                        source_by_id=source_by_id,
                        rankings=rankings,
                        positions_by_source=positions_by_source,
                    )
                except Exception:
                    errors_by_judge_model[judge_model] += 1
                    continue

    return records, rows_by_judge_model, errors_by_judge_model


def compute_pooled_rankings(
    records: dict[tuple[int, str], dict[str, RowEval]],
    judges: list[str],
    internal_weights: dict[str, float],
    model_weights: dict[str, float],
    self_factor: float,
) -> Counter[str]:
    pooled_rankings: Counter[str] = Counter()

    for _, by_model in sorted(records.items()):
        present_models = [j for j in judges if j in by_model]
        if not present_models:
            continue

        source_by_id = by_model[present_models[0]].source_by_id

        all_rankings: list[list[int]] = []
        all_weights: list[float] = []
        all_judge_models: list[str] = []
        for judge_model in present_models:
            for internal_judge in JUDGE_KEYS:
                all_rankings.append(by_model[judge_model].rankings[internal_judge])
                all_weights.append(
                    model_weights[judge_model] * internal_weights[internal_judge]
                )
                all_judge_models.append(judge_model)

        _, pooled_ranked_ids = borda_rank(
            all_rankings,
            all_weights,
            source_by_id=source_by_id,
            ranking_judge_models=all_judge_models,
            self_factor=self_factor,
        )
        pooled_rankings[winner_source(pooled_ranked_ids, source_by_id)] += 1

    return pooled_rankings


def output_row(
    internal_weight_string: str,
    model_weight_string: str,
    policy_name: str,
    pooled_rankings: Counter[str],
    include_policy: bool,
) -> None:
    cols = [
        internal_weight_string,
        model_weight_string,
    ]
    if include_policy:
        cols.append(policy_name)
    cols.extend(
        [
            str(pooled_rankings.get("gpt_single", 0)),
            str(pooled_rankings.get("gemini_pro_single", 0)),
            str(pooled_rankings.get("claude", 0)),
            str(pooled_rankings.get("gemini", 0)),
        ]
    )
    print("\t".join(cols))


def run_batch(
    ensemble_run: str,
    judges: list[str],
    start: int,
    end: int,
    all_policies: bool = False,
) -> None:
    chunks = selected_chunks(ensemble_run, judges, start, end)
    if not chunks:
        raise FileNotFoundError(
            f"No raw Run 2 chunks found for ensemble_run={ensemble_run}, "
            f"judges={judges}, start={start}, end={end}"
        )

    records, _, _ = load_records(ensemble_run, judges, chunks)

    policies = SELF_POLICIES if all_policies else [("no_self_correction", 1.0)]

    header = ["persona_weights", "discriminator_weights"]
    if all_policies:
        header.append("self_policy")
    header.extend(["gpt", "gemini_pro", "claude", "gemini_flash"])
    print("\t".join(header))

    for internal_weight_string, model_weight_string in DEFAULT_RUNS:
        internal_weights = parse_weight_string(
            internal_weight_string,
            JUDGE_KEYS,
            "internal judge",
            allow_extra=False,
        )
        model_weights = parse_weight_string(
            model_weight_string,
            judges,
            "judge model",
            allow_extra=True,
        )

        for policy_name, self_factor in policies:
            pooled_rankings = compute_pooled_rankings(
                records,
                judges,
                internal_weights,
                model_weights,
                self_factor,
            )
            output_row(
                internal_weight_string,
                model_weight_string,
                policy_name,
                pooled_rankings,
                include_policy=all_policies,
            )


def usage() -> str:
    return """Usage:
  python metrics_v6.py
  python metrics_v6.py run2 <ensemble_run> <start> <end>
  python metrics_v6.py run2 <ensemble_run> <judge_models_csv> <start> <end>

Options:
  --all-policies    Print pooled rows for all three v5 self-preference policies.
                    Default prints only no_self_correction.

Examples:
  python metrics_v6.py
  python metrics_v6.py run2 ensemble 0 -1
  python metrics_v6.py run2 ensemble claude,gemini_pro,gpt 0 -1
  python metrics_v6.py run2 ensemble 0 -1 --all-policies
"""


def main() -> None:
    args = sys.argv[1:]
    all_policies = False
    if "--all-policies" in args:
        all_policies = True
        args = [arg for arg in args if arg != "--all-policies"]

    if not args:
        ensemble_run = "ensemble"
        judges = discover_judges(ensemble_run)
        start = 0
        end = -1
    else:
        task = args[0]
        if task != "run2":
            raise ValueError(f"Unknown task: {task}\n{usage()}")

        ensemble_run = args[1] if len(args) > 1 else "ensemble"

        if len(args) == 5 and "," in args[2]:
            judges = [
                resolve_model_alias(x.strip())
                for x in args[2].split(",")
                if x.strip()
            ]
            start = int(args[3])
            end = int(args[4])
        elif len(args) == 4:
            judges = discover_judges(ensemble_run)
            start = int(args[2])
            end = int(args[3])
        else:
            raise ValueError(usage())

    if not judges:
        raise FileNotFoundError(
            f"No judge model directories found under "
            f"{ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}"
        )

    run_batch(
        ensemble_run=ensemble_run,
        judges=judges,
        start=start,
        end=end,
        all_policies=all_policies,
    )


if __name__ == "__main__":
    main()
