"""
JOKER 2026 Run 2 metrics v5: ensemble discriminator diagnostics with adjustable weights
and self-preference sensitivity analysis.

Reads raw discriminator Run 2 TSV files and prints aggregate counts to console.
Does not save files and does not modify existing outputs.

Expected input path:
  ../data/processed/discriminate/run2/{ensemble_run}/{judge_model}/{chunk}.tsv

Usage from src:
  python metrics_v5.py run2 ensemble 45_30_15_10 33_33_34 0 -1
  python metrics_v5.py run2 ensemble 25_25_25_25 25_25_25_25 0 1
  python metrics_v5.py run2 ensemble claude,gpt,gemini_pro 45_30_15_10 33_33_34 0 -1

Weight strings:
  - internal judge weights are ordered: comedian_pun_expert_editor_translator
  - judge model weights are ordered by the discovered/provided judge model list printed at runtime
    (for example: claude, gemini_pro, gpt). If extra model weights are provided, extras are ignored.

Self-preference sensitivity:
  v5 prints the same cross-model Borda aggregations under three self-vote policies:
    - no_self_correction: self votes have normal weight
    - half_self_weight: when judge model and candidate source share a family, that candidate's
      Borda contribution from that judge model is multiplied by 0.5
    - remove_self_votes: when judge model and candidate source share a family, that candidate's
      Borda contribution from that judge model is multiplied by 0

What it prints:
  - Top-1 counts by judge model and generator/source
  - Top-1 counts by internal judge persona and generator/source
  - Top-1 counts by judge model x internal judge
  - Weighted internal-judge Borda winner counts by judge model
  - Judges-first-then-models Borda aggregation
  - Models-first-then-judges Borda aggregation
  - Pooled Borda over all rankings from all judge models and internal judges
  - The above aggregation methods under three self-preference correction policies
  - Agreement between judge models for per-model Borda winners
  - Self-preference lift diagnostics
  - Prompt-position exposure counts
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
from typing import Any, Iterable

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
        has_numeric_chunk = any(Path(tsv).stem.isdigit() for tsv in glob.glob(ensure_slash(path) + "*.tsv"))
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


def parse_weight_string(weight_string: str, names: list[str], label: str, allow_extra: bool = False) -> dict[str, float]:
    parts = [p for p in str(weight_string).split("_") if p != ""]
    try:
        values = [float(p.replace("p", ".")) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid {label} weight string {weight_string!r}; expected numbers joined by underscores") from exc

    if len(values) < len(names):
        raise ValueError(
            f"{label} weight string {weight_string!r} has {len(values)} values but needs at least {len(names)} "
            f"for: {', '.join(names)}"
        )
    if len(values) > len(names) and not allow_extra:
        raise ValueError(
            f"{label} weight string {weight_string!r} has {len(values)} values but needs exactly {len(names)} "
            f"for: {', '.join(names)}"
        )
    if len(values) > len(names) and allow_extra:
        print(
            f"Note: {label} weight string has {len(values)} values but only {len(names)} models are active; "
            f"using the first {len(names)} values and ignoring extras."
        )
        values = values[: len(names)]

    return dict(zip(names, values))


def format_weight_key(weights: dict[str, float], names: list[str]) -> str:
    out = []
    for name in names:
        value = float(weights[name])
        out.append(str(int(value)) if value.is_integer() else str(value).replace(".", "p"))
    return "_".join(out)


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


def source_columns_from_counters(counters: Iterable[Counter[str]]) -> list[str]:
    keys: set[str] = set()
    for counter in counters:
        keys.update(counter.keys())
    extras = sorted([k for k in keys if k not in SOURCE_ORDER])
    return SOURCE_ORDER + extras


def print_table(title: str, rows: list[tuple[str, Counter[str]]]) -> None:
    print("\n" + title)
    print("=" * len(title))
    if not rows:
        print("  No data")
        return

    cols = source_columns_from_counters(counter for _, counter in rows)
    header = ["group"] + cols + ["total"]
    table: list[list[Any]] = []
    for label, counter in rows:
        table.append([label] + [counter.get(col, 0) for col in cols] + [sum(counter.values())])

    widths = [len(h) for h in header]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row: list[Any]) -> str:
        return "  ".join(str(cell).rjust(widths[i]) if i else str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(header))
    print(fmt(["-" * w for w in widths]))
    for row in table:
        print(fmt(row))


def print_simple_table(title: str, header: list[str], rows: list[list[Any]]) -> None:
    print("\n" + title)
    print("=" * len(title))
    if not rows:
        print("  No data")
        return
    widths = [len(str(h)) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row: list[Any]) -> str:
        return "  ".join(str(cell).rjust(widths[i]) if i else str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(header))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt(row))


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


def load_records(ensemble_run: str, judges: list[str], chunks: list[int]) -> tuple[dict[tuple[int, str], dict[str, RowEval]], Counter[str], Counter[str]]:
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
            print(f"Loaded {path}: {len(df)} rows")
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


def compute_cross_aggregates(
    records: dict[tuple[int, str], dict[str, RowEval]],
    judges: list[str],
    internal_weights: dict[str, float],
    model_weights: dict[str, float],
    self_factor: float,
) -> tuple[Counter[str], Counter[str], Counter[str], dict[str, Counter[str]], Counter[str], Counter[str], int]:
    per_model_borda_winners: dict[str, Counter[str]] = defaultdict(Counter)
    judges_first_models_second: Counter[str] = Counter()
    models_first_judges_second: Counter[str] = Counter()
    pooled_rankings: Counter[str] = Counter()
    judge_model_agreement: Counter[str] = Counter()
    pair_agreement: Counter[str] = Counter()
    complete_rows = 0

    for _, by_model in sorted(records.items()):
        present_models = [j for j in judges if j in by_model]
        if not present_models:
            continue
        if len(present_models) == len(judges):
            complete_rows += 1

        source_by_id = by_model[present_models[0]].source_by_id

        # Per-model internal Borda winners.
        per_model_rankings: list[list[int]] = []
        per_model_weights: list[float] = []
        per_model_names: list[str] = []
        per_model_winner_sources: dict[str, str] = {}
        for judge_model in present_models:
            _, ranked_ids = weighted_internal_borda_for_eval(
                by_model[judge_model],
                internal_weights,
                self_factor=self_factor,
            )
            src = winner_source(ranked_ids, source_by_id)
            per_model_borda_winners[judge_model][src] += 1
            per_model_rankings.append(ranked_ids)
            per_model_weights.append(model_weights[judge_model])
            per_model_names.append(judge_model)
            per_model_winner_sources[judge_model] = src

        # 1) Judges first, then models.
        _, ranked_after_models = borda_rank(
            per_model_rankings,
            per_model_weights,
            source_by_id=source_by_id,
            ranking_judge_models=per_model_names,
            self_factor=self_factor,
        )
        judges_first_models_second[winner_source(ranked_after_models, source_by_id)] += 1

        # Agreement among per-model Borda winners.
        winner_values = list(per_model_winner_sources.values())
        judge_model_agreement[str(len(set(winner_values)))] += 1
        for i, a in enumerate(present_models):
            for b in present_models[i + 1:]:
                key = f"{a}={b}"
                if per_model_winner_sources[a] == per_model_winner_sources[b]:
                    pair_agreement[key] += 1

        # 2) Models first, then judges.
        per_internal_aggregate_rankings: list[list[int]] = []
        for internal_judge in JUDGE_KEYS:
            rankings_for_persona = [by_model[j].rankings[internal_judge] for j in present_models]
            weights_for_persona = [model_weights[j] for j in present_models]
            model_names_for_persona = present_models[:]
            _, ranked_ids = borda_rank(
                rankings_for_persona,
                weights_for_persona,
                source_by_id=source_by_id,
                ranking_judge_models=model_names_for_persona,
                self_factor=self_factor,
            )
            per_internal_aggregate_rankings.append(ranked_ids)
        _, ranked_after_judges = borda_rank(
            per_internal_aggregate_rankings,
            [internal_weights[j] for j in JUDGE_KEYS],
        )
        models_first_judges_second[winner_source(ranked_after_judges, source_by_id)] += 1

        # 3) Entire pool of rankings.
        all_rankings: list[list[int]] = []
        all_weights: list[float] = []
        all_judge_models: list[str] = []
        for judge_model in present_models:
            for internal_judge in JUDGE_KEYS:
                all_rankings.append(by_model[judge_model].rankings[internal_judge])
                all_weights.append(model_weights[judge_model] * internal_weights[internal_judge])
                all_judge_models.append(judge_model)
        _, pooled_ranked_ids = borda_rank(
            all_rankings,
            all_weights,
            source_by_id=source_by_id,
            ranking_judge_models=all_judge_models,
            self_factor=self_factor,
        )
        pooled_rankings[winner_source(pooled_ranked_ids, source_by_id)] += 1

    return (
        judges_first_models_second,
        models_first_judges_second,
        pooled_rankings,
        per_model_borda_winners,
        judge_model_agreement,
        pair_agreement,
        complete_rows,
    )


def analyze_run2(
    ensemble_run: str,
    judges: list[str],
    internal_weights: dict[str, float],
    model_weights: dict[str, float],
    start: int,
    end: int,
) -> None:
    chunks = selected_chunks(ensemble_run, judges, start, end)
    if not chunks:
        raise FileNotFoundError(
            f"No raw Run 2 chunks found for ensemble_run={ensemble_run}, judges={judges}, start={start}, end={end}"
        )

    internal_weight_key = format_weight_key(internal_weights, JUDGE_KEYS)
    model_weight_key = format_weight_key(model_weights, judges)

    print("Run 2 metrics v5")
    print(f"Ensemble run: {ensemble_run}")
    print(f"Judge models: {', '.join(judges)}")
    print(f"Chunks: {chunks[0]}..{chunks[-1]} ({len(chunks)} chunks)")
    print(f"Internal judge Borda weights: {internal_weight_key}")
    print(f"Judge-model Borda weights: {model_weight_key}")
    print("Self-preference policies: no_self_correction, half_self_weight, remove_self_votes")

    records, rows_by_judge_model, errors_by_judge_model = load_records(ensemble_run, judges, chunks)

    judge_model_top1: dict[str, Counter[str]] = defaultdict(Counter)
    internal_judge_top1: dict[str, Counter[str]] = defaultdict(Counter)
    judge_model_by_internal: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    position_exposures: dict[int, Counter[str]] = defaultdict(Counter)

    raw_self_top1_by_judge: dict[str, Counter[str]] = defaultdict(Counter)
    raw_nonself_top1_by_judge: dict[str, Counter[str]] = defaultdict(Counter)

    for _, by_model in sorted(records.items()):
        for judge_model, row_eval in by_model.items():
            for src, pos in row_eval.positions_by_source.items():
                position_exposures[pos][src] += 1
            for internal_judge in JUDGE_KEYS:
                top_id = row_eval.rankings[internal_judge][0]
                src = row_eval.source_by_id[top_id]
                judge_model_top1[judge_model][src] += 1
                internal_judge_top1[internal_judge][src] += 1
                judge_model_by_internal[judge_model][internal_judge][src] += 1
                if same_model_family(judge_model, src):
                    raw_self_top1_by_judge[judge_model][src] += 1
                else:
                    raw_nonself_top1_by_judge[judge_model][src] += 1

    print("\nRows loaded by judge model")
    for judge in judges:
        print(f"  {judge}: {rows_by_judge_model[judge]}")

    print_table(
        "Top-1 counts by judge model, summed across all four internal judges",
        [(judge, judge_model_top1[judge]) for judge in judges],
    )

    print_table(
        "Top-1 counts by internal judge, summed across judge models",
        [(key, internal_judge_top1[key]) for key in JUDGE_KEYS],
    )

    for judge_model in judges:
        print_table(
            f"Top-1 counts for judge model: {judge_model}",
            [(key, judge_model_by_internal[judge_model][key]) for key in JUDGE_KEYS],
        )

    # Self-preference top-1 diagnostic. This is not adjusted Borda; it is raw top-1 lift.
    rows = []
    for judge_model in judges:
        total_top1 = sum(judge_model_top1[judge_model].values())
        own_top1 = sum(raw_self_top1_by_judge[judge_model].values())
        own_rate = own_top1 / total_top1 if total_top1 else 0.0
        own_sources = ",".join(sorted(raw_self_top1_by_judge[judge_model].keys())) or "none"
        rows.append([judge_model, own_sources, own_top1, total_top1, f"{own_rate:.3f}"])
    print_simple_table(
        "Raw top-1 self-family preference by judge model",
        ["judge_model", "own_sources", "own_top1", "total_top1", "own_rate"],
        rows,
    )

    # Cross-model metrics under all self-correction policies.
    for policy_name, self_factor in SELF_POLICIES:
        (
            judges_first_models_second,
            models_first_judges_second,
            pooled_rankings,
            per_model_borda_winners,
            judge_model_agreement,
            pair_agreement,
            complete_rows,
        ) = compute_cross_aggregates(
            records,
            judges,
            internal_weights,
            model_weights,
            self_factor,
        )

        print("\n" + "#" * 88)
        print(f"Self-preference policy: {policy_name} (self_factor={self_factor})")
        print("#" * 88)
        print(f"Complete rows with all requested judge models present: {complete_rows}")

        print_table(
            f"Weighted internal-judge Borda winner counts by judge model ({internal_weight_key})",
            [(judge, per_model_borda_winners[judge]) for judge in judges],
        )

        print_table(
            "Judges first, then models: Borda winners",
            [("judges_then_models", judges_first_models_second)],
        )

        print_table(
            "Models first, then judges: Borda winners",
            [("models_then_judges", models_first_judges_second)],
        )

        print_table(
            "Pooled Borda over all judge-model × internal-judge rankings",
            [("pooled_rankings", pooled_rankings)],
        )

        if judge_model_agreement:
            print("\nPer-row agreement among judge-model Borda winners")
            print("================================================")
            print("distinct_winners  rows")
            for key in sorted(judge_model_agreement.keys(), key=lambda x: int(x)):
                label = f"{key} distinct"
                print(f"{label:16}  {judge_model_agreement[key]}")

        if pair_agreement:
            print("\nPairwise agreement counts among judge-model Borda winners")
            print("========================================================")
            total_rows = sum(1 for _, by_model in records.items() if by_model)
            for key in sorted(pair_agreement):
                print(f"{key}: {pair_agreement[key]} / {total_rows}")

    print_table(
        "Prompt-position exposure counts by generator/source",
        [(str(pos), position_exposures[pos]) for pos in sorted(position_exposures)],
    )

    if errors_by_judge_model:
        print_table(
            "Rows skipped due to discriminator errors or parse errors",
            [(judge, Counter({"errors": errors_by_judge_model[judge]})) for judge in judges],
        )


def usage() -> str:
    return """Usage:
  python metrics_v5.py run2 <ensemble_run> <internal_weights> <model_weights> <start> <end>
  python metrics_v5.py run2 <ensemble_run> <judge_models_csv> <internal_weights> <model_weights> <start> <end>

Examples:
  python metrics_v5.py run2 ensemble 25_25_25_25 25_25_25_25 0 1
  python metrics_v5.py run2 ensemble 45_30_15_10 33_33_34 0 -1
  python metrics_v5.py run2 ensemble claude,gpt,gemini_pro 45_30_15_10 33_33_34 0 -1

Weight order:
  internal_weights: comedian_pun_expert_editor_translator
  model_weights: discovered/provided judge model order printed at runtime
"""


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError(usage())

    task = sys.argv[1]
    if task != "run2":
        raise ValueError(f"Unknown task: {task}\n{usage()}")

    ensemble_run = sys.argv[2] if len(sys.argv) > 2 else "ensemble"

    # Forms:
    #   run2 ensemble internal_weights model_weights start end
    #   run2 ensemble judges_csv internal_weights model_weights start end
    if len(sys.argv) >= 8 and "," in sys.argv[3]:
        judges = [resolve_model_alias(x.strip()) for x in sys.argv[3].split(",") if x.strip()]
        internal_weight_string = sys.argv[4]
        model_weight_string = sys.argv[5]
        start = int(sys.argv[6])
        end = int(sys.argv[7])
    elif len(sys.argv) >= 7:
        judges = discover_judges(ensemble_run)
        internal_weight_string = sys.argv[3]
        model_weight_string = sys.argv[4]
        start = int(sys.argv[5])
        end = int(sys.argv[6])
    else:
        raise ValueError(usage())

    if not judges:
        raise FileNotFoundError(
            f"No judge model directories found under {ensure_slash(OUTPUT_ROOT)}{ensure_slash(ensemble_run)}"
        )

    internal_weights = parse_weight_string(internal_weight_string, JUDGE_KEYS, "internal judge", allow_extra=False)
    model_weights = parse_weight_string(model_weight_string, judges, "judge model", allow_extra=True)

    analyze_run2(ensemble_run, judges, internal_weights, model_weights, start, end)


if __name__ == "__main__":
    main()
