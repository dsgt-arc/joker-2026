"""
Quality-first discriminator for French pun candidates.

Input: TSV chunks from generator.py with candidate_json.
Usage:
  python discriminator.py judge google/gemini-3-pro 0 -1

Saves chunks to: generate_dir/{model}/judged/{chunk}.tsv
"""

import asyncio
import ast
import itertools
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Awaitable, Callable

import pandas as pd

from config import generate_dir
from data import load_all, save
from utils import get_response_async

DEFAULT_MODEL = os.environ.get("DISCRIMINATOR_MODEL", "anthropic/claude-3.5-sonnet")
MAX_CONCURRENCY = int(os.environ.get("DISCRIMINATOR_MAX_CONCURRENCY", "4"))
CHUNK_SIZE = int(os.environ.get("DISCRIMINATOR_CHUNK_SIZE", "50"))
MAX_CANDIDATES = int(os.environ.get("DISCRIMINATOR_MAX_CANDIDATES", "12"))
VERBOSE = os.environ.get("DISCRIMINATOR_VERBOSE", "1") == "1"
RANDOM_SEED = int(os.environ.get("DISCRIMINATOR_RANDOM_SEED", "17"))

PAIRWISE_JUDGES = [
    "native_french_humor",
    "french_wordplay_mechanics",
    "comedic_intent_preservation",
    "joker_compliance",
]

JUDGE_WEIGHTS = {
    "native_french_humor": 4.0,
    "french_wordplay_mechanics": 3.0,
    "comedic_intent_preservation": 1.5,
    "joker_compliance": 2.0,
}

OUTPUT_COLUMNS = [
    "final_pun",
    "final_candidate_index",
    "final_score",
    "pairwise_json",
    "audit_json",
    "repair_json",
    "discriminator_error",
]


def log(*args: Any) -> None:
    if VERBOSE:
        print(*args)


def norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def safe_json_loads(x: Any, fallback: Any) -> Any:
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str) or not x.strip():
        return fallback
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return fallback


async def run_async_apply(
    chunk_df: pd.DataFrame,
    apply_async_fn: Callable[[pd.Series], Awaitable[pd.Series]],
    result_columns: list[str],
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(index, row):
        async with semaphore:
            result = await apply_async_fn(row)
            return index, result

    tasks = [asyncio.create_task(worker(index, row)) for index, row in chunk_df.iterrows()]
    results: dict[Any, pd.Series] = {}
    try:
        for task in asyncio.as_completed(tasks):
            index, result = await task
            results[index] = result
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    ordered = [results[index] for index in chunk_df.index]
    result_df = pd.DataFrame(ordered, index=chunk_df.index)
    return result_df[result_columns]


PAIRWISE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "winner": {"type": "string"},
        "confidence": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["winner", "confidence", "reason"],
}

AUDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "humor": {"type": "integer"},
        "recognizability": {"type": "integer"},
        "authenticity": {"type": "integer"},
        "wordplay_success": {"type": "integer"},
        "source_relation": {"type": "integer"},
        "syntax_errors": {"type": "integer"},
        "severe_errors": {"type": "array", "items": {"type": "string"}},
        "minor_errors": {"type": "array", "items": {"type": "string"}},
        "diagnosis": {"type": "string"},
    },
    "required": [
        "humor", "recognizability", "authenticity", "wordplay_success",
        "source_relation", "syntax_errors", "severe_errors", "minor_errors", "diagnosis"
    ],
}

REPAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "repaired_french": {"type": "string"},
        "changed": {"type": "integer"},
        "why_better": {"type": "string"},
    },
    "required": ["repaired_french", "changed", "why_better"],
}


def judge_prompt(judge: str, row: pd.Series, a: dict[str, Any], b: dict[str, Any], a_label: str, b_label: str) -> str:
    english = norm_space(row.get("text_clean", ""))
    pun_word = norm_space(row.get("pun_word", ""))
    pun_type = norm_space(row.get("pun_type", ""))

    if judge == "native_french_humor":
        task = """
You are a native French humor judge. You see only the French candidates.
Choose the sentence that is funnier, clearer, more idiomatic, and more immediately recognizable as a pun to native French speakers.
Do not consider literal faithfulness to English.
"""
        context = ""
    elif judge == "french_wordplay_mechanics":
        task = """
You are a French wordplay mechanics judge. You see only the French candidates.
Choose the candidate with stronger functional wordplay: double meaning, homophony/homography, idiomatic twist, elegance, and immediate click.
Do not reward random absurdity without real wordplay.
"""
        context = ""
    elif judge == "comedic_intent_preservation":
        task = """
You are a bilingual comedic adaptation judge.
Do NOT reward literal translation. Creative adaptation is allowed and often better.
Choose the candidate that better recreates the source joke's comedic experience: ambiguity, surprise, idiomatic twist, groan-worthy pun energy, or lexical collision.
"""
        context = f"English joke: {english}\nEnglish pun word: {pun_word}\nEnglish pun type: {pun_type}\n"
    elif judge == "joker_compliance":
        task = """
You are judging for a JOKER-style wordplay translation task.
Choose the candidate that better preserves, to the extent possible, both the form and sense of the original wordplay while remaining funny and natural French.
Do not require literal wording. Do penalize candidates that become unrelated generic jokes or lose wordplay entirely.
"""
        context = f"English joke: {english}\nEnglish pun word: {pun_word}\nEnglish pun type: {pun_type}\n"
    else:
        raise ValueError(judge)

    return f"""
{task.strip()}

{context}
Candidate {a_label}: {a.get('french', '')}
Candidate {b_label}: {b.get('french', '')}

Return only valid JSON:
{{
  "winner": "{a_label}|{b_label}|tie",
  "confidence": 0-3,
  "reason": "brief English reason"
}}
""".strip()


async def pairwise_vote(
    row: pd.Series,
    a: dict[str, Any],
    b: dict[str, Any],
    judge: str,
    model: str,
    swapped: bool,
) -> dict[str, Any]:
    a_label, b_label = ("B", "A") if swapped else ("A", "B")
    left, right = (b, a) if swapped else (a, b)
    prompt = judge_prompt(judge, row, left, right, a_label, b_label)
    resp = await get_response_async(
        prompt,
        model,
        response_schema=PAIRWISE_SCHEMA,
        required_keys=["winner", "confidence", "reason"],
        routing_preset="stable",
    )
    winner_label = str(resp.get("winner", "tie")).strip().upper()
    if swapped:
        winner = "a" if winner_label == "A" else "b" if winner_label == "B" else "tie"
    else:
        winner = "a" if winner_label == "A" else "b" if winner_label == "B" else "tie"
    return {
        "judge": judge,
        "winner": winner,
        "confidence": int(resp.get("confidence", 1) or 1),
        "reason": str(resp.get("reason", "")),
        "swapped": swapped,
    }


async def compare_pair(row: pd.Series, candidates: list[dict[str, Any]], i: int, j: int, model: str) -> list[dict[str, Any]]:
    votes: list[dict[str, Any]] = []
    for judge in PAIRWISE_JUDGES:
        # Order-swap check for bias mitigation. Quality-first: pay for both.
        for swapped in (False, True):
            try:
                vote = await pairwise_vote(row, candidates[i], candidates[j], judge, model, swapped)
                vote["i"] = i
                vote["j"] = j
                votes.append(vote)
            except Exception as e:
                votes.append({"judge": judge, "i": i, "j": j, "winner": "error", "confidence": 0, "reason": str(e), "swapped": swapped})
    return votes


def rank_from_votes(n: int, votes: list[dict[str, Any]]) -> dict[int, float]:
    scores = defaultdict(float)
    for v in votes:
        judge = v.get("judge", "")
        weight = JUDGE_WEIGHTS.get(judge, 1.0)
        conf = max(1, min(3, int(v.get("confidence", 1) or 1)))
        value = weight * conf
        i, j = int(v["i"]), int(v["j"])
        if v.get("winner") == "a":
            scores[i] += value
        elif v.get("winner") == "b":
            scores[j] += value
        elif v.get("winner") == "tie":
            scores[i] += value * 0.5
            scores[j] += value * 0.5
    for k in range(n):
        scores[k] += 0.0
    return dict(scores)


def tournament_pairs(n: int) -> list[tuple[int, int]]:
    # Full round-robin up to MAX_CANDIDATES=12 is 66 pairs; expensive but quality-first.
    return list(itertools.combinations(range(n), 2))


async def audit_candidate(row: pd.Series, candidate: dict[str, Any], model: str) -> dict[str, Any]:
    prompt = f"""
You are doing a strict final audit of a French pun candidate for a JOKER-style wordplay translation task.

English source joke:
{norm_space(row.get('text_clean', ''))}

Candidate French pun:
{candidate.get('french', '')}

Score:
- humor: 0-3, funny to native French speakers
- recognizability: 0-3, joke is obvious quickly
- authenticity: 0-4, natural idiomatic French
- wordplay_success: 0-3, functional pun/double meaning
- source_relation: 0-2, recognizable relation to source sense/mechanism; do not demand literalness
- syntax_errors: 0-2, where 2 means no syntax/word-choice errors

List severe errors only if they seriously threaten success. Use these labels when applicable:
broken_french, unnatural_french, no_wordplay, obscure_wordplay, weak_humor, source_relation_lost, mechanism_lost, syntax_error, word_choice_error.

Return only valid JSON.
""".strip()
    return await get_response_async(
        prompt,
        model,
        response_schema=AUDIT_SCHEMA,
        required_keys=list(AUDIT_SCHEMA["properties"].keys()),
        routing_preset="stable",
    )


def audit_score(a: dict[str, Any]) -> float:
    base = (
        4 * int(a.get("humor", 0))
        + 3 * int(a.get("recognizability", 0))
        + 3 * int(a.get("wordplay_success", 0))
        + 2 * int(a.get("authenticity", 0))
        + 1 * int(a.get("source_relation", 0))
        + 2 * int(a.get("syntax_errors", 0))
    )
    severe_penalty = 6 * len(a.get("severe_errors", []) or [])
    return float(base - severe_penalty)


async def repair_candidate(row: pd.Series, candidate: dict[str, Any], audit: dict[str, Any], model: str) -> dict[str, Any]:
    severe = audit.get("severe_errors", []) or []
    minor = audit.get("minor_errors", []) or []
    if not severe:
        return {"repaired_french": candidate.get("french", ""), "changed": 0, "why_better": "No severe repair needed."}

    prompt = f"""
Improve this French pun candidate. Keep the successful parts. Fix only the diagnosed problems.
Do not make it more literal unless that improves the joke.

English source joke:
{norm_space(row.get('text_clean', ''))}

Current French candidate:
{candidate.get('french', '')}

Diagnosis:
Severe errors: {severe}
Minor errors: {minor}
Notes: {audit.get('diagnosis', '')}

Return only valid JSON:
{{
  "repaired_french": "one improved French pun sentence",
  "changed": 0 or 1,
  "why_better": "brief English explanation"
}}
""".strip()
    return await get_response_async(
        prompt,
        model,
        response_schema=REPAIR_SCHEMA,
        required_keys=["repaired_french", "changed", "why_better"],
        routing_preset="stable",
    )


async def judge_row(row: pd.Series, model: str) -> pd.Series:
    errors: list[str] = []
    candidates = safe_json_loads(row.get("candidate_json", "[]"), [])
    if not isinstance(candidates, list):
        candidates = []
    candidates = [c for c in candidates if isinstance(c, dict) and norm_space(c.get("french", ""))]
    candidates = candidates[:MAX_CANDIDATES]

    if not candidates:
        return pd.Series({
            "final_pun": "ERROR",
            "final_candidate_index": -1,
            "final_score": 0.0,
            "pairwise_json": "[]",
            "audit_json": "[]",
            "repair_json": "[]",
            "discriminator_error": "no candidates",
        })

    if len(candidates) == 1:
        return pd.Series({
            "final_pun": candidates[0]["french"],
            "final_candidate_index": 0,
            "final_score": 0.0,
            "pairwise_json": "[]",
            "audit_json": "[]",
            "repair_json": "[]",
            "discriminator_error": "single candidate",
        })

    rng = random.Random(RANDOM_SEED + int(row.name if str(row.name).isdigit() else 0))
    pairs = tournament_pairs(len(candidates))
    rng.shuffle(pairs)

    all_votes: list[dict[str, Any]] = []
    # Sequential per row to avoid API storm; concurrency happens across rows.
    for i, j in pairs:
        all_votes.extend(await compare_pair(row, candidates, i, j, model))

    pair_scores = rank_from_votes(len(candidates), all_votes)
    top_indices = sorted(pair_scores, key=lambda k: pair_scores[k], reverse=True)[:3]

    audits: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for idx in top_indices:
        try:
            audit = await audit_candidate(row, candidates[idx], model)
            audit["candidate_index"] = idx
            audit["candidate_french"] = candidates[idx].get("french", "")
            audit["pairwise_score"] = pair_scores[idx]
            audit["audit_score"] = audit_score(audit)
            audits.append(audit)

            repair = await repair_candidate(row, candidates[idx], audit, model)
            repair["candidate_index"] = idx
            repairs.append(repair)
        except Exception as e:
            errors.append(f"audit/repair candidate {idx}: {e}")

    # Include repaired candidates if changed, then audit them lightly by score heuristic via final pairwise mini-tournament.
    final_pool: list[dict[str, Any]] = []
    for a in audits:
        idx = a["candidate_index"]
        final_pool.append({"french": candidates[idx]["french"], "origin_index": idx, "kind": "original", "prior_score": pair_scores[idx] + audit_score(a)})
    for r in repairs:
        if int(r.get("changed", 0) or 0) == 1 and norm_space(r.get("repaired_french", "")):
            final_pool.append({"french": norm_space(r["repaired_french"]), "origin_index": r["candidate_index"], "kind": "repair", "prior_score": pair_scores[r["candidate_index"]]})

    if len(final_pool) == 1:
        winner = final_pool[0]
        final_score = float(winner.get("prior_score", 0.0))
    else:
        mini_votes: list[dict[str, Any]] = []
        for i, j in itertools.combinations(range(len(final_pool)), 2):
            mini_votes.extend(await compare_pair(row, final_pool, i, j, model))
        all_votes.extend([{**v, "final_round": True} for v in mini_votes])
        mini_scores = rank_from_votes(len(final_pool), mini_votes)
        best_pool_idx = max(mini_scores, key=lambda k: mini_scores[k] + 0.01 * float(final_pool[k].get("prior_score", 0.0)))
        winner = final_pool[best_pool_idx]
        final_score = mini_scores[best_pool_idx] + 0.01 * float(winner.get("prior_score", 0.0))

    log(row.name, "winner=", winner["french"][:120])
    return pd.Series({
        "final_pun": winner["french"],
        "final_candidate_index": int(winner.get("origin_index", -1)),
        "final_score": round(final_score, 3),
        "pairwise_json": json.dumps(all_votes, ensure_ascii=False),
        "audit_json": json.dumps(audits, ensure_ascii=False),
        "repair_json": json.dumps(repairs, ensure_ascii=False),
        "discriminator_error": " | ".join(errors),
    })


async def judge_candidates(df: pd.DataFrame, model: str, start: int = 0, end: int = -1) -> None:
    chunks = [df.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
    end = len(chunks) if end == -1 else end

    for i in range(start, end):
        chunk = chunks[i].copy()
        chunk[OUTPUT_COLUMNS] = await run_async_apply(chunk, lambda row: judge_row(row, model), OUTPUT_COLUMNS)
        save(chunk, f"{generate_dir}{model}/judged/{i}.tsv")


async def main() -> None:
    task = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    if task == "judge":
        df = load_all(f"{generate_dir}{model}/candidates/")
        await judge_candidates(df, model, start, end)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    asyncio.run(main())
