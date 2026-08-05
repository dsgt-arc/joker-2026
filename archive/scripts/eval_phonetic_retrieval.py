from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from retrieval_v18 import PhoneticRetriever, clean


def load_eval_pairs(path: Path, split: str = "dev", include_synthetic: bool = False):
    df = pd.read_csv(path, sep="\t")
    df = df[df["anchor_split"].eq(split)].copy()

    if not include_synthetic and "candidate_id" in df.columns:
        df = df[df["candidate_id"].astype(int) >= 0].copy()

    positives = defaultdict(set)
    rel_positives = defaultdict(lambda: defaultdict(set))

    for _, r in df.iterrows():
        q = clean(r["anchor_ipa"])
        t = clean(r["candidate_ipa"])
        rel = clean(r["relation_type"])

        if not q or not t or q == t:
            continue

        positives[q].add(t)
        rel_positives[rel][q].add(t)

    return positives, rel_positives


def evaluate(retriever, positives, ks=(1, 5, 10, 25, 50), batch_size=2048):
    queries = list(positives.keys())
    max_k = max(ks) + 20

    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    n = 0

    for i in tqdm(range(0, len(queries), batch_size), desc="evaluating"):
        batch = queries[i : i + batch_size]
        results = retriever.search_many(batch, top_k=max_k)

        for q in batch:
            pos = positives[q]

            retrieved = []
            for r in results.get(q, []):
                ipa = clean(r.get("ipa", ""))
                if ipa and ipa != q:
                    retrieved.append(ipa)

            n += 1

            for k in ks:
                if any(x in pos for x in retrieved[:k]):
                    hits[k] += 1

            for rank, ipa in enumerate(retrieved, start=1):
                if ipa in pos:
                    rr_sum += 1.0 / rank
                    break

    row = {f"R@{k}": hits[k] / n if n else 0.0 for k in ks}
    row["MRR"] = rr_sum / n if n else 0.0
    row["queries"] = n
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="data/retrieval/phonetic",
        help="Directory containing eval_qrels.tsv",
    )
    ap.add_argument("--qrels", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-synthetic", action="store_true")
    ap.add_argument("--min-queries-per-relation", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    qrels_path = Path(args.qrels) if args.qrels else data_dir / "eval_qrels.tsv"
    out_path = Path(args.out) if args.out else data_dir / "phonetic_eval_results_v2.csv"

    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

    positives, rel_positives = load_eval_pairs(
        qrels_path,
        split=args.split,
        include_synthetic=args.include_synthetic,
    )

    print(f"Loaded {len(positives):,} query IPA forms from {qrels_path}")

    retriever = PhoneticRetriever(top_k=70)

    rows = []
    rows.append({
        "relation": "all",
        **evaluate(
            retriever,
            positives,
            batch_size=args.batch_size,
        ),
    })

    for rel in sorted(rel_positives):
        rel_pos = rel_positives[rel]
        if len(rel_pos) < args.min_queries_per_relation:
            continue

        rows.append({
            "relation": rel,
            **evaluate(
                retriever,
                rel_pos,
                batch_size=args.batch_size,
            ),
        })

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c.startswith("R@") or c == "MRR"]
    for c in metric_cols:
        df[c] = df[c].map(lambda x: round(float(x), 4))

    df = df.sort_values(["relation"]).reset_index(drop=True)
    print(df.to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()