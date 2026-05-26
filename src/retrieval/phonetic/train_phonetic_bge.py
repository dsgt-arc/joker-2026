import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from datasets import Dataset
import sentence_transformers.fit_mixin as fit_mixin
fit_mixin.Dataset = Dataset

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


SEED = 13
random.seed(SEED)

MODEL_DIR = Path("/storage/scratch1/0/rtaylor351/joker_retrieval/phonetic/models")

TRAIN_PATH = Path(
    os.environ.get(
        "PHONETIC_TRAIN_PATH",
        "/storage/scratch1/0/rtaylor351/joker_retrieval/phonetic/data_rebuilt/train_pairs_clean.tsv",
    )
)

BASE_MODEL = os.environ.get("PHONETIC_BASE_MODEL", "BAAI/bge-m3")
OUTPUT_DIR = Path(
    os.environ.get(
        "PHONETIC_OUTPUT_DIR",
        str(MODEL_DIR / "bge-m3-ipa-rebuilt-v1"),
    )
)

BATCH_SIZE = int(os.environ.get("PHONETIC_BATCH_SIZE", "128"))
EPOCHS = int(os.environ.get("PHONETIC_EPOCHS", "1"))
MAX_ROWS = int(os.environ.get("PHONETIC_MAX_ROWS", "0"))

RELATION_TARGETS = {
    "identity": 200_000,
    "exact_homophone": 200_000,
    "near_homophone_edit1": 160_000,
    "near_homophone_edit2": 160_000,
    "synthetic_schwa_drop": 40_000,
    "synthetic_r_variant": 40_000,
    "synthetic_nasal_variant": 40_000,
    "synthetic_boundary_drop": 2_000,
    "strong_rhyme": 120_000,
    "weak_rhyme": 30_000,
    "consonant_skeleton": 80_000,
    "vowel_skeleton": 40_000,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def read_train_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing training file: {path}")

    log(f"[1/7] Loading training TSV: {path}")
    log(f"      File size: {path.stat().st_size / (1024 * 1024):.1f} MB")
    start = time.time()
    df = pd.read_csv(path, sep="\t")
    log(f"      Loaded rows: {len(df):,} in {time.time() - start:.1f}s")
    return df


def validate_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    log("[2/7] Validating columns and filtering empty IPA rows")

    needed = {"anchor_ipa", "candidate_ipa", "relation_type"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=["anchor_ipa", "candidate_ipa", "relation_type"]).copy()
    df["anchor_ipa"] = df["anchor_ipa"].astype(str)
    df["candidate_ipa"] = df["candidate_ipa"].astype(str)
    df = df[
        (df["anchor_ipa"].str.len() > 0)
        & (df["candidate_ipa"].str.len() > 0)
    ].copy()

    log(f"      Rows before: {before:,}")
    log(f"      Rows after:  {len(df):,}")
    log("      Relation counts:")
    log(df["relation_type"].value_counts().to_string())
    return df


def balanced_sample(df: pd.DataFrame) -> pd.DataFrame:
    log("[3/7] Balanced sampling by relation type")

    parts = []
    per_type_cap = None

    if MAX_ROWS > 0:
        per_type_cap = max(1, MAX_ROWS // len(RELATION_TARGETS))
        log(f"      MAX_ROWS={MAX_ROWS:,}; per-type cap={per_type_cap:,}")

    for relation_type, target_n in tqdm(
        RELATION_TARGETS.items(),
        desc="sampling relation types",
        unit="type",
    ):
        sub = df[df["relation_type"] == relation_type]
        if len(sub) == 0:
            log(f"      WARNING: no rows for {relation_type}")
            continue

        n = target_n if per_type_cap is None else min(target_n, per_type_cap)
        replace = len(sub) < n

        sampled = sub.sample(n=n, replace=replace, random_state=SEED)
        parts.append(sampled)

        log(
            f"      {relation_type}: "
            f"source={len(sub):,}, sampled={len(sampled):,}, replace={replace}"
        )

    if not parts:
        raise RuntimeError("No training rows sampled.")

    out = pd.concat(parts, ignore_index=True)
    log("      Shuffling sampled rows")
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    if MAX_ROWS > 0 and len(out) > MAX_ROWS:
        out = out.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)

    log(f"      Final sampled rows: {len(out):,}")
    log("      Sampled relation counts:")
    log(out["relation_type"].value_counts().to_string())
    return out


def build_examples(train_df: pd.DataFrame) -> list[InputExample]:
    log("[4/7] Building SentenceTransformers InputExample objects")

    examples = []
    for row in tqdm(
        train_df.itertuples(index=False),
        total=len(train_df),
        desc="creating InputExample",
        unit="rows",
    ):
        examples.append(InputExample(texts=[row.anchor_ipa, row.candidate_ipa]))

    log(f"      Built examples: {len(examples):,}")
    return examples


def main() -> None:
    log("Starting phonetic embedding training")
    log(f"TRAIN_PATH={TRAIN_PATH}")
    log(f"BASE_MODEL={BASE_MODEL}")
    log(f"OUTPUT_DIR={OUTPUT_DIR}")
    log(f"MAX_ROWS={MAX_ROWS}")
    log(f"BATCH_SIZE={BATCH_SIZE}")
    log(f"EPOCHS={EPOCHS}")

    df = read_train_file(TRAIN_PATH)
    df = validate_and_filter(df)
    train_df = balanced_sample(df)
    examples = build_examples(train_df)

    log("[5/7] Building DataLoader")
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    log(f"      Training batches per epoch: {len(loader):,}")

    log("[6/7] Loading embedding model")
    model = SentenceTransformer(BASE_MODEL, trust_remote_code=True)
    log("      Model loaded")

    log("[7/7] Training")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(100, int(len(loader) * EPOCHS * 0.05))
    log(f"      Warmup steps: {warmup_steps:,}")
    log(f"      Output dir: {OUTPUT_DIR}")

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
    )

    log(f"Saved model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
