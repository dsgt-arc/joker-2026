#!/usr/bin/env python3
import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


@dataclass
class TrainConfig:
    base_model: str
    train_path: str
    train_file_type: str
    language: str
    output_root: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    temperature: float
    max_length: int
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    corpus_path: Optional[str]
    corpus_file_type: str
    corpus_text_col: str
    skip_faiss: bool


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.anchor = df["anchor"].tolist()
        self.positive = df["positive"].tolist()
        self.hard_negative = df["hard_negative"].fillna("").tolist()

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        return {
            "anchor": self.anchor[idx],
            "positive": self.positive[idx],
            "hard_negative": self.hard_negative[idx],
        }


def collate_fn(items):
    return {
        "anchor": [x["anchor"] for x in items],
        "positive": [x["positive"] for x in items],
        "hard_negative": [x["hard_negative"] for x in items],
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(path: str, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(path)
    if file_type == "parquet":
        return pd.read_parquet(path)
    raise ValueError("file_type must be csv or parquet")


def load_pairs(path: str, file_type: str, language: str) -> pd.DataFrame:
    df = read_table(path, file_type)
    required = {"language", "anchor", "positive"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "hard_negative" not in df.columns:
        df["hard_negative"] = ""

    df = df.dropna(subset=["language", "anchor", "positive"]).copy()
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df = df[df["language"] == language].reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Need at least 100 rows for language={language}; got {len(df)}")
    return df


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode_texts(model, tokenizer, texts, max_length: int, device: str):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**batch)
    if hasattr(out, "last_hidden_state"):
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
    else:
        emb = out[0].mean(dim=1)
    return F.normalize(emb, p=2, dim=1)


def make_model_and_tokenizer(cfg: TrainConfig, device: str):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_cfg)
    model.to(device)
    model.print_trainable_parameters()
    return model, tokenizer


@torch.no_grad()
def evaluate_recall_at_k(model, tokenizer, df_eval, cfg: TrainConfig, device: str, k_values=(1, 5, 10)):
    model.eval()
    anchors = df_eval["anchor"].tolist()
    positives = df_eval["positive"].tolist()

    emb_a, emb_p = [], []
    for i in range(0, len(anchors), cfg.batch_size):
        emb_a.append(encode_texts(model, tokenizer, anchors[i:i + cfg.batch_size], cfg.max_length, device).cpu())
        emb_p.append(encode_texts(model, tokenizer, positives[i:i + cfg.batch_size], cfg.max_length, device).cpu())

    emb_a = torch.cat(emb_a, dim=0)
    emb_p = torch.cat(emb_p, dim=0)
    sim = emb_a @ emb_p.T

    ranks = torch.argsort(sim, dim=1, descending=True)
    labels = torch.arange(sim.size(0)).unsqueeze(1)

    metrics = {}
    for k in k_values:
        topk = ranks[:, :k]
        metrics[f"recall@{k}"] = (topk == labels).any(dim=1).float().mean().item()
    return metrics


def encode_corpus(model, tokenizer, texts, cfg: TrainConfig, device: str):
    all_emb = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), cfg.batch_size), desc="Encoding corpus"):
            batch = texts[i:i + cfg.batch_size]
            emb = encode_texts(model, tokenizer, batch, cfg.max_length, device)
            all_emb.append(emb.detach().cpu().numpy())
    return np.vstack(all_emb).astype("float32")


def build_and_save_faiss(embeddings: np.ndarray, out_dir: str, prefix: str):
    embs = embeddings.copy()
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, f"{prefix}.faiss")
    emb_path = os.path.join(out_dir, f"{prefix}_embeddings.npy")
    faiss.write_index(index, idx_path)
    np.save(emb_path, embs)
    return idx_path, emb_path


def train(cfg: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    set_seed(cfg.seed)

    df = load_pairs(cfg.train_path, cfg.train_file_type, cfg.language)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=cfg.seed)

    train_loader = DataLoader(
        PairDataset(train_df),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model, tokenizer = make_model_and_tokenizer(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * cfg.warmup_ratio),
        total_steps,
    )

    out_dir = os.path.join(cfg.output_root, f"qwen3-emb-8b-{cfg.language}-lora")
    os.makedirs(out_dir, exist_ok=True)

    best_r1 = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"{cfg.language} epoch {epoch + 1}/{cfg.epochs}"):
            emb_a = encode_texts(model, tokenizer, batch["anchor"], cfg.max_length, device)
            emb_p = encode_texts(model, tokenizer, batch["positive"], cfg.max_length, device)

            sim_ap = (emb_a @ emb_p.T) / cfg.temperature
            labels = torch.arange(sim_ap.size(0), device=sim_ap.device)

            hard_neg = [x for x in batch["hard_negative"] if isinstance(x, str) and x.strip()]
            if len(hard_neg) == len(batch["anchor"]):
                emb_n = encode_texts(model, tokenizer, batch["hard_negative"], cfg.max_length, device)
                sim_an = (emb_a @ emb_n.T) / cfg.temperature
                logits = torch.cat([sim_ap, sim_an], dim=1)
            else:
                logits = sim_ap

            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        metrics = evaluate_recall_at_k(model, tokenizer, val_df, cfg, device)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f} metrics={metrics}")

        if metrics["recall@1"] > best_r1:
            best_r1 = metrics["recall@1"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"saved best adapter to {out_dir}")

    if cfg.skip_faiss:
        return

    if not cfg.corpus_path:
        raise ValueError("corpus_path is required when --skip-faiss is not set")

    corpus = read_table(cfg.corpus_path, cfg.corpus_file_type)
    if cfg.corpus_text_col not in corpus.columns:
        raise ValueError(f"Missing corpus text column: {cfg.corpus_text_col}")

    corpus = corpus.dropna(subset=[cfg.corpus_text_col]).reset_index(drop=True)

    base = AutoModel.from_pretrained(cfg.base_model, trust_remote_code=True, torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32))
    best_model = PeftModel.from_pretrained(base, out_dir)
    best_model.to(device)
    best_model.eval()

    corpus_embeddings = encode_corpus(
        best_model,
        tokenizer,
        corpus[cfg.corpus_text_col].astype(str).tolist(),
        cfg,
        device,
    )
    faiss_dir = os.path.join(cfg.output_root, "faiss")
    idx_path, emb_path = build_and_save_faiss(corpus_embeddings, faiss_dir, f"{cfg.language}_qwen3_emb8b")
    print(f"faiss_index={idx_path}")
    print(f"embedding_matrix={emb_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Qwen3 embedding adapter for one language and optionally build FAISS index")
    p.add_argument("--language", choices=["fr", "es"], required=True)
    p.add_argument("--train-path", required=True)
    p.add_argument("--train-file-type", choices=["csv", "parquet"], default="csv")
    p.add_argument("--base-model", default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--output-root", default="artifacts/embeddings")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--corpus-path", default=None)
    p.add_argument("--corpus-file-type", choices=["csv", "parquet"], default="csv")
    p.add_argument("--corpus-text-col", default="text")
    p.add_argument("--skip-faiss", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        base_model=args.base_model,
        train_path=args.train_path,
        train_file_type=args.train_file_type,
        language=args.language,
        output_root=args.output_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        max_length=args.max_length,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        corpus_path=args.corpus_path,
        corpus_file_type=args.corpus_file_type,
        corpus_text_col=args.corpus_text_col,
        skip_faiss=args.skip_faiss,
    )
    train(cfg)
