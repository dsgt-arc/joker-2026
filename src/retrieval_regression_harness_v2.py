#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


FLOAT_TOLERANCE_COLUMNS = {
    "retrieval_best_bridge_score": 1e-6,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run retrieval_refactored.py with the same CLI shape used in development and compare "
            "the produced TSV against a baseline. All columns must match exactly except "
            "configured float-jitter columns."
        )
    )
    p.add_argument("--project-root", required=True, help="Project root or src directory where retrieval_refactored.py can be run.")
    p.add_argument("--baseline", required=True, help="Baseline TSV path.")
    p.add_argument("--model", default="gemini")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--remove-actual-before-run", action="store_true")
    return p.parse_args()


def resolve_run_cwd(project_root: Path) -> Path:
    project_root = project_root.resolve()
    if (project_root / "retrieval_refactored.py").exists():
        return project_root
    if (project_root / "src" / "retrieval_refactored.py").exists():
        return project_root / "src"
    raise FileNotFoundError(f"Could not find retrieval_refactored.py under {project_root} or {project_root / 'src'}")


def default_actual_path_from_cwd(cwd: Path, model: str, start: int) -> Path:
    retrieval_file = (cwd / "retrieval_refactored.py").resolve()
    repo_root = retrieval_file.parents[1]
    chunk = start // 100
    return repo_root / "data" / "processed" / "retrieval" / model / f"{chunk}.tsv"


def extract_last_json_object(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    for i in range(len(text) - 1, -1, -1):
        if text[i] != "{":
            continue
        try:
            obj = json.loads(text[i:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def run_retrieval(args: argparse.Namespace, cwd: Path) -> Path:
    actual_guess = default_actual_path_from_cwd(cwd, args.model, args.start)
    if args.remove_actual_before_run and actual_guess.exists():
        actual_guess.unlink()

    cmd = [args.python, "retrieval_refactored.py", "retrieve", args.model, str(args.start), str(args.end)]
    print("Running:", " ".join(cmd))
    print("cwd:", cwd)
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)

    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")

    if proc.returncode != 0:
        raise SystemExit(f"Retrieval command failed with exit code {proc.returncode}")

    response = extract_last_json_object(proc.stdout)
    actual = Path(response["rows_path"]).expanduser().resolve() if response.get("rows_path") else actual_guess
    if not actual.exists():
        raise SystemExit(f"Actual output not found after run: {actual}")
    return actual


def compare_files(actual: Path, baseline: Path) -> int:
    if actual.read_bytes() == baseline.read_bytes():
        print("PASS: byte-for-byte identical")
        return 0

    adf = pd.read_csv(actual, sep="\t")
    bdf = pd.read_csv(baseline, sep="\t")

    print(f"actual bytes:   {actual.stat().st_size}")
    print(f"baseline bytes: {baseline.stat().st_size}")
    print(f"actual shape:   {adf.shape}")
    print(f"baseline shape: {bdf.shape}")

    if adf.shape != bdf.shape:
        print("FAIL: shape mismatch")
        return 1

    if list(adf.columns) != list(bdf.columns):
        print("FAIL: column order/name mismatch")
        print("actual columns:", list(adf.columns))
        print("baseline columns:", list(bdf.columns))
        return 1

    failures: list[str] = []
    tolerated: list[str] = []

    for col in adf.columns:
        if col in FLOAT_TOLERANCE_COLUMNS:
            tol = FLOAT_TOLERANCE_COLUMNS[col]
            av = pd.to_numeric(adf[col], errors="coerce")
            bv = pd.to_numeric(bdf[col], errors="coerce")
            both_nan = av.isna() & bv.isna()
            diff = (av - bv).abs()
            bad = ~(both_nan | (diff <= tol))
            if bad.any():
                rows = list(bad[bad].index[:10])
                failures.append(f"{col}: {int(bad.sum())} rows exceed tol={tol}; first rows={rows}")
            elif diff.fillna(0).gt(0).any():
                tolerated.append(f"{col}: tolerated max diff {float(diff.max())}")
            continue

        neq = adf[col].astype(str) != bdf[col].astype(str)
        if neq.any():
            rows = list(neq[neq].index[:10])
            failures.append(f"{col}: {int(neq.sum())} exact mismatches; first rows={rows}")

    if failures:
        print("FAIL: differences found")
        for item in failures:
            print("  -", item)
        return 1

    print("PASS: equivalent under approved tolerance")
    for item in tolerated:
        print("  -", item)
    return 0


def main() -> None:
    args = parse_args()
    cwd = resolve_run_cwd(Path(args.project_root))
    baseline = Path(args.baseline).expanduser().resolve()
    if not baseline.exists():
        raise SystemExit(f"Baseline not found: {baseline}")
    actual = run_retrieval(args, cwd)
    raise SystemExit(compare_files(actual, baseline))


if __name__ == "__main__":
    main()
