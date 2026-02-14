from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Any, Tuple, Optional
import re


def cartesian(product_dict: Dict[str, Sequence]) -> List[Dict[str, any]]:
    keys = list(product_dict.keys())
    combos = []
    for values in itertools.product(*[product_dict[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def merge_spaces(spaces: Sequence[Dict[str, Sequence]]) -> List[Dict[str, any]]:
    """Expand and deduplicate multiple Cartesian product spaces."""
    merged: List[Dict[str, any]] = []
    seen = set()
    for space in spaces:
        for cfg in cartesian(space):
            key = json.dumps(cfg, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(cfg)
    return merged


def build_space(expert: str) -> List[Dict[str, Any]]:
    """Return an expanded, targeted grid around observed strong regions."""
    if expert == "sbert":
        spaces = [
            # Frozen-encoder core region (best in prior sweep).
            {
                "lr": [5e-5, 7e-5, 1e-4, 1.5e-4],
                "weight_decay": [0.03, 0.05],
                "dropout": [0.1, 0.2, 0.3],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.03, 0.06, 0.1],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.0],
                "soft_weight": [0.0],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # Add mild regression signal variants around strong frozen settings.
            {
                "lr": [5e-5, 7e-5, 1e-4],
                "weight_decay": [0.03, 0.05],
                "dropout": [0.1, 0.2],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.02, 0.05],
                "soft_weight": [0.0],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # Partial unfreezing for potentially better ceilings.
            {
                "lr": [5e-5, 1e-4],
                "weight_decay": [0.03, 0.05],
                "dropout": [0.1],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [1, 2],
                "encoder_lr": [2e-6, 5e-6],
                "mse_weight": [0.0, 0.05],
                "soft_weight": [0.0],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # Smaller architecture probes.
            {
                "lr": [5e-5, 1e-4],
                "weight_decay": [0.03, 0.05],
                "dropout": [0.1, 0.2],
                "hidden_dim": [256, 384],
                "projector_dim": [256, 384],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.0],
                "soft_weight": [0.0],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # Keep a narrow expanded-annotators branch for completeness.
            {
                "lr": [5e-5, 1e-4],
                "weight_decay": [0.03, 0.05],
                "dropout": [0.1, 0.2, 0.3],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.0],
                "soft_weight": [0.0],
                "use_ema": [True],
                "expand_annotators": [True],
            },
        ]
    else:  # nli
        spaces = [
            # Frozen-encoder core region (best in prior sweep).
            {
                "lr": [5e-5, 7e-5, 1e-4],
                "weight_decay": [0.02, 0.03, 0.05],
                "dropout": [0.1, 0.15, 0.2],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.03, 0.06, 0.1],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.0],
                "soft_weight": [0.0],
                "pooling": ["mean"],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # Add mild regression signal around strong settings.
            {
                "lr": [5e-5, 7e-5, 1e-4],
                "weight_decay": [0.02, 0.03],
                "dropout": [0.1],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.02, 0.05, 0.1],
                "soft_weight": [0.0],
                "pooling": ["mean"],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # cls pooling sanity branch.
            {
                "lr": [5e-5, 7e-5, 1e-4],
                "weight_decay": [0.02, 0.03],
                "dropout": [0.1, 0.2],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.0],
                "soft_weight": [0.0],
                "pooling": ["cls"],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # Partial unfreezing for potentially better ceilings.
            {
                "lr": [7e-5, 1e-4],
                "weight_decay": [0.02, 0.03],
                "dropout": [0.1],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [1, 2],
                "encoder_lr": [1e-6, 2e-6],
                "mse_weight": [0.0, 0.05],
                "soft_weight": [0.0],
                "pooling": ["mean"],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # Smaller architecture probes.
            {
                "lr": [7e-5, 1e-4],
                "weight_decay": [0.02, 0.03],
                "dropout": [0.1],
                "hidden_dim": [256, 384],
                "projector_dim": [256, 384],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.0],
                "soft_weight": [0.0],
                "pooling": ["mean"],
                "use_ema": [True],
                "expand_annotators": [False],
            },
            # Keep a narrow expanded-annotators branch for completeness.
            {
                "lr": [5e-5, 7e-5, 1e-4],
                "weight_decay": [0.02, 0.03],
                "dropout": [0.1, 0.2],
                "hidden_dim": [256],
                "projector_dim": [256],
                "warmup_ratio": [0.06],
                "train_encoder_layers": [0],
                "encoder_lr": [0.0],
                "mse_weight": [0.0],
                "soft_weight": [0.0],
                "pooling": ["mean"],
                "use_ema": [True],
                "expand_annotators": [True],
            },
        ]
    return merge_spaces(spaces)


def sample_space(space: List[Dict[str, any]], max_trials: int, seed: int) -> List[Dict[str, any]]:
    if max_trials <= 0 or len(space) <= max_trials:
        return space
    random.seed(seed)
    return random.sample(space, max_trials)


def parse_slurm_timelimit(value: str) -> Optional[int]:
    """
    Parse SLURM_TIMELIMIT formats into seconds.
    Supported examples:
      - "1440" (minutes)
      - "HH:MM:SS"
      - "MM:SS"
      - "D-HH:MM:SS"
    """
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value) * 60
    day_part = 0
    time_part = value
    if "-" in value:
        day_s, time_part = value.split("-", 1)
        if not day_s.isdigit():
            return None
        day_part = int(day_s) * 24 * 3600
    parts = time_part.split(":")
    if len(parts) == 3:
        h_s, m_s, s_s = parts
        if not (h_s.isdigit() and m_s.isdigit() and s_s.isdigit()):
            return None
        return day_part + int(h_s) * 3600 + int(m_s) * 60 + int(s_s)
    if len(parts) == 2:
        m_s, s_s = parts
        if not (m_s.isdigit() and s_s.isdigit()):
            return None
        return day_part + int(m_s) * 60 + int(s_s)
    return None


def infer_time_budget_hours(args: argparse.Namespace) -> float:
    if args.time_budget_hours > 0:
        return args.time_budget_hours
    slurm_limit = parse_slurm_timelimit(os.environ.get("SLURM_TIMELIMIT", ""))
    if slurm_limit:
        return slurm_limit / 3600.0
    return 0.0


def load_runtime_stats(path: Path) -> Dict[str, float]:
    """Load median runtime stats from previous sweep CSV, if available."""
    stats = {
        "sbert_base": 85.0,
        "sbert_expand": 160.0,
        "nli_base": 240.0,
        "nli_expand": 520.0,
    }
    if not path.exists():
        return stats
    buckets: Dict[str, List[float]] = {k: [] for k in stats}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            cfg_raw = row.get("cfg")
            sec_raw = row.get("seconds")
            if not cfg_raw or not sec_raw:
                continue
            try:
                cfg = json.loads(cfg_raw)
                secs = float(sec_raw)
            except Exception:
                continue
            expert = row.get("expert")
            expand = bool(cfg.get("expand_annotators", False))
            if expert == "sbert":
                buckets["sbert_expand" if expand else "sbert_base"].append(secs)
            elif expert == "nli":
                buckets["nli_expand" if expand else "nli_base"].append(secs)
    for k, vals in buckets.items():
        if vals:
            vals_sorted = sorted(vals)
            stats[k] = vals_sorted[len(vals_sorted) // 2]
    return stats


def estimate_trial_seconds(cfg: Dict[str, any], expert: str, stats: Dict[str, float], args: argparse.Namespace) -> float:
    expand = bool(cfg.get("expand_annotators", False))
    if expert == "sbert":
        est = stats["sbert_expand"] if expand else stats["sbert_base"]
    else:
        est = stats["nli_expand"] if expand else stats["nli_base"]

    layers = int(cfg.get("train_encoder_layers", 0))
    if layers == 1:
        est *= args.unfreeze_factor_l1
    elif layers >= 2:
        est *= args.unfreeze_factor_l2

    hidden_dim = int(cfg.get("hidden_dim", 256))
    projector_dim = int(cfg.get("projector_dim", 256))
    if hidden_dim > 256:
        est *= args.wider_head_factor
    if projector_dim > 256:
        est *= args.wider_head_factor

    return est * args.runtime_margin


def select_trials_for_budget(
    trials: List[Dict[str, any]],
    expert: str,
    budget_seconds: float,
    stats: Dict[str, float],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, any]], float]:
    selected: List[Dict[str, any]] = []
    used = 0.0
    for cfg in trials:
        est = estimate_trial_seconds(cfg, expert, stats, args)
        if used + est <= budget_seconds:
            selected.append(cfg)
            used += est
    return selected, used


def parse_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def remove_file(path_str: str) -> None:
    if not path_str:
        return
    path = Path(path_str)
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        print(f"[warn] failed to delete {path}: {exc}")


def load_best_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"[warn] failed to read best summary {path}: {exc}")
    return {}


NUM_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
METRIC_RE = re.compile(
    rf"dev epoch (?P<epoch>\d+): .*?composite=(?P<composite>{NUM_RE}).*?mae=(?P<mae>{NUM_RE}).*?spearman=(?P<spearman>{NUM_RE}).*?acc_sd=(?P<acc_sd>{NUM_RE})"
)


def extract_best_dev_line(stdout: str) -> Tuple[str, Dict[str, str]]:
    """Pick the best dev epoch line by minimum composite score."""
    dev_lines = [ln for ln in stdout.splitlines() if "dev epoch" in ln]
    best_line = ""
    best_metrics: Dict[str, str] = {}
    best_composite = float("inf")
    best_epoch = -1

    for line in dev_lines:
        match = METRIC_RE.search(line)
        if not match:
            continue
        metrics = match.groupdict()
        composite = float(metrics["composite"])
        epoch = int(metrics["epoch"])
        if composite < best_composite or (composite == best_composite and epoch > best_epoch):
            best_composite = composite
            best_epoch = epoch
            best_line = line
            best_metrics = metrics

    if best_line:
        return best_line, best_metrics

    # Fallback for partial/failed logs.
    fallback_line = dev_lines[0] if dev_lines else ""
    fallback_match = METRIC_RE.search(fallback_line)
    return fallback_line, (fallback_match.groupdict() if fallback_match else {})


def run_trial(idx: int, expert: str, cfg: Dict[str, any], base_args: argparse.Namespace, out_dir: Path) -> Dict[str, any]:
    save_path = out_dir / f"{expert}_trial{idx}.pt"
    log_path = out_dir / f"{expert}_trial{idx}.log"
    plot_path = out_dir / f"{expert}_trial{idx}.png"
    train_encoder_layers = int(cfg.get("train_encoder_layers", 0))
    encoder_lr = cfg.get("encoder_lr")
    if encoder_lr is None:
        encoder_lr = 0.0 if train_encoder_layers == 0 else base_args.encoder_lr

    cmd = [
        base_args.python,
        "-m",
        "models.train_experts",
        "--expert",
        expert,
        "--model-name",
        base_args.model_name if expert == "sbert" else base_args.nli_model_name,
        "--train-encoder-layers",
        str(train_encoder_layers),
        "--lr",
        str(cfg["lr"]),
        "--encoder-lr",
        str(encoder_lr),
        "--weight-decay",
        str(cfg["weight_decay"]),
        "--dropout",
        str(cfg["dropout"]),
        "--hidden-dim",
        str(cfg["hidden_dim"]),
        "--projector-dim",
        str(cfg["projector_dim"]),
        "--warmup-ratio",
        str(cfg["warmup_ratio"]),
        "--early-stop-patience",
        str(base_args.early_stop_patience),
        "--coral-weight",
        "1.0",
        "--mse-weight",
        str(cfg["mse_weight"]),
        "--soft-weight",
        str(cfg["soft_weight"]),
        "--batch-size",
        str(base_args.batch_size_nli if expert == "nli" else base_args.batch_size_sbert),
        "--epochs",
        str(base_args.epochs),
        "--save-path",
        str(save_path),
        "--plot-path",
        str(plot_path),
        "--no-plot",
    ]
    if cfg.get("use_ema", True):
        cmd += ["--use-ema"]
    if cfg.get("expand_annotators", False):
        cmd += ["--expand-annotators"]
    if expert == "nli":
        cmd += ["--pooling", cfg["pooling"]]
    env = os.environ.copy()
    if base_args.pythonpath:
        env["PYTHONPATH"] = base_args.pythonpath
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    duration = time.time() - start
    dev_line, metrics = extract_best_dev_line(result.stdout)
    status = "ok" if result.returncode == 0 else f"err_{result.returncode}"
    log_file = ""
    keep_logs_mode = base_args.keep_logs
    should_write_log = keep_logs_mode != "none" and not (keep_logs_mode == "errors" and status == "ok")
    if should_write_log:
        tail_n = max(1, int(base_args.log_tail_lines))
        stdout_tail = result.stdout.splitlines()[-tail_n:]
        stderr_tail = result.stderr.splitlines()[-tail_n:]
        summary_lines = [
            f"cmd: {shlex.join(cmd)}",
            f"status: {status}",
            f"seconds: {duration:.1f}",
            f"best_dev_line: {dev_line}",
            "",
            "stdout_tail:",
            *stdout_tail,
            "",
            "stderr_tail:",
            *stderr_tail,
        ]
        try:
            log_path.write_text("\n".join(summary_lines))
            log_file = str(log_path)
        except OSError as exc:
            print(f"[warn] could not write log {log_path}: {exc}")
    return {
        "trial": idx,
        "expert": expert,
        "cfg": json.dumps(cfg),
        "dev_line": dev_line,
        "composite": metrics.get("composite", ""),
        "mae": metrics.get("mae", ""),
        "spearman": metrics.get("spearman", ""),
        "acc_sd": metrics.get("acc_sd", ""),
        "status": status,
        "seconds": round(duration, 1),
        "log": log_file,
        "ckpt": str(save_path) if save_path.exists() else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for NLI/SBERT experts.")
    parser.add_argument("--experts", nargs="+", default=["sbert", "nli"], choices=["sbert", "nli"])
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Max trials per expert (random subset of grid). Use 0 to run all configs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/sweep"))
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--pythonpath", type=str, default="src")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--nli-model-name", type=str, default="roberta-large-mnli")
    parser.add_argument("--batch-size-sbert", type=int, default=8)
    parser.add_argument("--batch-size-nli", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=0.0,
        help="Total walltime budget in hours. 0 means infer from SLURM_TIMELIMIT or disable if unavailable.",
    )
    parser.add_argument(
        "--safety-minutes",
        type=float,
        default=30.0,
        help="Buffer to leave unused before the scheduler time limit.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("runs/sweep/results.csv"),
        help="CSV used to estimate trial runtime for budget planning.",
    )
    parser.add_argument(
        "--runtime-margin",
        type=float,
        default=1.2,
        help="Conservative multiplier on runtime estimates.",
    )
    parser.add_argument(
        "--unfreeze-factor-l1",
        type=float,
        default=1.35,
        help="Runtime multiplier when train_encoder_layers=1.",
    )
    parser.add_argument(
        "--unfreeze-factor-l2",
        type=float,
        default=1.6,
        help="Runtime multiplier when train_encoder_layers>=2.",
    )
    parser.add_argument(
        "--wider-head-factor",
        type=float,
        default=1.12,
        help="Runtime multiplier per widened hidden/projector dimension (>256).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print planned trial counts/runtime and exit without running trials.",
    )
    parser.add_argument(
        "--keep-best-only",
        dest="keep_best_only",
        action="store_true",
        default=True,
        help="Keep only the best checkpoint per expert (overwrite best_<expert>.pt).",
    )
    parser.add_argument(
        "--keep-all-checkpoints",
        dest="keep_best_only",
        action="store_false",
        help="Keep every trial checkpoint file.",
    )
    parser.add_argument(
        "--keep-logs",
        choices=["best", "errors", "all", "none"],
        default="best",
        help="Log retention policy: keep best+errors, only errors, all, or none.",
    )
    parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=120,
        help="Number of stdout/stderr tail lines to store per trial log.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_summary_path = args.out_dir / "best_summary.json"
    persisted_best = load_best_summary(best_summary_path)
    rows: List[Dict[str, any]] = []
    runtime_stats = load_runtime_stats(args.history_csv)

    budget_hours = infer_time_budget_hours(args)
    budget_enabled = budget_hours > 0
    hard_budget_seconds = max(0.0, budget_hours * 3600.0 - args.safety_minutes * 60.0) if budget_enabled else 0.0

    planned_trials: Dict[str, List[Dict[str, any]]] = {}
    planned_estimates: Dict[str, float] = {}
    estimated_totals: Dict[str, float] = {}

    for expert in args.experts:
        space = build_space(expert)
        trials = sample_space(space, args.max_trials, args.seed)
        planned_trials[expert] = trials
        estimated_totals[expert] = sum(estimate_trial_seconds(cfg, expert, runtime_stats, args) for cfg in trials)

    if budget_enabled:
        total_estimated = sum(estimated_totals.values())
        if total_estimated <= 0:
            total_estimated = 1.0
        for expert in args.experts:
            expert_budget = hard_budget_seconds * (estimated_totals[expert] / total_estimated)
            selected, used = select_trials_for_budget(
                planned_trials[expert], expert, expert_budget, runtime_stats, args
            )
            planned_trials[expert] = selected
            planned_estimates[expert] = used
        total_planned_est = sum(planned_estimates.values())
        print(
            f"Budget planning enabled: {budget_hours:.2f}h total, {args.safety_minutes:.1f}m safety, "
            f"usable={hard_budget_seconds/3600.0:.2f}h, estimated planned={total_planned_est/3600.0:.2f}h"
        )
        for expert in args.experts:
            print(
                f"  {expert}: planned {len(planned_trials[expert])} trials, "
                f"estimated {planned_estimates.get(expert, 0.0)/3600.0:.2f}h"
            )
    else:
        for expert in args.experts:
            planned_estimates[expert] = estimated_totals[expert]
        total_planned_est = sum(planned_estimates.values())
        print(
            f"Budget planning disabled. Planned {sum(len(planned_trials[e]) for e in args.experts)} trials, "
            f"estimated {total_planned_est/3600.0:.2f}h"
        )
        for expert in args.experts:
            print(
                f"  {expert}: planned {len(planned_trials[expert])} trials, "
                f"estimated {planned_estimates.get(expert, 0.0)/3600.0:.2f}h"
            )

    if args.plan_only:
        print("Plan-only mode: exiting before running trials.")
        return

    best_state: Dict[str, Dict[str, Any]] = {}
    for expert in args.experts:
        entry = persisted_best.get(expert, {}) if isinstance(persisted_best, dict) else {}
        prior_score = parse_float(entry.get("score"), default=float("inf"))
        prior_trial = entry.get("trial")
        prior_log = entry.get("log", "")
        prior_ckpt = args.out_dir / f"best_{expert}.pt"
        if not prior_ckpt.exists():
            prior_score = float("inf")
            prior_trial = None
            prior_log = ""
        best_state[expert] = {
            "score": prior_score,
            "trial": prior_trial,
            "ckpt": str(prior_ckpt) if prior_ckpt.exists() else "",
            "log": prior_log,
        }

    sweep_start = time.time()
    for expert in args.experts:
        trials = planned_trials[expert]
        for i, cfg in enumerate(trials):
            if budget_enabled:
                elapsed = time.time() - sweep_start
                remaining = hard_budget_seconds - elapsed
                est_next = estimate_trial_seconds(cfg, expert, runtime_stats, args)
                if remaining < est_next:
                    print(
                        f"[stop-budget] {expert} trial {i} skipped: "
                        f"remaining {remaining/60.0:.1f}m < est_next {est_next/60.0:.1f}m"
                    )
                    break
            rec = run_trial(i, expert, cfg, args, args.out_dir)
            score = parse_float(rec.get("composite"))
            is_ok = rec.get("status") == "ok"
            is_better = is_ok and score < best_state[expert]["score"]

            old_best_log = best_state[expert]["log"]

            # Keep only best checkpoint per expert if requested.
            ckpt_file = rec.get("ckpt", "")
            if args.keep_best_only and ckpt_file:
                ckpt_path = Path(ckpt_file)
                if ckpt_path.exists():
                    if is_better:
                        best_ckpt_path = args.out_dir / f"best_{expert}.pt"
                        try:
                            ckpt_path.replace(best_ckpt_path)
                            rec["ckpt"] = str(best_ckpt_path)
                        except OSError as exc:
                            print(f"[warn] failed to move checkpoint {ckpt_path} -> {best_ckpt_path}: {exc}")
                    else:
                        remove_file(ckpt_file)
                        rec["ckpt"] = ""

            # Log retention policy.
            log_file = rec.get("log", "")
            if args.keep_logs == "none":
                remove_file(log_file)
                rec["log"] = ""
            elif args.keep_logs == "errors":
                if is_ok:
                    remove_file(log_file)
                    rec["log"] = ""
            elif args.keep_logs == "best":
                if is_ok and not is_better:
                    remove_file(log_file)
                    rec["log"] = ""
                elif is_ok and is_better and old_best_log and old_best_log != log_file:
                    remove_file(old_best_log)

            # Update best tracker.
            if is_better:
                best_state[expert]["score"] = score
                best_state[expert]["trial"] = rec.get("trial")
                best_state[expert]["ckpt"] = rec.get("ckpt", "")
                best_state[expert]["log"] = rec.get("log", "")

            rows.append(rec)
            print(f"[{rec['status']}] {expert} trial {i}: {rec['dev_line']}")

    for expert in args.experts:
        best = best_state[expert]
        if best["trial"] is None:
            print(f"[best] {expert}: no successful trial.")
            continue
        print(
            f"[best] {expert}: trial={best['trial']}, composite={best['score']:.4f}, "
            f"ckpt={best['ckpt'] or '<not kept>'}"
        )

    best_summary_payload = {
        expert: {
            "score": best_state[expert]["score"],
            "trial": best_state[expert]["trial"],
            "ckpt": best_state[expert]["ckpt"],
            "log": best_state[expert]["log"],
        }
        for expert in args.experts
    }
    try:
        best_summary_path.write_text(json.dumps(best_summary_payload, indent=2))
    except OSError as exc:
        print(f"[warn] failed to write best summary {best_summary_path}: {exc}")

    csv_path = args.out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "expert",
                "cfg",
                "composite",
                "mae",
                "spearman",
                "acc_sd",
                "dev_line",
                "status",
                "seconds",
                "log",
                "ckpt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done. Results written to {csv_path}")


if __name__ == "__main__":
    main()
