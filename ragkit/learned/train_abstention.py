#!/usr/bin/env python3
"""Train and evaluate the learned abstention classifier on SQuAD 2.0.

Runs the full experimental protocol from the design doc:
  - Sample N articles with a fixed seed, split by article 60/15/25.
  - Within each partition, split by article into indexed / withheld.
  - Build retrieval indexes over indexed articles only.
  - Retrieve each question exactly once against its partition's index and
    cache the 8 features (results/cache/abstention_cache.json).
  - Train a from-scratch logistic regression with L2 grid + epoch selection
    on the calibration partition, then Platt calibration on calibration.
  - Compare with a tuned threshold baseline on the test partition, with
    cluster bootstrap CIs (resampling at the article level, learned and
    baseline paired on the same resample).
  - Run feature ablation, secondary adversarial variants A and B, and the
    two sanity checks (null check + leakage diagnostic).
  - Save models/abstention_lr.json, results/abstention_results.json,
    results/abstention_report.md.

Run from the repo root:
    python -m ragkit.learned.train_abstention --max-articles 150 --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

from ragkit.learned.features import (
    FEATURE_COLUMNS,
    FEATURE_GROUP_COLUMNS,
    assign_indexed_withheld,
    build_or_load_cache,
    split_articles,
)
from ragkit.learned import logreg as LR


SQUAD_DEFAULT = "datasets/squad/train-v2.0.json"
MODEL_PATH = Path("models/abstention_lr.json")
RESULTS_JSON = Path("results/abstention_results.json")
REPORT_MD = Path("results/abstention_report.md")
CACHE_PATH = Path("results/cache/abstention_cache.json")

PARTITIONS = ("train", "calibration", "test")


# ---------------------------------------------------------------------------
# CLI and I/O
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Train and evaluate the learned abstention classifier."
    )
    p.add_argument("--max-articles", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=0,
                   help="0 = full-batch GD; >0 = mini-batch SGD")
    p.add_argument("--log-every", type=int, default=10,
                   help="checkpoint + logging interval in epochs")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--data-path", default=SQUAD_DEFAULT)
    return p.parse_args(argv)


def _load_squad(path):
    path = Path(path)
    if not path.exists():
        sys.exit(
            f"SQuAD file not found at {path}. "
            f"Run: python datasets/download_squad.py"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Could not parse {path}: {exc}")


# ---------------------------------------------------------------------------
# Splits and per-partition statistics
# ---------------------------------------------------------------------------

def _assert_disjoint(split):
    s = set(split["train"])
    c = set(split["calibration"])
    t = set(split["test"])
    if s & c:
        raise AssertionError("train / calibration article overlap")
    if s & t:
        raise AssertionError("train / test article overlap")
    if c & t:
        raise AssertionError("calibration / test article overlap")


def _print_partition_stats(squad, split, indexed_map, withheld_map):
    print("Partition sizes:")
    for name in PARTITIONS:
        n_chunks = sum(
            len(squad["data"][i]["paragraphs"]) for i in indexed_map[name]
        )
        print(
            f"  {name:<12s}: {len(split[name]):>3d} articles "
            f"({len(indexed_map[name])} indexed, {len(withheld_map[name])} withheld), "
            f"{n_chunks} chunks in index"
        )


# ---------------------------------------------------------------------------
# Primary / secondary datasets
# ---------------------------------------------------------------------------

def _filter_primary(records):
    """Primary task labels.

    label 1: indexed=True and is_impossible=False (answerable, in-corpus).
    label 0: indexed=False (article is withheld -> should abstain).
    Indexed + is_impossible questions are EXCLUDED from primary.
    """
    out = []
    for r in records:
        if r["indexed"] and r["is_impossible"]:
            continue
        out.append((r, 1 if r["indexed"] else 0))
    return out


def _filter_secondary(records):
    """Secondary task labels (hard / adversarial).

    Only indexed-article questions:
      label 1: is_impossible=False (answerable).
      label 0: is_impossible=True.
    """
    out = []
    for r in records:
        if not r["indexed"]:
            continue
        out.append((r, 0 if r["is_impossible"] else 1))
    return out


def _by_partition(pairs):
    out = {name: [] for name in PARTITIONS}
    for r, y in pairs:
        out[r["partition"]].append((r, y))
    return out


def _partition_arrays(pairs):
    if not pairs:
        return [], [], []
    labels = [y for _, y in pairs]
    X = [list(r["features"]) for r, _ in pairs]
    art_ids = [r["article_idx"] for r, _ in pairs]
    return X, labels, art_ids


def _prevalence(labels):
    n = len(labels)
    if n == 0:
        return {
            "n": 0, "n_pos": 0, "n_neg": 0,
            "prevalence": 0.0, "majority_accuracy": 0.0,
        }
    n_pos = sum(labels)
    n_neg = n - n_pos
    return {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "prevalence": n_pos / n,
        "majority_accuracy": max(n_pos, n_neg) / n,
    }


# ---------------------------------------------------------------------------
# Training and prediction helpers
# ---------------------------------------------------------------------------

def _train_pipeline(X_tr_raw, y_tr, X_cal_raw, y_cal, args):
    """Standardize with train stats, select (L2, epoch) on calibration,
    fit Platt on calibration, return the full model bundle."""
    means, stds = LR.compute_standardization(X_tr_raw)
    X_tr = LR.standardize(X_tr_raw, means, stds)
    X_cal = LR.standardize(X_cal_raw, means, stds)
    per_l2, best = LR.select_hyperparams(
        X_tr, y_tr, X_cal, y_cal,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, log_every=args.log_every,
    )
    best_loss, best_l2, w, b, best_epoch = best
    cal_logits = [sum(wj * xj for wj, xj in zip(w, x)) + b for x in X_cal]
    a_platt, b_platt = LR.fit_platt(cal_logits, y_cal)
    return {
        "means": means, "stds": stds,
        "per_l2": per_l2, "best_loss": best_loss,
        "best_l2": best_l2, "best_epoch": best_epoch,
        "w": w, "b": b,
        "platt_a": a_platt, "platt_b": b_platt,
    }


def _predict_learned(model, X_raw):
    X = LR.standardize(X_raw, model["means"], model["stds"])
    logits = [
        sum(wj * xj for wj, xj in zip(model["w"], x)) + model["b"]
        for x in X
    ]
    probs = LR.apply_platt(logits, model["platt_a"], model["platt_b"])
    return probs


def _fit_baseline_platt(f4_cal, y_cal, mean, std):
    if std == 0.0:
        std = 1.0
    return LR.fit_platt([(f - mean) / std for f in f4_cal], y_cal)


def _baseline_probs(f4, a, b, mean, std):
    if std == 0.0:
        std = 1.0
    return LR.apply_platt([(f - mean) / std for f in f4], a, b)


# ---------------------------------------------------------------------------
# Metrics + CI wrapper
# ---------------------------------------------------------------------------

def _test_metrics(labels, scores, probs):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return {
        "n": len(labels),
        "n_pos": sum(labels),
        "n_neg": len(labels) - sum(labels),
        "accuracy": LR.accuracy(labels, preds),
        "auroc": LR.auroc(labels, scores),
        "brier": LR.brier(labels, probs),
        "ece": LR.ece(labels, probs, n_bins=10),
        "preds": preds,
    }


def _bootstrap_ci(labels, scores, preds, article_ids, seed):
    acc_lo, acc_hi, _ = LR.cluster_bootstrap_accuracy(
        labels, preds, article_ids, n_resamples=1000, seed=seed,
    )
    auc_lo, auc_hi, auc_sk = LR.cluster_bootstrap_auroc(
        labels, scores, article_ids, n_resamples=1000, seed=seed,
    )
    return {
        "acc_ci": [acc_lo, acc_hi],
        "auroc_ci": [auc_lo, auc_hi],
        "auroc_ci_skipped": auc_sk,
    }


def _fmt(v, digits=4):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _fmt_ci(lo, hi, digits=4):
    if lo is None or hi is None:
        return "N/A"
    return f"[{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    if args.epochs < args.log_every:
        sys.exit(
            f"--epochs ({args.epochs}) must be >= --log-every ({args.log_every})"
        )

    t_start = time.time()
    print(f"[config] seed={args.seed} max_articles={args.max_articles} "
          f"lr={args.lr} epochs={args.epochs} batch_size={args.batch_size} "
          f"log_every={args.log_every} top_k={args.top_k}")

    squad = _load_squad(args.data_path)
    num_articles = len(squad["data"])
    split = split_articles(num_articles, args.max_articles, args.seed)
    indexed_map, withheld_map = {}, {}
    for name in PARTITIONS:
        indexed_map[name], withheld_map[name] = assign_indexed_withheld(
            split[name], args.seed, name,
        )
    _assert_disjoint(split)
    _print_partition_stats(squad, split, indexed_map, withheld_map)

    # ---- Retrieval (once per question, cached) ----
    cache = build_or_load_cache(
        squad, split, indexed_map, withheld_map,
        top_k=args.top_k, seed=args.seed, max_articles=args.max_articles,
        cache_path=CACHE_PATH, rebuild=args.rebuild_cache,
    )
    retrieval_seconds = time.time() - t_start
    records = cache["records"]

    # ---- Primary dataset ----
    primary_by = _by_partition(_filter_primary(records))
    primary_data = {}
    for name in PARTITIONS:
        X_raw, y, art_ids = _partition_arrays(primary_by[name])
        primary_data[name] = {"X_raw": X_raw, "y": y, "article_ids": art_ids}

    # ---- Primary model ----
    print("[train] fitting primary model (L2 × epoch grid on calibration)")
    model_primary = _train_pipeline(
        primary_data["train"]["X_raw"], primary_data["train"]["y"],
        primary_data["calibration"]["X_raw"], primary_data["calibration"]["y"],
        args,
    )
    print(f"[train] selected L2={model_primary['best_l2']:g} "
          f"epochs={model_primary['best_epoch']} "
          f"cal_loss={model_primary['best_loss']:.6f}")
    print(f"[train] weights={[round(w, 4) for w in model_primary['w']]} "
          f"bias={model_primary['b']:.4f}")

    # ---- Test (learned) ----
    test_y = primary_data["test"]["y"]
    test_ids = primary_data["test"]["article_ids"]
    test_probs = _predict_learned(model_primary, primary_data["test"]["X_raw"])
    learned_metrics = _test_metrics(test_y, test_probs, test_probs)
    learned_ci = _bootstrap_ci(
        test_y, test_probs, learned_metrics["preds"], test_ids, args.seed,
    )

    # ---- Baseline on primary ----
    f4_train = [X[3] for X in primary_data["train"]["X_raw"]]
    f4_cal = [X[3] for X in primary_data["calibration"]["X_raw"]]
    f4_test = [X[3] for X in primary_data["test"]["X_raw"]]
    baseline_t, baseline_train_acc = LR.tune_threshold(
        f4_train, primary_data["train"]["y"],
    )
    print(f"[baseline] tuned on TRAIN: t={baseline_t:.6f} "
          f"train_acc={baseline_train_acc:.4f}")
    b_platt_a, b_platt_b = _fit_baseline_platt(
        f4_cal, primary_data["calibration"]["y"],
        model_primary["means"][3], model_primary["stds"][3],
    )
    baseline_test_probs = _baseline_probs(
        f4_test, b_platt_a, b_platt_b,
        model_primary["means"][3], model_primary["stds"][3],
    )
    baseline_test_preds = [1 if f >= baseline_t else 0 for f in f4_test]
    baseline_metrics = _test_metrics(test_y, f4_test, baseline_test_probs)
    baseline_metrics["preds"] = baseline_test_preds
    baseline_metrics["accuracy"] = LR.accuracy(test_y, baseline_test_preds)
    baseline_ci = _bootstrap_ci(
        test_y, f4_test, baseline_test_preds, test_ids, args.seed,
    )

    # Repo default threshold (frozen reference)
    b005_preds = [1 if f >= 0.05 else 0 for f in f4_test]
    b005_accuracy = LR.accuracy(test_y, b005_preds)
    b005_ci = _bootstrap_ci(
        test_y, f4_test, b005_preds, test_ids, args.seed,
    )

    # Paired AUROC difference (same resamples)
    diff_lo, diff_hi, diff_sk = LR.cluster_bootstrap_auroc_diff(
        test_y, test_probs, f4_test, test_ids,
        n_resamples=1000, seed=args.seed,
    )
    excludes_zero = None
    if diff_lo is not None and diff_hi is not None:
        excludes_zero = (diff_lo > 0.0) or (diff_hi < 0.0)
    print(f"[primary test] learned AUROC={_fmt(learned_metrics['auroc'])} "
          f"baseline AUROC={_fmt(baseline_metrics['auroc'])} "
          f"diff CI={_fmt_ci(diff_lo, diff_hi)} "
          f"excludes_zero={excludes_zero}")

    # ---- Precision-coverage (test, primary) ----
    pc_learned_table, pc_learned_full = LR.precision_coverage(test_y, test_probs)
    pc_baseline_table, pc_baseline_full = LR.precision_coverage(test_y, f4_test)

    # ---- Feature ablation ----
    print("[ablation] running feature ablation")
    X_tr_std = LR.standardize(
        primary_data["train"]["X_raw"], model_primary["means"], model_primary["stds"],
    )
    X_te_std = LR.standardize(
        primary_data["test"]["X_raw"], model_primary["means"], model_primary["stds"],
    )
    ablation = LR.ablation_study(
        X_tr_std, primary_data["train"]["y"],
        X_te_std, test_y,
        FEATURE_GROUP_COLUMNS,
        l2=model_primary["best_l2"], lr=args.lr,
        epochs=model_primary["best_epoch"],
        batch_size=args.batch_size, seed=args.seed,
    )

    # ---- Secondary task ----
    print("[secondary] adversarial variants A and B")
    secondary_by = _by_partition(_filter_secondary(records))
    secondary_data = {}
    for name in PARTITIONS:
        X_raw, y, art_ids = _partition_arrays(secondary_by[name])
        secondary_data[name] = {"X_raw": X_raw, "y": y, "article_ids": art_ids}
    secondary_results = _run_secondary(secondary_data, model_primary, baseline_t, args)

    # ---- Sanity checks ----
    print("[sanity] null check (both train + test labels shuffled)")
    null_check = LR.evaluation_null_check(
        X_tr_std, primary_data["train"]["y"], X_te_std, test_y,
        l2=model_primary["best_l2"], lr=args.lr,
        epochs=model_primary["best_epoch"],
        batch_size=args.batch_size, n_shuffles=20, seed=args.seed,
    )
    print(f"[sanity] null mean={_fmt(null_check['mean'])} "
          f"std={_fmt(null_check['std'])} warn={null_check['warn']}")

    print("[sanity] feature leakage diagnostic (train labels shuffled only)")
    diagnostic = LR.feature_leakage_diagnostic(
        X_tr_std, primary_data["train"]["y"], X_te_std, test_y,
        l2=model_primary["best_l2"], lr=args.lr,
        epochs=model_primary["best_epoch"],
        batch_size=args.batch_size, n_shuffles=20, seed=args.seed,
    )
    print(f"[sanity] diagnostic mean={_fmt(diagnostic['mean'])} "
          f"std={_fmt(diagnostic['std'])} "
          f"min={_fmt(diagnostic['min'])} max={_fmt(diagnostic['max'])}")

    # ---- Feature shift report ----
    feature_means_per_partition = {}
    for name in PARTITIONS:
        X = primary_data[name]["X_raw"]
        feature_means_per_partition[name] = (
            LR.feature_means(X) if X else None
        )

    # ---- Partition stats ----
    partition_stats = {}
    for name in PARTITIONS:
        n_chunks = sum(
            len(squad["data"][i]["paragraphs"]) for i in indexed_map[name]
        )
        partition_stats[name] = {
            "num_articles": len(split[name]),
            "indexed_articles": len(indexed_map[name]),
            "withheld_articles": len(withheld_map[name]),
            "index_chunks": n_chunks,
            "primary": _prevalence(primary_data[name]["y"]),
            "secondary": _prevalence(secondary_data[name]["y"]),
        }

    # ---- Save ----
    t_end = time.time()
    training_seconds = t_end - t_start - retrieval_seconds

    results = {
        "config": {
            "seed": args.seed,
            "max_articles": args.max_articles,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "log_every": args.log_every,
            "top_k": args.top_k,
            "data_path": str(args.data_path),
        },
        "partition_stats": partition_stats,
        "split_article_ids": {n: list(split[n]) for n in PARTITIONS},
        "indexed_article_ids": {n: list(indexed_map[n]) for n in PARTITIONS},
        "withheld_article_ids": {n: list(withheld_map[n]) for n in PARTITIONS},
        "hyperparameter_grid": {
            f"{l2:g}": {
                "best_cal_loss": v["best_loss"],
                "best_epoch": v["best_epoch"],
            }
            for l2, v in model_primary["per_l2"].items()
        },
        "selected_model": {
            "l2": model_primary["best_l2"],
            "epochs": model_primary["best_epoch"],
            "cal_loss": model_primary["best_loss"],
            "weights": model_primary["w"],
            "bias": model_primary["b"],
            "means": model_primary["means"],
            "stds": model_primary["stds"],
            "platt_a": model_primary["platt_a"],
            "platt_b": model_primary["platt_b"],
        },
        "baseline_primary": {
            "threshold": baseline_t,
            "train_accuracy": baseline_train_acc,
            "platt_a": b_platt_a,
            "platt_b": b_platt_b,
        },
        "test_primary": {
            "learned": {
                "accuracy": learned_metrics["accuracy"],
                "accuracy_ci": learned_ci["acc_ci"],
                "auroc": learned_metrics["auroc"],
                "auroc_ci": learned_ci["auroc_ci"],
                "auroc_ci_skipped": learned_ci["auroc_ci_skipped"],
                "ece": learned_metrics["ece"],
                "brier": learned_metrics["brier"],
                "n": learned_metrics["n"],
                "n_pos": learned_metrics["n_pos"],
                "n_neg": learned_metrics["n_neg"],
            },
            "baseline_tuned": {
                "accuracy": baseline_metrics["accuracy"],
                "accuracy_ci": baseline_ci["acc_ci"],
                "auroc": baseline_metrics["auroc"],
                "auroc_ci": baseline_ci["auroc_ci"],
                "auroc_ci_skipped": baseline_ci["auroc_ci_skipped"],
                "ece": baseline_metrics["ece"],
                "brier": baseline_metrics["brier"],
                "threshold": baseline_t,
            },
            "baseline_at_0_05": {
                "accuracy": b005_accuracy,
                "accuracy_ci": b005_ci["acc_ci"],
                "auroc": baseline_metrics["auroc"],
                "auroc_ci": baseline_ci["auroc_ci"],
                "ece": baseline_metrics["ece"],
                "brier": baseline_metrics["brier"],
                "threshold": 0.05,
            },
            "auroc_diff_learned_minus_baseline": {
                "ci_low": diff_lo,
                "ci_high": diff_hi,
                "n_skipped_resamples": diff_sk,
                "excludes_zero": excludes_zero,
            },
            "precision_coverage": {
                "learned_table": pc_learned_table,
                "learned_full": pc_learned_full,
                "baseline_table": pc_baseline_table,
                "baseline_full": pc_baseline_full,
            },
        },
        "ablation": ablation,
        "secondary": secondary_results,
        "sanity_check_null": null_check,
        "sanity_check_diagnostic": diagnostic,
        "feature_means_per_partition": feature_means_per_partition,
        "feature_columns": FEATURE_COLUMNS,
        "runtime_seconds": {
            "retrieval": retrieval_seconds,
            "training_eval": training_seconds,
            "total": t_end - t_start,
        },
    }

    _save_model(
        model_primary, args, split, indexed_map, withheld_map,
        partition_stats, baseline_t, baseline_train_acc,
    )
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(_render_report(results))
    print(f"\nSaved:\n  {MODEL_PATH}\n  {RESULTS_JSON}\n  {REPORT_MD}")
    print(f"[runtime] retrieval={retrieval_seconds:.1f}s "
          f"training+eval={training_seconds:.1f}s "
          f"total={t_end - t_start:.1f}s")


def _run_secondary(secondary_data, model_primary, baseline_t, args):
    """Variants A and B; returns a results dict (either may be None)."""
    out = {
        "partition_stats": {
            name: _prevalence(secondary_data[name]["y"])
            for name in PARTITIONS
        },
    }
    test_y = secondary_data["test"]["y"]
    test_ids = secondary_data["test"]["article_ids"]
    if not test_y or len(set(test_y)) < 2:
        out["variant_A"] = None
        out["variant_B"] = None
        return out

    f4_test = [X[3] for X in secondary_data["test"]["X_raw"]]

    # Variant A: primary-trained model + primary-frozen baseline
    probs_a = _predict_learned(model_primary, secondary_data["test"]["X_raw"])
    preds_a = [1 if p >= 0.5 else 0 for p in probs_a]
    base_a_preds = [1 if f >= baseline_t else 0 for f in f4_test]
    a_lo, a_hi, a_sk = LR.cluster_bootstrap_auroc(
        test_y, probs_a, test_ids, n_resamples=1000, seed=args.seed,
    )
    out["variant_A"] = {
        "learned_auroc": LR.auroc(test_y, probs_a),
        "learned_auroc_ci": [a_lo, a_hi],
        "learned_auroc_ci_skipped": a_sk,
        "learned_accuracy": LR.accuracy(test_y, preds_a),
        "baseline_auroc": LR.auroc(test_y, f4_test),
        "baseline_accuracy": LR.accuracy(test_y, base_a_preds),
        "baseline_threshold": baseline_t,
        "n": len(test_y),
        "n_pos": sum(test_y),
        "n_neg": len(test_y) - sum(test_y),
    }

    # Variant B: secondary-trained model + secondary-tuned baseline
    try:
        if (len(secondary_data["train"]["y"]) < 10
                or len(set(secondary_data["train"]["y"])) < 2
                or len(set(secondary_data["calibration"]["y"])) < 2):
            out["variant_B"] = None
            return out
        model_b = _train_pipeline(
            secondary_data["train"]["X_raw"], secondary_data["train"]["y"],
            secondary_data["calibration"]["X_raw"], secondary_data["calibration"]["y"],
            args,
        )
        probs_b = _predict_learned(model_b, secondary_data["test"]["X_raw"])
        preds_b = [1 if p >= 0.5 else 0 for p in probs_b]
        f4_tr = [X[3] for X in secondary_data["train"]["X_raw"]]
        b_t_b, b_acc_b = LR.tune_threshold(f4_tr, secondary_data["train"]["y"])
        base_b_preds = [1 if f >= b_t_b else 0 for f in f4_test]
        b_lo, b_hi, b_sk = LR.cluster_bootstrap_auroc(
            test_y, probs_b, test_ids, n_resamples=1000, seed=args.seed,
        )
        out["variant_B"] = {
            "selected_l2": model_b["best_l2"],
            "selected_epochs": model_b["best_epoch"],
            "learned_auroc": LR.auroc(test_y, probs_b),
            "learned_auroc_ci": [b_lo, b_hi],
            "learned_auroc_ci_skipped": b_sk,
            "learned_accuracy": LR.accuracy(test_y, preds_b),
            "baseline_auroc": LR.auroc(test_y, f4_test),
            "baseline_accuracy": LR.accuracy(test_y, base_b_preds),
            "baseline_threshold": b_t_b,
            "baseline_train_accuracy": b_acc_b,
            "n": len(test_y),
            "n_pos": sum(test_y),
            "n_neg": len(test_y) - sum(test_y),
        }
    except Exception as exc:
        out["variant_B"] = {"error": str(exc)}
    return out


def _save_model(model, args, split, indexed_map, withheld_map,
                partition_stats, baseline_t, baseline_train_acc):
    payload = {
        "feature_columns": FEATURE_COLUMNS,
        "weights": model["w"],
        "bias": model["b"],
        "means": model["means"],
        "stds": model["stds"],
        "platt_a": model["platt_a"],
        "platt_b": model["platt_b"],
        "chosen_l2": model["best_l2"],
        "chosen_epochs": model["best_epoch"],
        "baseline_threshold": baseline_t,
        "baseline_train_accuracy": baseline_train_acc,
        "seed": args.seed,
        "config": {
            "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "log_every": args.log_every,
            "top_k": args.top_k, "max_articles": args.max_articles,
        },
        "data_sample_size": args.max_articles,
        "squad_article_ids_per_partition": {
            n: list(split[n]) for n in PARTITIONS
        },
        "indexed_articles_per_partition": {
            n: list(indexed_map[n]) for n in PARTITIONS
        },
        "withheld_articles_per_partition": {
            n: list(withheld_map[n]) for n in PARTITIONS
        },
        "index_sizes": {
            n: partition_stats[n]["index_chunks"] for n in PARTITIONS
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Report rendering (all values pulled from R; none hardcoded)
# ---------------------------------------------------------------------------

def _render_report(R):
    cfg = R["config"]
    L = []
    L.append("# Learned Abstention Classifier — SQuAD 2.0 Evaluation Report\n")

    L.append("## Configuration and seed\n")
    L.append(f"- seed: `{cfg['seed']}`")
    L.append(f"- max_articles (N): `{cfg['max_articles']}`")
    L.append(f"- lr: `{cfg['lr']}`")
    L.append(f"- max epochs: `{cfg['epochs']}`")
    L.append(f"- batch_size: `{cfg['batch_size']}` (0 = full batch)")
    L.append(f"- log_every: `{cfg['log_every']}`")
    L.append(f"- top_k: `{cfg['top_k']}`")
    L.append(f"- SQuAD file: `{cfg['data_path']}`\n")

    L.append("## Partition sizes (articles and index chunks)\n")
    L.append("| Partition | Articles | Indexed | Withheld | Index chunks |")
    L.append("|---|---:|---:|---:|---:|")
    for name in PARTITIONS:
        s = R["partition_stats"][name]
        L.append(
            f"| {name} | {s['num_articles']} | {s['indexed_articles']} | "
            f"{s['withheld_articles']} | {s['index_chunks']} |"
        )
    L.append("")

    L.append("## Prevalence and majority-class accuracy\n")
    L.append("### Primary task (indexed-answerable vs withheld-any)\n")
    L.append("| Partition | N | N+ | N- | Prevalence | Majority acc |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for name in PARTITIONS:
        p = R["partition_stats"][name]["primary"]
        L.append(
            f"| {name} | {p['n']} | {p['n_pos']} | {p['n_neg']} | "
            f"{_fmt(p['prevalence'])} | {_fmt(p['majority_accuracy'])} |"
        )
    L.append("")
    L.append("### Secondary task (indexed-answerable vs indexed-impossible)\n")
    L.append("| Partition | N | N+ | N- | Prevalence | Majority acc |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for name in PARTITIONS:
        p = R["partition_stats"][name]["secondary"]
        L.append(
            f"| {name} | {p['n']} | {p['n_pos']} | {p['n_neg']} | "
            f"{_fmt(p['prevalence'])} | {_fmt(p['majority_accuracy'])} |"
        )
    L.append("")

    L.append("## Hyperparameter grid (L2 × epoch, selected on CALIBRATION)\n")
    L.append("| L2 | Best calibration log-loss | Best epoch |")
    L.append("|---|---:|---:|")
    for l2_str, v in R["hyperparameter_grid"].items():
        L.append(
            f"| {l2_str} | {_fmt(v['best_cal_loss'])} | {v['best_epoch']} |"
        )
    sel = R["selected_model"]
    L.append(
        f"\n**Selected**: L2=`{sel['l2']:g}`, epochs=`{sel['epochs']}`, "
        f"calibration log-loss=`{_fmt(sel['cal_loss'])}`\n"
    )

    L.append("## Learned weights (standardized features)\n")
    L.append("| Feature | Weight |")
    L.append("|---|---:|")
    for name, w in zip(R["feature_columns"], sel["weights"]):
        L.append(f"| {name} | {_fmt(w)} |")
    L.append(f"| bias | {_fmt(sel['bias'])} |")
    L.append(
        f"\nPlatt: `a = {_fmt(sel['platt_a'])}`, "
        f"`b = {_fmt(sel['platt_b'])}`\n"
    )

    L.append("## Baseline\n")
    bp = R["baseline_primary"]
    L.append(
        f"- Tuned on TRAIN (max accuracy): "
        f"`t = {_fmt(bp['threshold'], 6)}`, "
        f"train accuracy = `{_fmt(bp['train_accuracy'])}`"
    )
    L.append("- Frozen reference: `t = 0.05` (repo default)")
    L.append(
        f"- Baseline Platt (for Brier / ECE): "
        f"`a = {_fmt(bp['platt_a'])}, b = {_fmt(bp['platt_b'])}`\n"
    )

    L.append("## Test partition — primary task\n")
    L.append("| Method | Accuracy [95% CI] | AUROC [95% CI] | ECE | Brier |")
    L.append("|---|---|---|---:|---:|")
    t = R["test_primary"]
    lrn, bsl, b05 = t["learned"], t["baseline_tuned"], t["baseline_at_0_05"]
    L.append(
        f"| Learned (Platt >= 0.5) | {_fmt(lrn['accuracy'])} "
        f"{_fmt_ci(*lrn['accuracy_ci'])} | {_fmt(lrn['auroc'])} "
        f"{_fmt_ci(*lrn['auroc_ci'])} | {_fmt(lrn['ece'])} | "
        f"{_fmt(lrn['brier'])} |"
    )
    L.append(
        f"| Baseline (tuned) | {_fmt(bsl['accuracy'])} "
        f"{_fmt_ci(*bsl['accuracy_ci'])} | {_fmt(bsl['auroc'])} "
        f"{_fmt_ci(*bsl['auroc_ci'])} | {_fmt(bsl['ece'])} | "
        f"{_fmt(bsl['brier'])} |"
    )
    L.append(
        f"| Baseline (t=0.05, frozen) | {_fmt(b05['accuracy'])} "
        f"{_fmt_ci(*b05['accuracy_ci'])} | {_fmt(b05['auroc'])} "
        f"{_fmt_ci(*b05['auroc_ci'])} | {_fmt(b05['ece'])} | "
        f"{_fmt(b05['brier'])} |"
    )
    L.append(
        "\nNote: AUROC, Brier and ECE are identical for the two baseline "
        "rows because they depend only on the score (f4) and its Platt fit, "
        "not on the threshold.\n"
    )
    diff = t["auroc_diff_learned_minus_baseline"]
    excl = diff["excludes_zero"]
    excl_str = "N/A" if excl is None else ("YES" if excl else "NO")
    L.append(
        f"**AUROC difference (learned - baseline):** "
        f"{_fmt_ci(diff['ci_low'], diff['ci_high'])} "
        f"(bootstrap resamples skipped: {diff['n_skipped_resamples']})\n"
    )
    L.append(f"- CI excludes 0: **{excl_str}**\n")

    L.append("## Precision-coverage (test, primary)\n")
    pc = t["precision_coverage"]
    L.append(
        "| Target coverage | Learned (realized cov / precision) | "
        "Baseline (realized cov / precision) |"
    )
    L.append("|---|---|---|")
    for target, (cl, pl), (cb, pb) in zip(
        (0.2, 0.4, 0.6, 0.8, 1.0),
        pc["learned_table"], pc["baseline_table"],
    ):
        L.append(
            f"| {_fmt(target)} | {_fmt(cl)} / {_fmt(pl)} | "
            f"{_fmt(cb)} / {_fmt(pb)} |"
        )
    L.append("")

    L.append("## Feature ablation (test, primary)\n")
    ab = R["ablation"]
    L.append(f"Full-model AUROC: `{_fmt(ab['full_auroc'])}`\n")
    L.append("### Drop one feature group (full - ablated)\n")
    L.append("| Group | Ablated AUROC | Drop (full - ablated) |")
    L.append("|---|---:|---:|")
    for g, v in ab["drop"].items():
        L.append(
            f"| {g} | {_fmt(v['auroc'])} | {_fmt(v['drop'])} |"
        )
    L.append("\n### Single-feature-only models (trained on one group)\n")
    L.append("| Group | Test AUROC |")
    L.append("|---|---:|")
    for g, v in ab["single"].items():
        L.append(f"| {g} | {_fmt(v['auroc'])} |")
    L.append("")

    L.append(
        "## Secondary task "
        "(adversarial: indexed-impossible vs indexed-answerable)\n"
    )
    sec = R["secondary"]
    if sec.get("variant_A"):
        A = sec["variant_A"]
        L.append("### Variant A — primary-trained model + primary-frozen baseline, on secondary test\n")
        L.append(
            f"- Learned AUROC: `{_fmt(A['learned_auroc'])}` "
            f"{_fmt_ci(*A['learned_auroc_ci'])} "
            f"(skipped: {A['learned_auroc_ci_skipped']})"
        )
        L.append(f"- Learned accuracy: `{_fmt(A['learned_accuracy'])}`")
        L.append(
            f"- Baseline AUROC (score = f4): `{_fmt(A['baseline_auroc'])}`"
        )
        L.append(
            f"- Baseline accuracy at primary-tuned "
            f"t=`{_fmt(A['baseline_threshold'], 6)}`: "
            f"`{_fmt(A['baseline_accuracy'])}`"
        )
        L.append(f"- Test set: n={A['n']}, n+={A['n_pos']}, n-={A['n_neg']}\n")
    else:
        L.append("Variant A: **N/A** (insufficient secondary test data)\n")

    if sec.get("variant_B") and "error" not in sec["variant_B"]:
        B = sec["variant_B"]
        L.append("### Variant B — secondary-trained model + secondary-tuned baseline, on secondary test\n")
        L.append(
            f"- Selected L2=`{B['selected_l2']:g}`, "
            f"epochs=`{B['selected_epochs']}`"
        )
        L.append(
            f"- Learned AUROC: `{_fmt(B['learned_auroc'])}` "
            f"{_fmt_ci(*B['learned_auroc_ci'])} "
            f"(skipped: {B['learned_auroc_ci_skipped']})"
        )
        L.append(f"- Learned accuracy: `{_fmt(B['learned_accuracy'])}`")
        L.append(
            f"- Baseline AUROC (score = f4): `{_fmt(B['baseline_auroc'])}`"
        )
        L.append(
            f"- Baseline accuracy at secondary-tuned "
            f"t=`{_fmt(B['baseline_threshold'], 6)}`: "
            f"`{_fmt(B['baseline_accuracy'])}`"
        )
        L.append(f"- Test set: n={B['n']}, n+={B['n_pos']}, n-={B['n_neg']}\n")
    else:
        L.append("Variant B: **N/A**\n")

    L.append("## Sanity checks\n")
    L.append("### Evaluation null check (both train and test labels shuffled)\n")
    nc = R["sanity_check_null"]
    L.append(
        f"- n={nc['n']} shuffles; "
        f"mean AUROC = `{_fmt(nc['mean'])}`, "
        f"std = `{_fmt(nc['std'])}`"
    )
    L.append(f"- WARNING fired: **{'YES' if nc['warn'] else 'NO'}**")
    L.append(
        "- Interpretation: this checks metric/label alignment, "
        "not feature leakage.\n"
    )
    L.append("### Feature leakage diagnostic (train labels shuffled, test labels real)\n")
    dg = R["sanity_check_diagnostic"]
    L.append(
        f"- n={dg['n']} shuffles; "
        f"mean = `{_fmt(dg['mean'])}`, "
        f"std = `{_fmt(dg['std'])}`, "
        f"min = `{_fmt(dg['min'])}`, "
        f"max = `{_fmt(dg['max'])}`"
    )
    L.append(
        "- Diagnostic only, never pass/fail: AUROC is scale-invariant, so a "
        "random weight direction can still align with the real test signal.\n"
    )

    L.append("## Raw feature means per partition (pre-standardization)\n")
    L.append("| Partition | " + " | ".join(R["feature_columns"]) + " |")
    L.append("|---|" + "---:|" * len(R["feature_columns"]))
    for name in PARTITIONS:
        m = R["feature_means_per_partition"][name]
        if m is None:
            L.append(
                f"| {name} | "
                + " | ".join(["N/A"] * len(R["feature_columns"]))
                + " |"
            )
        else:
            L.append(
                f"| {name} | "
                + " | ".join(_fmt(x) for x in m)
                + " |"
            )
    L.append("")

    L.append("## Runtime\n")
    rt = R["runtime_seconds"]
    L.append(
        f"- Retrieval (feature extraction + cache): "
        f"`{rt['retrieval']:.1f}s`"
    )
    L.append(
        f"- Training + evaluation + report: "
        f"`{rt['training_eval']:.1f}s`"
    )
    L.append(f"- Total: `{rt['total']:.1f}s`\n")

    L.append("## Interpretation (mechanical only)\n")
    L.append(
        f"- AUROC difference (learned - baseline) CI excludes 0: "
        f"**{excl_str}**"
    )
    L.append(
        "- No claim is made that the learned model \"beats\" the baseline; "
        "see the numbers in the tables above."
    )
    L.append(
        "- Retrieval dominates runtime (per the repo note: ~12 ms/query at "
        "1k chunks, ~36 ms at 2.7k chunks in pure Python); the disk cache "
        "makes re-training fast.\n"
    )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()