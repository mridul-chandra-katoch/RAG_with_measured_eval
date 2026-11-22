#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

from ragkit.learned.features import (
    FEATURE_COLUMNS, FEATURE_GROUP_COLUMNS, assign_indexed_withheld,
    build_or_load_cache, split_articles,
)
from ragkit.learned import logreg as LR

SQUAD_DEFAULT = "datasets/squad/train-v2.0.json"
MODEL_PATH = Path("models/abstention_lr.json")
RESULTS_JSON = Path("results/abstention_results.json")
REPORT_MD = Path("results/abstention_report.md")
CACHE_PATH = Path("results/cache/abstention_cache.json")

PARTITIONS = ("train", "calibration", "test")

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--max-articles", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--data-path", default=SQUAD_DEFAULT)
    return p.parse_args(argv)

def _load_squad(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"SQuAD file not found at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))

def _assert_disjoint(split):
    s, c, t = set(split["train"]), set(split["calibration"]), set(split["test"])
    assert not (s & c) and not (s & t) and not (c & t), "Article splits overlap!"

def _print_partition_stats(squad, split, indexed_map, withheld_map):
    print("Partition sizes:")
    for name in PARTITIONS:
        n_chunks = sum(len(squad["data"][i]["paragraphs"]) for i in indexed_map[name])
        print(f"  {name:<12s}: {len(split[name]):>3d} articles ({len(indexed_map[name])} indexed, {len(withheld_map[name])} withheld), {n_chunks} chunks")

def _filter_primary(records):
    out = []
    for r in records:
        if r["indexed"] and r["is_impossible"]:
            continue
        out.append((r, 1 if r["indexed"] else 0))
    return out

def _filter_secondary(records):
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
    return [list(r["features"]) for r, _ in pairs], [y for _, y in pairs], [r["article_idx"] for r, _ in pairs]

def _prevalence(labels):
    n = len(labels)
    if n == 0: return {"n": 0, "n_pos": 0, "n_neg": 0, "prevalence": 0.0, "majority_accuracy": 0.0}
    n_pos = sum(labels)
    return {"n": n, "n_pos": n_pos, "n_neg": n - n_pos, "prevalence": n_pos / n, "majority_accuracy": max(n_pos, n - n_pos) / n}

def _train_pipeline(X_tr_raw, y_tr, X_cal_raw, y_cal, args):
    means, stds = LR.compute_standardization(X_tr_raw)
    X_tr = LR.standardize(X_tr_raw, means, stds)
    X_cal = LR.standardize(X_cal_raw, means, stds)
    per_l2, best = LR.select_hyperparams(
        X_tr, y_tr, X_cal, y_cal, lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed, log_every=args.log_every,
    )
    best_loss, best_l2, w, b, best_epoch = best
    cal_logits = [sum(wj * xj for wj, xj in zip(w, x)) + b for x in X_cal]
    a_platt, b_platt = LR.fit_platt(cal_logits, y_cal)
    return {
        "means": means, "stds": stds, "per_l2": per_l2, "best_loss": best_loss,
        "best_l2": best_l2, "best_epoch": best_epoch, "w": w, "b": b,
        "platt_a": a_platt, "platt_b": b_platt,
    }

def _predict_learned(model, X_raw):
    X = LR.standardize(X_raw, model["means"], model["stds"])
    logits = [sum(wj * xj for wj, xj in zip(model["w"], x)) + model["b"] for x in X]
    return LR.apply_platt(logits, model["platt_a"], model["platt_b"])

def _fit_baseline_platt(f4_cal, y_cal, mean, std):
    return LR.fit_platt([(f - mean) / (std if std > 0 else 1.0) for f in f4_cal], y_cal)

def _baseline_probs(f4, a, b, mean, std):
    return LR.apply_platt([(f - mean) / (std if std > 0 else 1.0) for f in f4], a, b)

def _test_metrics(labels, scores, probs):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return {
        "n": len(labels), "n_pos": sum(labels), "n_neg": len(labels) - sum(labels),
        "accuracy": LR.accuracy(labels, preds), "auroc": LR.auroc(labels, scores),
        "brier": LR.brier(labels, probs), "ece": LR.ece(labels, probs, n_bins=10),
        "preds": preds,
    }

def _fmt(v, digits=4):
    return "N/A" if v is None else f"{v:.{digits}f}"

def _fmt_ci(lo, hi, digits=4):
    return "N/A" if lo is None or hi is None else f"[{_fmt(lo, digits)}, {_fmt(hi, digits)}]"

def main(argv=None):
    args = _parse_args(argv)
    t_start = time.time()
    
    squad = _load_squad(args.data_path)
    split = split_articles(len(squad["data"]), args.max_articles, args.seed)
    indexed_map, withheld_map = {}, {}
    for name in PARTITIONS:
        indexed_map[name], withheld_map[name] = assign_indexed_withheld(split[name], args.seed, name)
    _assert_disjoint(split)
    _print_partition_stats(squad, split, indexed_map, withheld_map)

    cache = build_or_load_cache(
        squad, split, indexed_map, withheld_map,
        top_k=args.top_k, seed=args.seed, max_articles=args.max_articles,
        cache_path=CACHE_PATH, rebuild=args.rebuild_cache,
    )
    retrieval_seconds = time.time() - t_start
    records = cache["records"]

    primary_by = _by_partition(_filter_primary(records))
    primary_data = {name: _partition_arrays(primary_by[name]) for name in PARTITIONS}

    print("[train] fitting primary model...")
    model_primary = _train_pipeline(
        primary_data["train"][0], primary_data["train"][1],
        primary_data["calibration"][0], primary_data["calibration"][1],
        args,
    )
    print(f"[train] selected L2={model_primary['best_l2']:g} epochs={model_primary['best_epoch']} cal_loss={model_primary['best_loss']:.6f}")

    test_y = primary_data["test"][1]
    test_ids = primary_data["test"][2]
    test_probs = _predict_learned(model_primary, primary_data["test"][0])
    learned_metrics = _test_metrics(test_y, test_probs, test_probs)

    # Baseline (Top TF-IDF score is now index 2: f3_tfidf)
    f4_train = [X[2] for X in primary_data["train"][0]]
    f4_cal = [X[2] for X in primary_data["calibration"][0]]
    f4_test = [X[2] for X in primary_data["test"][0]]
    
    baseline_t, baseline_train_acc = LR.tune_threshold(f4_train, primary_data["train"][1])
    b_platt_a, b_platt_b = _fit_baseline_platt(f4_cal, primary_data["calibration"][1], model_primary["means"][2], model_primary["stds"][2])
    baseline_test_probs = _baseline_probs(f4_test, b_platt_a, b_platt_b, model_primary["means"][2], model_primary["stds"][2])
    baseline_test_preds = [1 if f >= baseline_t else 0 for f in f4_test]
    baseline_metrics = _test_metrics(test_y, f4_test, baseline_test_probs)

    # Correct paired bootstrap CI
    diff_lo, diff_hi, _ = LR.cluster_bootstrap_auroc_diff(test_y, test_probs, f4_test, test_ids, n_resamples=1000, seed=args.seed)

    print(f"[primary test] learned AUROC={_fmt(learned_metrics['auroc'])} "
          f"baseline AUROC={_fmt(baseline_metrics['auroc'])} "
          f"diff CI={_fmt_ci(diff_lo, diff_hi)}")

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS,
        "weights": model_primary["w"], "bias": model_primary["b"],
        "means": model_primary["means"], "stds": model_primary["stds"],
        "platt_a": model_primary["platt_a"], "platt_b": model_primary["platt_b"],
        "chosen_l2": model_primary["best_l2"], "chosen_epochs": model_primary["best_epoch"],
        "baseline_threshold": baseline_t,
    }, indent=2))

    print(f"[runtime] total={time.time() - t_start:.1f}s")

if __name__ == "__main__":
    main()