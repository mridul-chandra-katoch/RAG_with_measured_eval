"""From-scratch logistic regression, Platt scaling, evaluation metrics,
cluster bootstrap CIs, ablation, and leakage sanity checks.

Design:
- No ragkit imports: this module is testable on its own.
- Python 3.12 stdlib only.
- Deterministic given the seed (full-batch and mini-batch paths).
- Bias is never L2-penalized.
- AUROC uses average ranks for ties (Mann-Whitney).
- Cluster bootstrap resamples ARTICLES with replacement; the paired
  AUROC-difference bootstrap uses the same resample for both methods.
"""

import math
import random
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def sigmoid(z: float) -> float:
    """Numerically stable logistic function 1 / (1 + exp(-z))."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def compute_loss_and_grad(X, y, w, b, l2=0.0):
    """Regularized cross-entropy and its exact gradient (full batch).

    Inputs:
        X: list[list[float]], n x d.
        y: list[int], 0/1.
        w: list[float], length d.
        b: float.
        l2: L2 strength on w (bias not penalized).
    Outputs:
        (loss, grad_w, grad_b).
    Assumptions:
        Non-empty X. Uses the mean over samples for the data term.
    """
    n = len(X)
    d = len(w)
    loss = 0.0
    g_w = [0.0] * d
    g_b = 0.0
    for i in range(n):
        z = sum(wj * xj for wj, xj in zip(w, X[i])) + b
        p = sigmoid(z)
        pc = max(min(p, 1.0 - 1e-15), 1e-15)
        loss -= y[i] * math.log(pc) + (1.0 - y[i]) * math.log(1.0 - pc)
        err = p - y[i]
        for j in range(d):
            g_w[j] += err * X[i][j]
        g_b += err
    loss = loss / n + 0.5 * l2 * sum(wj * wj for wj in w)
    for j in range(d):
        g_w[j] = g_w[j] / n + l2 * w[j]
    g_b = g_b / n
    return loss, g_w, g_b


def log_loss(X, y, w, b):
    """Unregularized mean cross-entropy (used for model selection)."""
    n = len(y)
    total = 0.0
    for i in range(n):
        z = sum(wj * xj for wj, xj in zip(w, X[i])) + b
        p = sigmoid(z)
        pc = max(min(p, 1.0 - 1e-15), 1e-15)
        total -= y[i] * math.log(pc) + (1.0 - y[i]) * math.log(1.0 - pc)
    return total / n


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_logreg(
    X,
    y,
    l2: float = 0.0,
    lr: float = 0.1,
    epochs: int = 100,
    batch_size: int = 0,
    seed: int = 42,
    callback: Optional[Callable] = None,
    callback_every: int = 1,
):
    """Train logistic regression by full-batch GD or mini-batch SGD.

    Inputs:
        X: list[list[float]] (already standardized if desired).
        y: list[int], 0/1 labels.
        l2: L2 regularization strength on weights; bias not penalized.
        lr: learning rate.
        epochs: number of passes.
        batch_size: 0 = full batch; >0 = mini-batch size.
        seed: RNG seed controlling mini-batch shuffle order.
        callback: optional callable(epoch, w, b) invoked every
            `callback_every` epochs.
        callback_every: checkpoint interval (>=1).
    Outputs:
        (w, b).
    Assumptions:
        Deterministic given (seed, batch_size, epochs) and X, y.
        Weights start at 0, bias at 0.
    """
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        if batch_size <= 0:
            _, g_w, g_b = compute_loss_and_grad(X, y, w, b, l2)
            for j in range(d):
                w[j] -= lr * g_w[j]
            b -= lr * g_b
        else:
            order = list(range(n))
            rng.shuffle(order)
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                m = len(batch)
                if m == 0:
                    continue
                g_w = [0.0] * d
                g_b = 0.0
                for i in batch:
                    z = sum(wj * xj for wj, xj in zip(w, X[i])) + b
                    p = sigmoid(z)
                    err = p - y[i]
                    for j in range(d):
                        g_w[j] += err * X[i][j]
                    g_b += err
                for j in range(d):
                    w[j] -= lr * (g_w[j] / m + l2 * w[j])
                b -= lr * (g_b / m)
        if callback is not None and epoch % callback_every == 0:
            callback(epoch, w, b)
    return w, b


def select_hyperparams(
    X_tr,
    y_tr,
    X_cal,
    y_cal,
    l2_grid=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    lr: float = 0.1,
    epochs: int = 100,
    batch_size: int = 0,
    seed: int = 42,
    log_every: int = 10,
    verbose: bool = False,
):
    """Grid-search (L2, epoch) by UNREGULARIZED calibration log-loss.

    Inputs:
        X_tr, y_tr: train partition (standardized with train stats).
        X_cal, y_cal: calibration partition (standardized with the same
            stats). TEST labels are never touched.
        l2_grid: candidate L2 values.
        lr, epochs, batch_size, seed: training knobs.
        log_every: checkpoint interval; also the resolution of the epoch
            search.
        verbose: if True, print calibration log-loss at every checkpoint.
    Outputs:
        (per_l2, best) where
        - per_l2: {L2: {"best_loss": float, "best_epoch": int,
                        "weights": list, "bias": float}}
        - best: (loss, L2, w, b, epoch) of the overall winner.
    Assumptions:
        The winning snapshot is used as the final model. Given the same
        seed, this is numerically identical to refitting for exactly
        `epochs = best_epoch` iterations.
    """
    per_l2 = {}
    best_overall = None  # (loss, l2, w, b, epoch)

    for l2 in l2_grid:
        tracker = {
            "best_loss": None,
            "best_epoch": None,
            "weights": None,
            "bias": None,
        }

        def _cb(epoch, w, b, _tracker=tracker, _l2=l2):
            nonlocal best_overall
            cal_loss = log_loss(X_cal, y_cal, w, b)
            if verbose:
                print(
                    f"  [l2={_l2:g} epoch={epoch}] cal_loss={cal_loss:.6f}"
                )
            if _tracker["best_loss"] is None or cal_loss < _tracker["best_loss"]:
                _tracker["best_loss"] = cal_loss
                _tracker["best_epoch"] = epoch
                _tracker["weights"] = list(w)
                _tracker["bias"] = b
            if best_overall is None or cal_loss < best_overall[0]:
                best_overall = (cal_loss, _l2, list(w), b, epoch)

        train_logreg(
            X_tr,
            y_tr,
            l2=l2,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            callback=_cb,
            callback_every=log_every,
        )
        per_l2[l2] = tracker

    if best_overall is None:
        raise RuntimeError(
            "select_hyperparams: no checkpoints recorded "
            "(epochs must be >= log_every and log_every >= 1)"
        )
    return per_l2, best_overall


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

def fit_platt(logits, labels, max_iter: int = 100, tol: float = 1e-8):
    """Fit p = sigmoid(a*logit + b) on the CALIBRATION partition.

    Inputs:
        logits: list[float], model raw scores (w·x + b).
        labels: list[int], 0/1.
        max_iter, tol: Newton iterations and convergence tolerance.
    Outputs:
        (a, b). Initialized at (1, 0). No regularization.
    Assumptions:
        Falls back to a small GD step if the 2x2 Hessian is singular.
    """
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        g_a = g_b = 0.0
        h_aa = h_ab = h_bb = 0.0
        for z, y in zip(logits, labels):
            p = sigmoid(a * z + b)
            r = p - y
            w = p * (1.0 - p)
            g_a += r * z
            g_b += r
            h_aa += w * z * z
            h_ab += w * z
            h_bb += w
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            a -= 0.01 * g_a
            b -= 0.01 * g_b
            continue
        da = -(h_bb * g_a - h_ab * g_b) / det
        db = -(-h_ab * g_a + h_aa * g_b) / det
        a += da
        b += db
        if abs(da) < tol and abs(db) < tol:
            break
    return a, b


def apply_platt(logits, a: float, b: float):
    """Apply Platt scaling: returns [sigmoid(a*logit + b) for logit in logits]."""
    return [sigmoid(a * z + b) for z in logits]


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def compute_standardization(X):
    """Return (means, stds) from X; std of 0 becomes 1.0."""
    n = len(X)
    d = len(X[0])
    means = [0.0] * d
    for x in X:
        for j in range(d):
            means[j] += x[j]
    means = [m / n for m in means]
    stds = [0.0] * d
    for x in X:
        for j in range(d):
            stds[j] += (x[j] - means[j]) ** 2
    stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]
    return means, stds


def standardize(X, means, stds):
    """Apply (x - mean) / std column-wise."""
    return [
        [(x[j] - means[j]) / stds[j] for j in range(len(x))]
        for x in X
    ]


def feature_means(X):
    """Per-column mean for the raw feature-shift report."""
    n = len(X)
    d = len(X[0])
    return [sum(x[j] for x in X) / n for j in range(d)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(labels, scores):
    """Mann-Whitney AUROC with average ranks for ties.

    Returns None if all labels are the same class.
    """
    n = len(labels)
    if n == 0:
        return None
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    sum_pos = sum(ranks[k] for k in range(n) if labels[order[k]] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy(labels, preds):
    """Fraction correct given 0/1 predictions."""
    return sum(1 for p, y in zip(preds, labels) if p == y) / len(labels)


def brier(labels, probs):
    """Mean squared error between probabilities and 0/1 labels."""
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels)


def ece(labels, probs, n_bins: int = 10):
    """Expected calibration error with 10 equal-width bins (default)."""
    n = len(labels)
    buckets = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(max(int(p * n_bins), 0), n_bins - 1)
        buckets[idx].append((p, y))
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_y = sum(y for _, y in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(avg_p - avg_y)
    return total


def precision_coverage(labels, scores, targets=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Precision at target coverages, treating tied scores as one block.

    Inputs:
        labels: list[int], 0/1.
        scores: list[float], higher = more likely to answer.
        targets: target coverage fractions.
    Outputs:
        (table, full_curve) where
        - table: list[(realized_coverage, precision)] aligned with targets;
        - full_curve: list[(coverage, precision)] at every distinct-score
          block boundary (in descending-score order).
    Assumptions:
        Tied items are never split; realized coverage may exceed the target.
    """
    n = len(labels)
    order = sorted(range(n), key=lambda i: -scores[i])
    blocks = []
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        blocks.append((j - i, sum(labels[order[k]] for k in range(i, j))))
        i = j

    def _walk(target_frac):
        need = target_frac * n
        cum_sz = cum_pos = 0
        for sz, ps in blocks:
            cum_sz += sz
            cum_pos += ps
            if cum_sz >= need:
                break
        if cum_sz == 0:
            return (0.0, 0.0)
        return (cum_sz / n, cum_pos / cum_sz)

    table = [_walk(t) for t in targets]

    full = []
    cum_sz = cum_pos = 0
    for sz, ps in blocks:
        cum_sz += sz
        cum_pos += ps
        full.append((cum_sz / n, cum_pos / cum_sz))

    return table, full


def tune_threshold(f4, labels):
    """Baseline threshold t: answer iff f4 >= t. Tuned to max ACCURACY.

    Inputs:
        f4: list[float], top TF-IDF scores on the TRAIN partition.
        labels: list[int], 0/1.
    Outputs:
        (t, train_accuracy).
    Assumptions:
        Search space is midpoints between consecutive sorted unique f4
        values, plus two edge candidates (always answer, always abstain).
        Ties on accuracy break toward the LOWER threshold.
    """
    unique = sorted(set(f4))
    if len(unique) < 2:
        t = unique[0] if unique else 0.0
        acc = max(sum(labels) / len(labels), 1.0 - sum(labels) / len(labels))
        return t, acc
    candidates = [
        (unique[i] + unique[i + 1]) / 2.0 for i in range(len(unique) - 1)
    ]
    candidates.append(unique[0] - 1e-6)   # always answer
    candidates.append(unique[-1] + 1e-6)  # always abstain
    best_t, best_acc = None, -1.0
    for t in candidates:
        preds = [1 if f >= t else 0 for f in f4]
        acc = accuracy(labels, preds)
        if acc > best_acc or (acc == best_acc and (best_t is None or t < best_t)):
            best_acc = acc
            best_t = t
    return best_t, best_acc


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------

def _percentile(sorted_vals, p):
    """Linear-interpolation percentile of a pre-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = p * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _group_by_article(article_ids):
    """Return {article_id: [row_indices]} preserving row order."""
    groups = {}
    for i, a in enumerate(article_ids):
        groups.setdefault(a, []).append(i)
    return groups


def cluster_bootstrap_auroc(labels, scores, article_ids, n_resamples=1000, seed=42):
    """Cluster bootstrap 95% CI for AUROC (resample articles with replacement).

    Returns (lo, hi, n_skipped). n_skipped counts resamples with a single class.
    """
    rng = random.Random(seed)
    unique = sorted(set(article_ids))
    groups = _group_by_article(article_ids)
    vals = []
    for _ in range(n_resamples):
        picks = [rng.choice(unique) for _ in range(len(unique))]
        idx = []
        for a in picks:
            idx.extend(groups[a])
        auc = auroc([labels[i] for i in idx], [scores[i] for i in idx])
        if auc is not None:
            vals.append(auc)
    n_skipped = n_resamples - len(vals)
    if not vals:
        return None, None, n_skipped
    vals.sort()
    return _percentile(vals, 0.025), _percentile(vals, 0.975), n_skipped


def cluster_bootstrap_accuracy(labels, preds, article_ids, n_resamples=1000, seed=42):
    """Cluster bootstrap 95% CI for accuracy. Always returns (lo, hi, 0)."""
    rng = random.Random(seed)
    unique = sorted(set(article_ids))
    groups = _group_by_article(article_ids)
    vals = []
    for _ in range(n_resamples):
        picks = [rng.choice(unique) for _ in range(len(unique))]
        idx = []
        for a in picks:
            idx.extend(groups[a])
        correct = sum(1 for i in idx if labels[i] == preds[i])
        vals.append(correct / len(idx))
    vals.sort()
    return _percentile(vals, 0.025), _percentile(vals, 0.975), 0


def cluster_bootstrap_auroc_diff(
    labels, scores_a, scores_b, article_ids, n_resamples=1000, seed=42
):
    """Paired cluster bootstrap 95% CI for AUROC(a) - AUROC(b).

    Uses the SAME resample for both methods so the difference is paired.
    Returns (lo, hi, n_skipped).
    """
    rng = random.Random(seed)
    unique = sorted(set(article_ids))
    groups = _group_by_article(article_ids)
    vals = []
    for _ in range(n_resamples):
        picks = [rng.choice(unique) for _ in range(len(unique))]
        idx = []
        for a in picks:
            idx.extend(groups[a])
        y_s = [labels[i] for i in idx]
        auc_a = auroc(y_s, [scores_a[i] for i in idx])
        auc_b = auroc(y_s, [scores_b[i] for i in idx])
        if auc_a is not None and auc_b is not None:
            vals.append(auc_a - auc_b)
    n_skipped = n_resamples - len(vals)
    if not vals:
        return None, None, n_skipped
    vals.sort()
    return _percentile(vals, 0.025), _percentile(vals, 0.975), n_skipped


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def ablation_study(
    X_tr,
    y_tr,
    X_te,
    y_te,
    group_cols,
    l2=0.0,
    lr=0.1,
    epochs=100,
    batch_size=0,
    seed=42,
):
    """Train with each feature group dropped, and each group alone.

    Inputs:
        X_tr, y_tr, X_te, y_te: standardized with TRAIN stats.
        group_cols: dict {group_name: [column indices]}.
        l2, lr, epochs, batch_size, seed: the SELECTED hyperparameters.
    Outputs:
        {
          "full_auroc": float | None,
          "drop": {group: {"auroc": float | None, "drop": float | None}},
          "single": {group: {"auroc": float | None}},
        }
    Assumptions:
        AUROC is rank-invariant so re-standardization and Platt refit are
        skipped. TEST labels are used only to SCORE, never to select.
    """
    d = len(X_tr[0])
    w_full, b_full = train_logreg(
        X_tr, y_tr, l2=l2, lr=lr, epochs=epochs,
        batch_size=batch_size, seed=seed,
    )
    full_scores = [
        sum(wj * xj for wj, xj in zip(w_full, x)) + b_full for x in X_te
    ]
    full_auroc = auroc(y_te, full_scores)

    drop_out = {}
    for group, cols in group_cols.items():
        keep = [j for j in range(d) if j not in cols]
        X_tr_sub = [[x[j] for j in keep] for x in X_tr]
        X_te_sub = [[x[j] for j in keep] for x in X_te]
        w, b = train_logreg(
            X_tr_sub, y_tr, l2=l2, lr=lr, epochs=epochs,
            batch_size=batch_size, seed=seed,
        )
        scores = [sum(wj * xj for wj, xj in zip(w, x)) + b for x in X_te_sub]
        auc = auroc(y_te, scores)
        if full_auroc is not None and auc is not None:
            delta = full_auroc - auc
        else:
            delta = None
        drop_out[group] = {"auroc": auc, "drop": delta}

    single_out = {}
    for group, cols in group_cols.items():
        X_tr_sub = [[x[j] for j in cols] for x in X_tr]
        X_te_sub = [[x[j] for j in cols] for x in X_te]
        w, b = train_logreg(
            X_tr_sub, y_tr, l2=l2, lr=lr, epochs=epochs,
            batch_size=batch_size, seed=seed,
        )
        scores = [sum(wj * xj for wj, xj in zip(w, x)) + b for x in X_te_sub]
        single_out[group] = {"auroc": auroc(y_te, scores)}

    return {"full_auroc": full_auroc, "drop": drop_out, "single": single_out}


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def evaluation_null_check(
    X_tr, y_tr, X_te, y_te, l2, lr, epochs, batch_size, n_shuffles=20, seed=42,
):
    """Metric/label-alignment check: shuffle BOTH train and test labels.

    Inputs:
        Standardized features and labels for train / test.
        Selected hyperparameters.
        n_shuffles: number of independent shuffle pairs.
        seed: fixed seed for shuffle draws.
    Outputs:
        {"mean", "std", "n", "warn", "aurocs"}.
        warn=True if |mean - 0.5| > 0.03 or any |AUROC - 0.5| > 0.05.
    Assumptions:
        Detects bugs in the metric/label alignment, NOT feature leakage.
    """
    rng = random.Random(seed)
    aurocs = []
    for _ in range(n_shuffles):
        y_tr_s = list(y_tr)
        y_te_s = list(y_te)
        rng.shuffle(y_tr_s)
        rng.shuffle(y_te_s)
        w, b = train_logreg(
            X_tr, y_tr_s, l2=l2, lr=lr, epochs=epochs,
            batch_size=batch_size, seed=seed,
        )
        scores = [sum(wj * xj for wj, xj in zip(w, x)) + b for x in X_te]
        auc = auroc(y_te_s, scores)
        if auc is not None:
            aurocs.append(auc)
    if not aurocs:
        return {"mean": None, "std": None, "n": 0, "warn": False, "aurocs": []}
    mean = sum(aurocs) / len(aurocs)
    std = math.sqrt(sum((a - mean) ** 2 for a in aurocs) / len(aurocs))
    warn = abs(mean - 0.5) > 0.03 or any(abs(a - 0.5) > 0.05 for a in aurocs)
    return {
        "mean": mean, "std": std, "n": len(aurocs),
        "warn": warn, "aurocs": aurocs,
    }


def feature_leakage_diagnostic(
    X_tr, y_tr, X_te, y_te, l2, lr, epochs, batch_size, n_shuffles=20, seed=42,
):
    """Diagnostic only: shuffled TRAIN labels -> REAL test labels.

    Outputs:
        {"mean", "std", "min", "max", "n", "aurocs"}.
    Assumptions:
        This is DIAGNOSTIC. AUROC is scale-invariant, so even a random
        direction can align with the real signal; do NOT use a pass/fail
        threshold on this.
    """
    rng = random.Random(seed)
    aurocs = []
    for _ in range(n_shuffles):
        y_tr_s = list(y_tr)
        rng.shuffle(y_tr_s)
        w, b = train_logreg(
            X_tr, y_tr_s, l2=l2, lr=lr, epochs=epochs,
            batch_size=batch_size, seed=seed,
        )
        scores = [sum(wj * xj for wj, xj in zip(w, x)) + b for x in X_te]
        auc = auroc(y_te, scores)
        if auc is not None:
            aurocs.append(auc)
    if not aurocs:
        return {
            "mean": None, "std": None, "min": None, "max": None,
            "n": 0, "aurocs": [],
        }
    mean = sum(aurocs) / len(aurocs)
    std = math.sqrt(sum((a - mean) ** 2 for a in aurocs) / len(aurocs))
    return {
        "mean": mean, "std": std,
        "min": min(aurocs), "max": max(aurocs),
        "n": len(aurocs), "aurocs": aurocs,
    }