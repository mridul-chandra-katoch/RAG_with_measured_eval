"""Unit tests for ragkit.learned.logreg.

Covers: finite-difference gradient check, AUROC with hand-computed ties,
linear separability recovery, mini-batch determinism, Platt scaling,
metrics, precision-coverage tie handling, threshold tuning, cluster
bootstrap (incl. single-class skip), and split disjointness by article.
"""

import unittest

from ragkit.learned import logreg
from ragkit.learned.features import split_articles


class TestSigmoid(unittest.TestCase):
    def test_sigmoid_stable(self):
        self.assertAlmostEqual(logreg.sigmoid(0.0), 0.5)
        self.assertGreaterEqual(logreg.sigmoid(-1000.0), 0.0)
        self.assertLessEqual(logreg.sigmoid(1000.0), 1.0)
        self.assertAlmostEqual(
            logreg.sigmoid(3.0) + logreg.sigmoid(-3.0), 1.0, places=12
        )


class TestGradient(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        import random as _r
        rng = _r.Random(0)
        X = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(6)]
        y = [0, 1, 0, 1, 1, 0]
        w = [0.1, -0.2, 0.3]
        b = 0.05
        l2 = 0.01

        _, g_w, g_b = logreg.compute_loss_and_grad(X, y, w, b, l2)

        eps = 1e-6
        for j in range(3):
            w_p = list(w); w_p[j] += eps
            w_m = list(w); w_m[j] -= eps
            lp, _, _ = logreg.compute_loss_and_grad(X, y, w_p, b, l2)
            lm, _, _ = logreg.compute_loss_and_grad(X, y, w_m, b, l2)
            num = (lp - lm) / (2 * eps)
            self.assertAlmostEqual(
                num, g_w[j], places=5,
                msg=f"grad_w[{j}] mismatch",
            )

        lp, _, _ = logreg.compute_loss_and_grad(X, y, w, b + eps, l2)
        lm, _, _ = logreg.compute_loss_and_grad(X, y, w, b - eps, l2)
        num_b = (lp - lm) / (2 * eps)
        self.assertAlmostEqual(num_b, g_b, places=5, msg="grad_b mismatch")


class TestAUROC(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertAlmostEqual(
            logreg.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0
        )

    def test_inverse_separation(self):
        self.assertAlmostEqual(
            logreg.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0
        )

    def test_all_ties(self):
        self.assertAlmostEqual(
            logreg.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5
        )

    def test_partial_ties(self):
        # Sorted: (0.1,0) (0.5,0) (0.5,1) (0.9,1); ranks 1, 2.5, 2.5, 4.
        # sum_pos = 2.5 + 4 = 6.5; (6.5 - 3) / 4 = 0.875.
        self.assertAlmostEqual(
            logreg.auroc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]), 0.875
        )

    def test_single_class_returns_none(self):
        self.assertIsNone(logreg.auroc([0, 0, 0], [0.1, 0.2, 0.3]))
        self.assertIsNone(logreg.auroc([1, 1, 1], [0.1, 0.2, 0.3]))


class TestLogisticRegression(unittest.TestCase):
    def test_linearly_separable(self):
        X = [
            [-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0],
            [1.0, 1.0], [1.0, 2.0], [2.0, 1.0],
        ]
        y = [0, 0, 0, 1, 1, 1]
        w, b = logreg.train_logreg(
            X, y, l2=0.01, lr=0.5, epochs=2000, batch_size=0, seed=42,
        )
        preds = [
            1 if sum(wj * xj for wj, xj in zip(w, x)) + b >= 0 else 0
            for x in X
        ]
        self.assertEqual(preds, y)

    def test_full_batch_deterministic(self):
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
        y = [0, 0, 1, 1]
        w1, b1 = logreg.train_logreg(
            X, y, l2=0.0, lr=0.1, epochs=20, batch_size=0, seed=7
        )
        w2, b2 = logreg.train_logreg(
            X, y, l2=0.0, lr=0.1, epochs=20, batch_size=0, seed=7
        )
        self.assertEqual(w1, w2)
        self.assertEqual(b1, b2)

    def test_minibatch_runs_and_is_deterministic(self):
        X = [
            [0.1, 0.2], [0.3, -0.1], [-0.2, 0.5],
            [0.6, 0.4], [0.7, -0.3], [-0.4, 0.8],
        ]
        y = [0, 0, 1, 1, 1, 0]
        w1, b1 = logreg.train_logreg(
            X, y, l2=0.0, lr=0.1, epochs=50, batch_size=2, seed=1
        )
        w2, b2 = logreg.train_logreg(
            X, y, l2=0.0, lr=0.1, epochs=50, batch_size=2, seed=1
        )
        self.assertEqual(len(w1), 2)
        self.assertIsInstance(b1, float)
        self.assertEqual(w1, w2)
        self.assertEqual(b1, b2)


class TestPlatt(unittest.TestCase):
    def test_recovers_identity_on_calibrated_data(self):
        # For each logit z, create n=100 samples with round(100*sigmoid(z))
        # positives. Then the true calibration mapping is (a, b) = (1, 0).
        logits = []
        labels = []
        for z in (-2.0, -1.0, 0.0, 1.0, 2.0):
            p = logreg.sigmoid(z)
            n_pos = round(100 * p)
            for _ in range(n_pos):
                logits.append(z); labels.append(1)
            for _ in range(100 - n_pos):
                logits.append(z); labels.append(0)
        a, b = logreg.fit_platt(logits, labels)
        self.assertAlmostEqual(a, 1.0, delta=0.1)
        self.assertAlmostEqual(b, 0.0, delta=0.2)

    def test_increasing_a_when_signal_is_informative(self):
        logits = [1.0] * 200 + [-1.0] * 200
        labels = [1] * 200 + [0] * 200
        a, _ = logreg.fit_platt(logits, labels)
        self.assertGreater(a, 0.5)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(
            logreg.accuracy([0, 1, 0, 1], [0, 1, 1, 0]), 0.5
        )

    def test_brier(self):
        # preds [1, 0, 0, 1] vs labels [1, 0, 1, 1] -> error only at idx 2.
        # probs [1, 0, 0, 1] -> MSE = (0+0+1+0)/4 = 0.25.
        self.assertAlmostEqual(
            logreg.brier([1, 0, 1, 1], [1.0, 0.0, 0.0, 1.0]), 0.25
        )

    def test_ece_perfectly_calibrated(self):
        labels = [0, 1, 0, 1]
        probs = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(logreg.ece(labels, probs, n_bins=10), 0.0)

    def test_precision_coverage_ties_one_block(self):
        # All scores tied: the entire set is one block, so target 0.5
        # must include all four samples (realized coverage = 1.0).
        labels = [1, 0, 1, 0]
        scores = [0.5, 0.5, 0.5, 0.5]
        table, _ = logreg.precision_coverage(labels, scores, targets=(0.5,))
        cov, prec = table[0]
        self.assertAlmostEqual(cov, 1.0)
        self.assertAlmostEqual(prec, 0.5)


class TestThresholdTuning(unittest.TestCase):
    def test_simple_separation(self):
        f4 = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        t, acc = logreg.tune_threshold(f4, labels)
        self.assertAlmostEqual(acc, 1.0)
        # Only the midpoint between 0.2 and 0.8 gives 100% accuracy.
        self.assertAlmostEqual(t, 0.5, places=6)

    def test_edge_always_abstain(self):
        # All labels are 0; best is to abstain on everything, i.e. threshold
        # above every f4 value.
        f4 = [0.1, 0.2, 0.3]
        labels = [0, 0, 0]
        t, acc = logreg.tune_threshold(f4, labels)
        self.assertAlmostEqual(acc, 1.0)
        self.assertGreater(t, max(f4))


class TestBootstrap(unittest.TestCase):
    def _toy_clustered(self):
        labels, scores, article_ids = [], [], []
        for a in range(20):
            labels.extend([0, 1])
            scores.extend([0.1 + 0.01 * a, 0.9 - 0.01 * a])
            article_ids.extend([a, a])
        return labels, scores, article_ids

    def test_auroc_ci_covers_truth(self):
        labels, scores, ids = self._toy_clustered()
        lo, hi, skipped = logreg.cluster_bootstrap_auroc(
            labels, scores, ids, n_resamples=200, seed=0
        )
        self.assertIsNotNone(lo)
        self.assertGreaterEqual(lo, 0.9)
        self.assertLessEqual(hi, 1.0)

    def test_diff_ci_zero_when_same(self):
        labels, scores, ids = self._toy_clustered()
        lo, hi, skipped = logreg.cluster_bootstrap_auroc_diff(
            labels, scores, scores, ids, n_resamples=200, seed=0
        )
        self.assertIsNotNone(lo)
        self.assertAlmostEqual(lo, 0.0, places=10)
        self.assertAlmostEqual(hi, 0.0, places=10)

    def test_single_class_skipped(self):
        labels = [1, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.4]
        ids = [0, 1, 2, 3]
        lo, hi, skipped = logreg.cluster_bootstrap_auroc(
            labels, scores, ids, n_resamples=10, seed=0
        )
        self.assertIsNone(lo)
        self.assertIsNone(hi)
        self.assertEqual(skipped, 10)

    def test_accuracy_ci_bounds(self):
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        preds = [0, 1, 0, 1, 1, 0, 0, 1]  # one error at idx 4
        ids = [0, 1, 2, 3, 4, 5, 6, 7]
        lo, hi, skipped = logreg.cluster_bootstrap_accuracy(
            labels, preds, ids, n_resamples=50, seed=0
        )
        self.assertLessEqual(lo, 1.0)
        self.assertGreaterEqual(hi, 0.0)
        self.assertLessEqual(lo, hi)
        self.assertEqual(skipped, 0)


class TestSplitDisjoint(unittest.TestCase):
    def test_split_disjoint_and_deterministic(self):
        a = split_articles(442, 150, 42)
        self.assertEqual(
            len(a["train"]) + len(a["calibration"]) + len(a["test"]), 150
        )
        self.assertEqual(set(a["train"]) & set(a["calibration"]), set())
        self.assertEqual(set(a["train"]) & set(a["test"]), set())
        self.assertEqual(set(a["calibration"]) & set(a["test"]), set())
        b = split_articles(442, 150, 42)
        self.assertEqual(a, b)

    def test_split_too_small_raises(self):
        with self.assertRaises(ValueError):
            split_articles(442, 5, 42)


if __name__ == "__main__":
    unittest.main()