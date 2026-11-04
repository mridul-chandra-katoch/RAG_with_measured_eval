# Learned Abstention Classifier — SQuAD 2.0 Evaluation Report

## Configuration and seed

- seed: `42`
- max_articles (N): `150`
- lr: `0.1`
- max epochs: `500`
- batch_size: `0` (0 = full batch)
- log_every: `10`
- top_k: `20`
- SQuAD file: `datasets/squad/train-v2.0.json`

## Partition sizes (articles and index chunks)

| Partition | Articles | Indexed | Withheld | Index chunks |
|---|---:|---:|---:|---:|
| train | 90 | 63 | 27 | 2773 |
| calibration | 22 | 15 | 7 | 607 |
| test | 38 | 27 | 11 | 1121 |

## Prevalence and majority-class accuracy

### Primary task (indexed-answerable vs withheld-any)

| Partition | N | N+ | N- | Prevalence | Majority acc |
|---|---:|---:|---:|---:|---:|
| train | 21544 | 12209 | 9335 | 0.5667 | 0.5667 |
| calibration | 5002 | 2699 | 2303 | 0.5396 | 0.5396 |
| test | 7572 | 4810 | 2762 | 0.6352 | 0.6352 |

### Secondary task (indexed-answerable vs indexed-impossible)

| Partition | N | N+ | N- | Prevalence | Majority acc |
|---|---:|---:|---:|---:|---:|
| train | 18016 | 12209 | 5807 | 0.6777 | 0.6777 |
| calibration | 4127 | 2699 | 1428 | 0.6540 | 0.6540 |
| test | 7893 | 4810 | 3083 | 0.6094 | 0.6094 |

## Hyperparameter grid (L2 × epoch, selected on CALIBRATION)

| L2 | Best calibration log-loss | Best epoch |
|---|---:|---:|
| 0 | 0.4021 | 500 |
| 0.0001 | 0.4021 | 500 |
| 0.001 | 0.4020 | 500 |
| 0.01 | 0.4033 | 500 |
| 0.1 | 0.4439 | 500 |
| 1 | 0.5737 | 250 |

**Selected**: L2=`0.001`, epochs=`500`, calibration log-loss=`0.4020`

## Learned weights (standardized features)

| Feature | Weight |
|---|---:|
| f1 | -0.0030 |
| f2 | -0.0741 |
| f3 | 1.7350 |
| f4 | 0.7388 |
| f5a | 0.0406 |
| f5b | -0.0011 |
| f6 | -0.7472 |
| f7 | 0.6730 |
| bias | 0.8593 |

Platt: `a = 1.3262`, `b = 0.9124`

## Baseline

- Tuned on TRAIN (max accuracy): `t = 0.208614`, train accuracy = `0.7539`
- Frozen reference: `t = 0.05` (repo default)
- Baseline Platt (for Brier / ECE): `a = 2.0717, b = 1.0283`

## Test partition — primary task

| Method | Accuracy [95% CI] | AUROC [95% CI] | ECE | Brier |
|---|---|---|---:|---:|
| Learned (Platt >= 0.5) | 0.8168 [0.7787, 0.8526] | 0.8936 [0.8621, 0.9227] | 0.0215 | 0.1276 |
| Baseline (tuned) | 0.7291 [0.6916, 0.7666] | 0.8107 [0.7568, 0.8604] | 0.0549 | 0.1720 |
| Baseline (t=0.05, frozen) | 0.6352 [0.4838, 0.8017] | 0.8107 [0.7568, 0.8604] | 0.0549 | 0.1720 |

Note: AUROC, Brier and ECE are identical for the two baseline rows because they depend only on the score (f4) and its Platt fit, not on the threshold.

**AUROC difference (learned - baseline):** [0.0574, 0.1067] (bootstrap resamples skipped: 0)

- CI excludes 0: **YES**

## Precision-coverage (test, primary)

| Target coverage | Learned (realized cov / precision) | Baseline (realized cov / precision) |
|---|---|---|
| 0.2000 | 0.2001 / 0.9914 | 0.2001 / 0.9320 |
| 0.4000 | 0.4000 / 0.9584 | 0.4000 / 0.8795 |
| 0.6000 | 0.6001 / 0.8739 | 0.6001 / 0.8138 |
| 0.8000 | 0.8001 / 0.7559 | 0.8001 / 0.7339 |
| 1.0000 | 1.0000 / 0.6352 | 1.0000 / 0.6352 |

## Feature ablation (test, primary)

Full-model AUROC: `0.8936`

### Drop one feature group (full - ablated)

| Group | Ablated AUROC | Drop (full - ablated) |
|---|---:|---:|
| f1 | 0.8936 | 0.0000 |
| f2 | 0.8938 | -0.0002 |
| f3 | 0.8432 | 0.0505 |
| f4 | 0.8983 | -0.0047 |
| f5 | 0.8938 | -0.0001 |
| f6 | 0.8778 | 0.0158 |
| f7 | 0.8914 | 0.0022 |

### Single-feature-only models (trained on one group)

| Group | Test AUROC |
|---|---:|
| f1 | 0.6182 |
| f2 | 0.5548 |
| f3 | 0.8708 |
| f4 | 0.8107 |
| f5 | 0.6281 |
| f6 | 0.4616 |
| f7 | 0.7947 |

## Secondary task (adversarial: indexed-impossible vs indexed-answerable)

### Variant A — primary-trained model + primary-frozen baseline, on secondary test

- Learned AUROC: `0.5510` [0.5013, 0.6031] (skipped: 0)
- Learned accuracy: `0.5971`
- Baseline AUROC (score = f4): `0.5633`
- Baseline accuracy at primary-tuned t=`0.208614`: `0.5751`
- Test set: n=7893, n+=4810, n-=3083

### Variant B — secondary-trained model + secondary-tuned baseline, on secondary test

- Selected L2=`0.01`, epochs=`500`
- Learned AUROC: `0.5651` [0.5213, 0.6121] (skipped: 0)
- Learned accuracy: `0.6086`
- Baseline AUROC (score = f4): `0.5633`
- Baseline accuracy at secondary-tuned t=`0.079686`: `0.6093`
- Test set: n=7893, n+=4810, n-=3083

## Sanity checks

### Evaluation null check (both train and test labels shuffled)

- n=20 shuffles; mean AUROC = `0.4995`, std = `0.0051`
- WARNING fired: **NO**
- Interpretation: this checks metric/label alignment, not feature leakage.

### Feature leakage diagnostic (train labels shuffled, test labels real)

- n=20 shuffles; mean = `0.4787`, std = `0.1368`, min = `0.2253`, max = `0.7275`
- Diagnostic only, never pass/fail: AUROC is scale-invariant, so a random weight direction can still align with the real test signal.

## Raw feature means per partition (pre-standardization)

| Partition | f1 | f2 | f3 | f4 | f5a | f5b | f6 | f7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.0326 | 0.0007 | 19.5429 | 0.2459 | 0.5578 | 0.4375 | 10.0905 | 0.5516 |
| calibration | 0.0326 | 0.0006 | 15.5887 | 0.2180 | 0.6371 | 0.5407 | 9.8663 | 0.5276 |
| test | 0.0326 | 0.0006 | 17.8485 | 0.2382 | 0.6276 | 0.4956 | 9.5543 | 0.5559 |

## Runtime

- Retrieval (feature extraction + cache): `0.5s`
- Training + evaluation + report: `1132.6s`
- Total: `1133.1s`

## Interpretation (mechanical only)

- AUROC difference (learned - baseline) CI excludes 0: **YES**
- No claim is made that the learned model "beats" the baseline; see the numbers in the tables above.
- Retrieval dominates runtime (per the repo note: ~12 ms/query at 1k chunks, ~36 ms at 2.7k chunks in pure Python); the disk cache makes re-training fast.

