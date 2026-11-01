# CES-FL upgrade seed 17: completed training and test evaluation

Observed UTC: 2026-11-01T04:42:46.223256+00:00. Source commit: `875bce025084871438ef799ca9086985c4375672`.

**All five ten-epoch training runs and all 25 frozen-checkpoint test evaluations completed.** Every evaluation worker exited zero. Selected checkpoints reloaded and reproduced the saved validation AUC exactly; checkpoint bytes were rehashed after testing. Model state and validation thresholds remained unchanged.

## Selected validation results

| Arm | Epoch | Validation AUC (%) | ids8 worst-group AUC (%) | ids8 FPR gap (pp) |
|---|---:|---:|---:|---:|
| R0_frozen_bce | 7 | 98.1949 | 95.7111 | 28.2642 |
| R1_partial_bce | 1 | 98.2215 | 95.9441 | 24.2907 |
| R2_partial_dro | 5 | 98.2358 | 95.9591 | 22.9662 |
| RN_partial_dro_nochange_pilot | 3 | 97.5249 | 95.0694 | 25.4336 |
| RD_partial_dro_age_pilot | 5 | 98.1618 | 95.7299 | 19.2435 |

## Frozen-checkpoint test AUC

Percentages. Every column uses the same locked native manifest for all arms. FairForge-2 is transfer testing from FF++.

| Arm | FF++ | DFDC | Celeb-DF | DFD | FairForge-2 transfer |
|---|---:|---:|---:|---:|---:|
| R0_frozen_bce | 98.3467 | 59.4712 | 74.3252 | 82.3578 | 41.6819 |
| R1_partial_bce | 98.4013 | 58.9726 | 74.2086 | 81.8093 | 41.8710 |
| R2_partial_dro | 98.3962 | 59.5189 | 74.6706 | 82.1211 | 42.8294 |
| RN_partial_dro_nochange_pilot | 97.8472 | 59.6744 | 71.0136 | 81.5317 | 38.8312 |
| RD_partial_dro_age_pilot | 98.1965 | 59.8601 | 72.7958 | 82.2801 | 39.5290 |

## FF++ test fairness at each stored validation threshold

Group gaps are max minus min among eligible groups, in percentage points. They differ from Ding-style F_FPR, which sums absolute group deviations from the pooled FPR. Both are retained in the aggregate JSON.

| Arm | Test FPR (%) | ids8 worst AUC (%) | ids8 FPR gap | gender×race FPR gap | age FPR gap | skin FPR gap |
|---|---:|---:|---:|---:|---:|---:|
| R0_frozen_bce | 10.5333 | 92.2444 | 16.6398 | 14.2410 | 8.0651 | 2.6057 |
| R1_partial_bce | 10.3349 | 92.5481 | 13.9732 | 13.4879 | 11.1242 | 2.1582 |
| R2_partial_dro | 10.0926 | 92.4444 | 11.9776 | 13.0742 | 10.8491 | 3.1799 |
| RN_partial_dro_nochange_pilot | 9.4755 | 90.8593 | 11.5254 | 10.7615 | 9.9931 | 3.8692 |
| RD_partial_dro_age_pilot | 10.0705 | 93.2667 | 11.9776 | 11.2959 | 10.8884 | 2.6538 |

## Interpretation and uncertainty

RN/RD are diagnostics using six reviewed TRAIN sources, not a broad demographic study. RD does not show a uniform fairness benefit over R2: on FF++ test the ids8 FPR gap is unchanged at 11.9776pp; gender×race and skin FPR gaps decrease, but overall AUC falls from 98.3962% to 98.1965%. Worst-group AUC, FNR gaps and other axes must also be inspected. These are single-seed observations, not significance claims.

Transfer performance is weak, especially on public FairForge-2. Scores were not inverted, recalibrated on test, or used to change the frozen checkpoints/configuration. The separate FF2 original-TRAIN BCE plan was locked before these FF2 transfer results were viewed. A future FF2 in-domain result answers a different question and must retain this transfer result.

1,000-replicate percentile CIs use connected recorded-contributor components, seed 17. All group/family metrics, eligible groups and valid replicate counts are in [aggregate.json](aggregate.json). For FF++ test: R2 overall AUC 95% CI 97.8585–98.8959%; RD 97.6269–98.7197%. These intervals quantify dataset uncertainty, not seed variability or paired method differences. Sparse-group bootstrap replicates can be undefined and are omitted explicitly; for age gaps only 607/1,000 replicates are valid. Hidden donor links and proxy identities limit independence.

Both stored validation-calibrated and fixed probability 0.5 metrics are reported. Validation calibration achieved 409/4,092 false positives (9.9951%) for all arms. Actual test FPR differs by dataset; no requested test FPR is guaranteed. Some external benchmark demographic fields are unavailable, so unsupported subgroup comparisons remain undefined.

Native DFDC contains 1,476 exact duplicate excess rows. Their benchmark weighting was preserved through the previously audited unique index; conflicting metadata is rejected. Public FF2 age and skin labels are automated proxies, not human ground truth.

This report does not establish direct protocol equivalence with Ding or an overall fairness win. The prior CESel-MA gate remains FAIL. Learned gates are weights; signed local contributions are not validated forgery masks. No forensic-sensitivity or spatial-map-consistency training is represented.

## Execution evidence

Three fixture tests passed on CPU (7.089s), gpu01 CUDA (11.177s), and gpu02 CUDA (10.550s). Tests cover official-exporter agreement, frozen model/threshold/checkpoint state, changed-label calibration isolation, manifest hash rejection and preservation/rejection of native DFDC duplicate cases. These are software checks, not synthetic fairness evidence.

CPU command: `CES_EVAL_SMOKE_DEVICE=cpu /home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z/runtime/test_frozen_evaluation.py`.

CUDA commands: the same script with `CES_EVAL_SMOKE_DEVICE=cuda`, launched through `/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z/runtime/gpu_eval_worker.py --stage SMOKE_gpu01` and `--stage SMOKE_gpu02` on the corresponding SSH host. Full commands, bindings and logs are in [the execution package](../../proposals/upgrade_tests_s17_20261101T040407Z/README.md).

Five evaluation workers used bounded shared GPUs over the explicitly authorized direct-SSH route, with no claimed PBS allocation. All five exited zero after approximately 677 seconds. No other jobs were stopped. The finite publication observer exited naturally after all 25 evaluations completed; it cannot launch, retry, train, generate edits, or modify calibration.

Cluster evaluation root: `/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z`.

Cluster training root: `/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z`.

[Separate FF2 in-domain split and prepared plan](../../proposals/ff2_indomain_s17_20261101T042828Z/README.md).
