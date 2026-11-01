# CES-FL seed17 — authorized test evaluation

Observed UTC: **2026-11-01T04:35:47.096824+00:00**. All five training runs completed; these are inference/uncertainty tasks.

**FairForge-2 below is FF++-trained transfer testing.** A separate FairForge-2 public in-domain training plan uses its own TRAIN/VAL/TEST.

Selected checkpoint hashes/epochs and validation thresholds were frozen before testing. No test-based calibration, model selection, gradient update or source/configuration change is performed.

| Arm | Dataset | State | Test AUC (%) | Actual test FPR (%) | ids8 FPR gap (pp) |
|---|---|---|---:|---:|---:|
| R0_frozen_bce | ffpp | DONE | 98.35 | 10.53 | 16.64 |
| R0_frozen_bce | dfdc | DONE | 59.47 | 32.93 | — |
| R0_frozen_bce | celebdf | DONE | 74.33 | 64.69 | — |
| R0_frozen_bce | dfd | DONE | 82.36 | 42.63 | — |
| R0_frozen_bce | fairforge2 | DONE | 41.68 | 57.14 | 25.77 |
| R1_partial_bce | ffpp | DONE | 98.40 | 10.33 | 13.97 |
| R1_partial_bce | dfdc | DONE | 58.97 | 31.57 | — |
| R1_partial_bce | celebdf | DONE | 74.21 | 65.50 | — |
| R1_partial_bce | dfd | DONE | 81.81 | 42.59 | — |
| R1_partial_bce | fairforge2 | DONE | 41.87 | 59.57 | 25.68 |
| R2_partial_dro | ffpp | DONE | 98.40 | 10.09 | 11.98 |
| R2_partial_dro | dfdc | DONE | 59.52 | 28.90 | — |
| R2_partial_dro | celebdf | DONE | 74.67 | 68.91 | — |
| R2_partial_dro | dfd | DONE | 82.12 | 43.86 | — |
| R2_partial_dro | fairforge2 | DONE | 42.83 | 60.25 | 25.16 |
| RN_partial_dro_nochange_pilot | ffpp | DONE | 97.85 | 9.48 | 11.53 |
| RN_partial_dro_nochange_pilot | dfdc | DONE | 59.67 | 26.36 | — |
| RN_partial_dro_nochange_pilot | celebdf | DONE | 71.01 | 61.32 | — |
| RN_partial_dro_nochange_pilot | dfd | DONE | 81.53 | 33.89 | — |
| RN_partial_dro_nochange_pilot | fairforge2 | DONE | 38.83 | 59.07 | 25.07 |
| RD_partial_dro_age_pilot | ffpp | DONE | 98.20 | 10.07 | 11.98 |
| RD_partial_dro_age_pilot | dfdc | DONE | 59.86 | 24.93 | — |
| RD_partial_dro_age_pilot | celebdf | DONE | 72.80 | 60.55 | — |
| RD_partial_dro_age_pilot | dfd | DONE | 82.28 | 40.14 | — |
| RD_partial_dro_age_pilot | fairforge2 | DONE | 39.53 | 58.30 | 25.51 |

Completed evaluations: **25/25**. CI_PENDING means inference finished but uncertainty is still running.

[Complete aggregate evidence, all group/family metrics, CIs, hashes and process IDs](../../CES_FL/results/upgrade_tests_s17_20261101T040407Z/aggregate.json).

Inference uses the unchanged875bce model source. The external evaluation wrapper was tested on CPU and both CUDA nodes: three tests each, covering official-exporter agreement, immutable state/thresholds and exact native duplicate handling. Every selected checkpoint also replays validation before its test loop (AUC tolerance1e-5; calibration remains frozen).

Locked TEST row counts: FF++25,401; DFDC22,857; Celeb-DF28,458; DFD9,385; FairForge-287,564. Native DFDC includes1,476exact duplicate excess rows; their original weighting is preserved through the already audited unique index. Conflicting metadata is rejected.

Both fixed probability0.5 and stored validation-real empirical-budget thresholds are reported. Actual test FPR is measured; the requested validation target does not guarantee a test FPR. Undefined/unsupported subgroup metrics remain undefined.

CIs use1,000fixed-score bootstrap replicates of connected recorded-contributor components, seed17. Fewer than two components yields no independent-component CI. Hidden donors, proxy identities, sparse groups, single-seed training and multiple comparisons limit conclusions.

RN/RD used only six reviewed training sources. A fairness win or direct equivalence with Ding is not established merely by a test AUC or a single favorable group metric. The historical gate remains FAIL.

Execution follows the authorized SSH route on gpu01/gpu02, bounded shared VRAM,10%PyTorch allocator cap and no claimed PBS allocation. Existing jobs are untouched.

The finite24h observer reads/publishes only these already launched tasks. It cannot launch/retry/cancel training, generate images or change the calibration. Images, weights and row-level predictions remain private on cluster storage.

Cluster root: /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z
