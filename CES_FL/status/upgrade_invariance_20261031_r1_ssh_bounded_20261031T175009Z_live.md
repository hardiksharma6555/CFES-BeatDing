# CES-FL controlled upgrade — direct SSH training

Observed UTC: **2026-10-31T20:40:29.471056+00:00**. Snapshot from actual GPU processes and saved artifacts.

**CUDA smoke passed on gpu01 and gpu02: eight tests per node.** Each production arm also repeats input preflight and CUDA smoke.

Following the latest direct-SSH instruction, these processes have **no PBS allocation or PBS job ID**. The duplicate queued smoke44906 was cancelled before execution; existing jobs44861/44881 were not stopped.

| Arm | Host | GPU index | Model PID | State | Epochs complete | Best validation AUC (%) |
|---|---|---:|---:|---|---:|---:|
| R0_frozen_bce | gpu01 | 2 | — | COMPLETED_CHECKPOINT_REVIEWED | 10/10 | 98.1949 |
| R1_partial_bce | gpu02 | 6 | — | COMPLETED_CHECKPOINT_REVIEWED | 10/10 | 98.2215 |
| R2_partial_dro | gpu01 | 1 | — | COMPLETED_CHECKPOINT_REVIEWED | 10/10 | 98.2358 |
| RN_partial_dro_nochange_pilot | gpu02 | 5 | — | COMPLETED_CHECKPOINT_REVIEWED | 10/10 | 97.5249 |
| RD_partial_dro_age_pilot | gpu01 | 4 | — | COMPLETED_CHECKPOINT_REVIEWED | 10/10 | 98.1618 |

[Aggregate evidence, logs, hashes, validation subgroup metrics and pair exposure](../../CES_FL/results/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z/aggregate.json).

The exact approved source remains commit875bce025084871438ef799ca9086985c4375672 from [the supervisor fork](https://github.com/hardiksharma6555/CFES-BeatDing/tree/875bce025084871438ef799ca9086985c4375672). Scientific configuration/input hashes, seed17,10epochs, original sampling, validation-AUC selection and validation-only FPR calibration remain fixed.

Operational change: bounded GPU sharing via SSH, at least24GiB free at binding, shared project leases, and a0.1PyTorch allocator fraction (about8GiB on80GB H100) through a separately recorded sitecustomize hook. No existing workload is evicted. The memory cap affects execution resources; it does not change the model or loss. Shared GPU contention affects elapsed time.

RN/RD use six reviewed sources only. They are small-pool diagnostics, not a300-source study or proof of fairness. Test sets are not evaluated or tuned against. The historical CESel-MA gate remains FAIL.

The finite24h observer only reads/publishes these already launched processes. It cannot launch, retry or cancel jobs. Read observation times. Checkpoint review checks strict reload and recorded selection/calibration, not an independent validation replay.

Output root: /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z
Model source: /home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672
