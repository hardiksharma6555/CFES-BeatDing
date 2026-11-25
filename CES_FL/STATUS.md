## Lightweight deployment preparation — 25 November 2026

The new [student study](proposals/lh_lite_feasibility_20261125_r1/README.md) has real FF++ CPU preparation/backward PASS, verified teacher/input hashes, and six synthetic lifecycle plus four numerical loss tests. The bounded CPU check completed with exit0. GPU training is NOT running; up to four new GPU-hours on one H100 has been requested and is pending. No lightweight accuracy, fairness-retention, or ESP32 deployment result is claimed. This dated preparation does not update other studies' process state.

## FairForge-2 baseline preparation — 2026-11-03T11:33:27.003004+00:00

## Full L_H channel-selection extension — 10 November2026

[Proposal](proposals/lh_channel_selection_20261110_r1/README.md) · [standalone Python](src/ces_fl_lh_channel_selection.py) · [full diagram](proposals/lh_channel_selection_20261110_r1/figures/L_H_Channel_Selection_Full.png) · [operator instructions](proposals/lh_channel_selection_20261110_r1/GPU_OPERATOR_PROMPT.md). Fourteen CPU unit tests, original and masked full-L_H synthetic lifecycle checks and objective-preservation checks passed. The new fixed mask preserves hierarchical error control and all matched-response losses. Production inputs, human qualification, CUDA and budget checks remain pending. No production training launched, no new results or TEST access. Historical observations below are preserved.

New [baseline suite and operator prompt](proposals/ff2_baseline_suite_s17_20261103T110324Z/GPU_OPERATOR_PROMPT.md): user approved72 additional GPU-hours, maximum3 allocated GPUs. Native BCE/DAG/DAW CPU checks pass; CUDA/input checks and training are pending. PG-FDD/FairAdapter adaptations remain conditional; RSEF source unavailable. No jobs launched here; existing jobs/results preserved. This is a preparation observation, not live cluster status. Published to the writable fork because upstream push permission remains unavailable.

# Current CES-FL status

## Preparation snapshot — 10 November 2026

`lh_gradient_conflict_20261110_r1` is prepared for operator execution: CPU tests12/12, reused-helper checks9/9, full-width ResNet backward and projection-disabled/native update parity PASS. Private binding, CUDA, repeated TRAIN diagnostics and real cost gates are pending. No local GPU allocation or training was launched. [Scope and current dependencies](proposals/lh_gradient_conflict_20261110_r1/README.md) · [Operator handoff](proposals/lh_gradient_conflict_20261110_r1/GPU_OPERATOR_PROMPT.md).

The newer external source-tail cohort is completed at `aa074b2`, with no meaningful fairness benefit established and primary uncertainty inconclusive. It is not this L_H experiment and must not be rerun by this handoff. All previous results/protocols are preserved. Older RUNNING/pending entries below refer to their recorded dates; this branch provides no unattended monitor.

## Plain Xception head ablation — 3 November 2026

[Project memory and research blueprint](PROJECT_MEMORY.md) · [Two-arm proposal](proposals/plain_head_s17_20261103T101418Z/README.md) · [GPU operator prompt](proposals/plain_head_s17_20261103T101418Z/GPU_OPERATOR_PROMPT.md). New standalone Xception+GAP+linear source:20,809,001 total /6,790,433 exit-policy trainable parameters. CPU synthetic smoke and targeted checks pass; CUDA/input checks and both real-data runs are pending. Existing H and completed results are preserved. This publication launches no jobs.

## One hierarchical original-real risk pilot — 3 November 2026

[Current H process/results](results/hierarchical_real_risk_s17_20261103_r1/STATUS.md) · [Failure diagnosis](proposals/hierarchical_real_risk_s17_20261103_r1/FAILURE_DIAGNOSIS.md) · [Frozen single-candidate proposal](proposals/hierarchical_real_risk_s17_20261103_r1/README.md). E/F finished and rejected their fixed guards. H is seed17/FF++ only and changes only the joint-event real fairness summand; no new Ding, FairForge-2 or TEST computation. The retained bootstrap support requirement makes numerical success at best INCONCLUSIVE. Historical snapshots below remain preserved.


## Seed-17 support/error-control pilots — 3 November 2026

Two CES-FL-only candidates are actually training on FF++ TRAIN/validation after fresh CPU/CUDA, real-input and recipe-review gates. [Live process, metric, exposure and budget evidence](results/support_warmup_s17_20261103_r1/STATUS.md) · [Audit and actual execution report](proposals/support_warmup_s17_20261103_r1/AUDIT_AND_RUN_REPORT.md) · [Frozen proposal and decision rules](proposals/support_warmup_s17_20261103_r1/README.md). No new Ding-model computation, new FairForge-2 experiment or TEST inference. This is a separate branch; historical snapshots below remain preserved.


## Latest verified milestone — 1 November 2026

All five controlled-upgrade arms completed ten epochs. All 25 authorized frozen-checkpoint test evaluations completed, with unchanged validation thresholds and exact validation-AUC replay. [Results and limitations](results/upgrade_tests_s17_20261101T040407Z/REPORT.md) · [Completed test observations](status/upgrade_tests_s17_20261101T040407Z_live.md).

A separate [FairForge-2 public in-domain plan](proposals/ff2_indomain_s17_20261101T042828Z/README.md) is prepared with verified TRAIN/VAL/TEST manifests. No training for that new plan has started. FairForge-2 scores in the completed batch are FF++-trained transfer results. The older snapshot below is preserved and does not represent the current five-arm batch.

## Preserved original pilot snapshot — 31 October 2026

Last observed (UTC): **2026-10-31T06:11:36.029000+00:00**.

Point-in-time observation. No monitoring daemon or automatic GitHub publication.

Experiment: `seed17_encoder_controls_20261030`; seed 17.

| Arm | Observed state | Selected epoch | Validation AUC (%) | Real test evaluated |
|---|---|---:|---:|---|
| P_pretrain | COMPLETED_VERIFIED | 9 | 98.1711 | False |
| A_imagenet_frozen | COMPLETED_VERIFIED | 8 | 67.9134 | False |
| B_detection_frozen | COMPLETED_VERIFIED | 7 | 98.1949 | False |
| C_detection_invariance | COMPLETED_VERIFIED | 3 | 98.0661 | False |

Review-source SHA256: `0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de`.

Review source matches the completed experiment source: **True**.
If this becomes False, historical CPU/CUDA receipts and results do not verify the newly edited code.

## Interpretation and pending work

Read [the dated handoff](../docs/ces_fl/2026-10-31/HANDOFF_2026-10-31.md) for the historical fairness conclusion. This status page does not infer fairness from overall AUC.

The current experiment is invariance-only. Full forensic selectivity is not validated. The historical CESel-MA gate remains FAIL.

Table1 requires real FF++ test evaluation; Table2 requires DFDC/Celeb-DF/DFD adapters and evaluation; Table3 requires a separately prepared current-method ResNet-50 experiment. FairForge-2 evaluation/training is a separate phase. No training is launched by refreshing this page.

Explicit evaluation records tracked: **0**. Their detailed evidence/status appears in [current.json](status/current.json). Presence of files is not verification of protocol compliance.

Checkpoint hashes above are from existing verification receipts; checkpoint bytes were not rehashed during this refresh. Recorded progress without verified completion is not proof a process is still alive.

## Resource observation

```json
{
  "scheduler": {
    "exit_code": 0,
    "stdout": "mgmt01: \n                                                            Req'd  Req'd   Elap\nJob ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n43937.mgmt01    sachin.* workq    v3_sched2  13813*   1   8    --  240:0 R 163:0\n44835.mgmt01    sachin.* gpu      symtl2     344307   1   8  200gb 10:00 R 06:49\n44836.mgmt01    sachin.* gpu      poolg      21698*   1   8  200gb 12:00 R 03:42\n44842.mgmt01    sachin.* workq    symtl      38286*   1   8  200gb 12:00 R 00:48\n44847.mgmt01    sachin.* gpu      cesX_s17      --    1   8   32gb 12:00 Q   -- \n44848.mgmt01    sachin.* gpu      cesP_s17      --    1   8   32gb 12:00 Q   -- \n44849.mgmt01    sachin.* gpu      cesB_s17      --    1   8   32gb 12:00 H   -- \n44850.mgmt01    sachin.* gpu      cesC_s17      --    1   8   32gb 12:00 H   -- \n44851.mgmt01    sachin.* gpu      cesA_s17      --    1   8   32gb 12:00 Q   --",
    "stderr": ""
  },
  "gpu01": {
    "exit_code": 0,
    "stdout": "0, NVIDIA H100 80GB HBM3, 81559 MiB, 33375 MiB, 100 %\n1, NVIDIA H100 80GB HBM3, 81559 MiB, 49629 MiB, 98 %\n2, NVIDIA H100 80GB HBM3, 81559 MiB, 49443 MiB, 89 %\n3, NVIDIA H100 80GB HBM3, 81559 MiB, 26904 MiB, 71 %\n4, NVIDIA H100 80GB HBM3, 81559 MiB, 64777 MiB, 100 %\n5, NVIDIA H100 80GB HBM3, 81559 MiB, 49763 MiB, 100 %\n6, NVIDIA H100 80GB HBM3, 81559 MiB, 49064 MiB, 92 %\n7, NVIDIA H100 80GB HBM3, 81559 MiB, 76137 MiB, 0 %",
    "stderr": ""
  },
  "gpu02": {
    "exit_code": 0,
    "stdout": "0, NVIDIA H100 80GB HBM3, 81559 MiB, 77585 MiB, 0 %\n1, NVIDIA H100 80GB HBM3, 81559 MiB, 76577 MiB, 0 %\n2, NVIDIA H100 80GB HBM3, 81559 MiB, 76577 MiB, 0 %\n3, NVIDIA H100 80GB HBM3, 81559 MiB, 76575 MiB, 0 %\n4, NVIDIA H100 80GB HBM3, 81559 MiB, 54729 MiB, 0 %\n5, NVIDIA H100 80GB HBM3, 81559 MiB, 54711 MiB, 0 %\n6, NVIDIA H100 80GB HBM3, 81559 MiB, 77028 MiB, 100 %\n7, NVIDIA H100 80GB HBM3, 81559 MiB, 71352 MiB, 100 %",
    "stderr": ""
  }
}
```

Read [WORKFLOW.md](WORKFLOW.md) before proposing or launching a run. See [README.md](README.md) for code, configuration and result paths.

## Additional authorized benchmark batch

The stage table above describes the original pilot. Current batch: [benchmark_s17_20261031T112401](https://github.com/hardikbha/CFES-BeatDing/blob/main/CES_FL/status/benchmark_s17_20261031T112401_live.md). Read that separately timestamped page for its submitted jobs and actual progress.
