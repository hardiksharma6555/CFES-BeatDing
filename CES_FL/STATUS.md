# Current CES-FL status

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
