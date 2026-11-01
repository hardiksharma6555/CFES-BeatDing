# Frozen upgrade test evaluation: actual execution evidence

Observed UTC: 2026-11-01T04:42:46.223256+00:00. All25 tasks verified PASS; no new training or image generation.

[Results and limitations](../../results/upgrade_tests_s17_20261101T040407Z/REPORT.md) · [Aggregate metrics and CIs](../../results/upgrade_tests_s17_20261101T040407Z/aggregate.json).

Scientific model source is the unchanged detached 875bce checkout; the external inference wrapper and three fixture tests are included here. Source/configuration/checkpoint/manifests were locked before testing. Private predictions remain on cluster storage and are not published.

| Arm | Host | SSH worker PID | Exit |
|---|---|---:|---:|
| R0_frozen_bce | gpu01 | 391865 | 0 |
| R1_partial_bce | gpu02 | 982090 | 0 |
| R2_partial_dro | gpu01 | 391995 | 0 |
| RN_partial_dro_nochange_pilot | gpu02 | 982186 | 0 |
| RD_partial_dro_age_pilot | gpu01 | 392181 | 0 |

These are direct-SSH process IDs, not PBS job IDs. GPU UUIDs, commands, actual binding, source/driver hashes and shared-memory limits are in the runtime process receipts. The user authorized direct SSH on gpu01/gpu02; no PBS allocation is claimed and other jobs were not stopped.

Read software_checks/cpu.log and logs/SMOKE_gpu01.ssh.log /SMOKE_gpu02.ssh.log: three tests passed on each device. Python3.12 emitted a multiprocessing fork deprecation warning; execution completed without deadlock. The five full validation replay/evaluation chains subsequently exited zero.

The observer published the completed aggregate and exited naturally. Node wall clocks lag the management clock by about 648 seconds; compare worker monotonic durations or management observations, not cross-host wall timestamps.

Do not rerun these commands into their existing output directories. Reports and receipts retain exact absolute cluster paths for reproducibility. A portable replay requires deliberately mapped data paths and verified hashes.
