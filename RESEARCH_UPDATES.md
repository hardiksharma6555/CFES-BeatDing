# Shared research updates and reviewer index

Repository: https://github.com/hardikbha/CFES-BeatDing

This public repository is the shared review record for the cluster experiments. Anyone with the link can read it. Writing changes requires GitHub permission; training execution separately requires cluster access and an approved configuration.

## Where to look

| File or folder | What it provides |
|---|---|
| [Completed upgrade test report](CES_FL/results/upgrade_tests_s17_20261101T040407Z/REPORT.md) | All five ten-epoch arms and25 evaluations verified; validation/test metrics, CIs, mixed fairness and weak transfer findings. |
| [FairForge-2 public split and in-domain plan](CES_FL/proposals/ff2_indomain_s17_20261101T042828Z/README.md) | Separate public TRAIN/VAL/TEST audit, seed 17 BCE pretraining configuration and paired-control blockers; not launched. |
| [Completed direct-SSH upgrade training](CES_FL/status/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z_live.md) | Final ten-epoch completion observations for the five arms across gpu01/gpu02. |
| [Direct-SSH execution receipt](CES_FL/proposals/upgrade_invariance_20261031_r1/execution_ssh_20261031T175009Z/README.md) | Exact source/configuration hashes, CUDA logs, process IDs, operational memory policy and preserved previous attempts. |
| [Superseded PBS recovery](CES_FL/status/upgrade_invariance_20261031_r1_recovery_20261031T165803Z_live.md) | Historical queued smoke 44906; cancelled before execution when the user requested direct SSH. |
| [Recovery diagnosis and receipt](CES_FL/proposals/upgrade_invariance_20261031_r1/execution_recovery_20261031T165803Z/README.md) | Initial missing-Git failure, checked operational correction, repeated input preflights and exact recovery artifacts. |
| [Initial upgrade attempt](CES_FL/status/upgrade_invariance_20261031_r1_20261031T135409Z_live.md) | Failed smoke 44880; all five training submissions stopped. Includes the recovered failure diagnosis. |
| [Upgrade execution receipt](CES_FL/proposals/upgrade_invariance_20261031_r1/execution_20261031T135409Z/README.md) | Exact fetched commit, detached worktree, input/CPU checks, unchanged configurations and operational execution code/logs. |
| [Current benchmark observations](CES_FL/status/benchmark_s17_20261031T112401_live.md) | Timestamped scheduler states, recorded training progress and emerging test results from the authorized seed 17 batch. |
| [FairForge-2 in-domain audit](FAIRFORGE2_INDOMAIN_RESULTS_20261031.md) | Verified historical train-on-FF2/test-on-FF2 results, internal versus public split distinctions, and exact cluster evidence paths. |
| [CES-FL README](CES_FL/README.md) | Current-method overview and links to experiment records. |
| [Source](CES_FL/src/) and [configurations](CES_FL/configs/) | Reviewable implementation and recorded configurations; each result must retain its actual source/configuration identity. |
| [Completed seed 17 evidence](CES_FL/results/completed_seed17/) | Historical pilot histories and verification records. These are preserved when new results appear. |
| [Experiment proposals](CES_FL/proposals/) | Explicit requests, dependencies, configurations and launch evidence. A Git push does not launch training. |
| [Supervisor workflow](CES_FL/WORKFLOW.md) | How to review code, propose changes and hand off an exact commit to the cluster operator. |
| [Supervisor/LLM prompt](CES_FL/SUPERVISOR_PROMPT.md) | Starting instructions for another reviewer or research assistant. |

## Publication policy

Publish scoped updates after verified launches, failures, checkpoint selection, evaluation, code/configuration changes and review decisions. Record the experiment ID, observation time, source/configuration hashes, dataset/split identity, seed, scheduler state, checkpoint/result paths, measured metrics and unresolved blockers as applicable. Preserve prior results and label incomplete or unchecked work explicitly.

Keep paper values, local reproductions, historical methods and the current CES-FL method distinct. Separate validation from test and FF2 in-domain results from FF++-to-FF2 transfer. Do not report a fairness improvement merely from detection AUC.

Publish aggregate metrics and relevant textual evidence. Dataset images, private row-level records, credentials and binary weights stay on the cluster; publish permitted paths and hashes rather than those files.

## Freshness and automatic updates

The already authorized benchmark has a finite observation/publication monitor. At **2026-10-31 11:47 UTC**, PID 1762946 was present and its log showed a successful GitHub publication at 11:47:11 UTC. It observes only that submitted batch, stops on completion/failure or after 24 hours, and does not submit new experiments. Its log is `/home/sachin.chaudhary/ces_fl_pipeline/experiments/benchmark_s17_20261031T112401/logs/github_monitor.log`.

The initial controlled-upgrade executor PID 3531362 stopped after smoke 44880 failed before CUDA because Git was absent on the compute nodes. After the user's later direct-SSH instruction, queued smoke 44906 and its submission executor were superseded without stopping existing jobs 44861/44881. Both-node SSH CUDA smoke passed, and all five exact arms subsequently completed ten epochs. Training observer PID 3109255 exited after completion. Following separate user authorization, all 25 frozen-checkpoint evaluations completed; evaluation observer PID 709332 published the final aggregate and exited naturally. These finite observers do not launch additional experiments, retry, cancel or generate images. The new FF2 in-domain plan remains unlaunched.

GitHub displays the last successfully published observation, not a direct connection to the scheduler. Read the timestamp; a stale RUNNING entry does not prove a job is still active. Outside these finite request-specific processes, updates are made when the researcher/assistant checks and publishes the evidence; indefinite background monitoring is not configured.

## Update record

- **1 November 2026, Ding/public FF2 baseline check:** No matching public-only Ding retrain found. Rechecked the full-split run log, manifest counts and saved-score AUCs; the historical 99.3573% belongs to 182,884 TEST images. Current archived GDA still matches the previously tested source with a detached fairness term. [Evidence and comparison recommendation](CES_FL/proposals/ff2_indomain_s17_20261101T042828Z/DING_BASELINE_STATUS_20261101.md). No architecture/configuration change or training launch.
- **1 November 2026:** Verified all five training exits/checkpoint hashes and independent validation replays. Completed the 25 authorized frozen-checkpoint evaluations with CPU and both-node CUDA software checks; published group/family metrics, CIs and actual achieved test FPRs. Current FF2 results are transfer tests. Verified the separate public FF2 TRAIN/VAL/TEST membership and recorded-contributor separation, and prepared the original-image seed 17 BCE pretraining configuration without launching it. No model/loss/gate changes or new images.
- **31 October 2026, 17:58 UTC:** Both-node direct-SSH CUDA smoke passed (eight tests each). All five approved configurations entered production training: R0/R2/RD on gpu01 and R1/RN on gpu02. Actual GPU processes and provenance were checked. The direct-SSH shared-VRAM route is recorded explicitly; it has no new PBS allocation. Scientific settings remain fixed and existing jobs were preserved.
- **31 October 2026, 17:05 UTC:** Diagnosed initial smoke 44880: the wrapper could not execute Git on the compute node. Preserved its recovered traceback, checked a shared Git binary on both nodes, reran all five input preflights, and submitted replacement smoke 44906. It is queued; no upgrade training or new validation/fairness results exist yet. The scientific source/configuration/input hashes remain fixed.
- **31 October 2026:** Fetched and verified upgrade commit 875bce025084871438ef799ca9086985c4375672 in a new detached worktree. All five real-input preflights and19target-environment CPU tests passed. Submitted initial smoke-only job44880.mgmt01; at submission it was queued and CUDA remained unverified. The guarded executor will publish actual training IDs only after actual smoke success.
- **31 October 2026:** Published the verified FairForge-2 in-domain audit. Historical internal/full FF2 results include Ding classification 99.26% AUC, Ding SFD+GDA 99.36%, older CFES detection-only 97.60% and older CFES 98.10%. Separately, public-only CFES-D reached 98.26%; completed matched public-split Ding/DAW/DAG/PG retrains were not found. See the audit for limitations and exact artifacts.
- **31 October 2026:** Added this shared index and README links; verified public visibility and the existing benchmark publisher. No new training, generation, job cancellation or research-protocol change was made for this documentation update.
