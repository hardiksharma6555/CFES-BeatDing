# CES-FL — supervisor review and training handoff

## Lightweight deployment study — 25 November 2026

[Study and measured sizes](proposals/lh_lite_feasibility_20261125_r1/README.md) · [Trainer](src/ces_fl_lite_training.py) · [GPU operator prompt](proposals/lh_lite_feasibility_20261125_r1/GPU_OPERATOR_PROMPT.md). Three untrained resource prototypes fit below10MB, including a224×224 MobileNetV2 at2.32MB INT8. Six synthetic lifecycle/guard tests, four numerical loss tests and real FF++ CPU preparation/backward checks passed. The teacher and original TRAIN/validation inputs are bound. New GPU allowance and CUDA checks remain pending; no lightweight task-accuracy or ESP32 execution result exists. This is separate from historical FF2 and CVPR studies.

## Full L_H channel-selection extension — 10 November2026

[Proposal](proposals/lh_channel_selection_20261110_r1/README.md) · [standalone Python](src/ces_fl_lh_channel_selection.py) · [full diagram](proposals/lh_channel_selection_20261110_r1/figures/L_H_Channel_Selection_Full.png) · [operator instructions](proposals/lh_channel_selection_20261110_r1/GPU_OPERATOR_PROMPT.md). Fourteen CPU unit tests, original and masked full-L_H synthetic lifecycle checks and objective-preservation checks passed. The new fixed mask preserves hierarchical error control and all matched-response losses. Production inputs, human qualification, CUDA and budget checks remain pending. No production training launched, no new results or TEST access. Historical observations below are preserved.

## Bounded provenance and gradient-conflict handoff — 10 November 2026

[New proposal](proposals/lh_gradient_conflict_20261110_r1/README.md) · [GPU operator instructions](proposals/lh_gradient_conflict_20261110_r1/GPU_OPERATOR_PROMPT.md) · [CPU checks](proposals/lh_gradient_conflict_20261110_r1/CHECKS.md).

Prepared a metadata-only FF2 provenance inventory,20-event zero-update FF++ gradient diagnostic and one conditional alignment-gradient candidate. Twelve new tests, nine reused-helper checks and a full-width CPU backward/parity check passed. Cluster binding/CUDA/profile/diagnostic gates remain pending; no GPU job launched here. Native model, H0 configuration and completed results are preserved. At most one GPU and2aggregate GPU-hours within a verified remaining original L_H allowance, not a new grant. No TEST/CAL, new Ding, CLIP or source-tail run.

Newer externally completed work is linked rather than silently merged: [source-tail results aa074b2](https://github.com/hardikbha/CFES-BeatDing/blob/aa074b2024a9730a9d66f6020440d2137e7cd6f1/CES_FL/results/source_tail_training_20261109_r1/REPORT.md) found no meaningful fairness benefit; [source audit70916d0](https://github.com/hardikbha/CFES-BeatDing/blob/70916d030c9da5fba6063f83e4bf3af1b59bc6f5/CES_FL/results/ff2_source_stratified_audit_20261109_r1/REPORT.md) remains blocked/inconclusive for its specified estimands. Preserve those cohorts. Historical observations below are not live status.

## Repeated alignment diagnostic and C1 follow-up — 5 November 2026

[Proposal](proposals/lh_alignment_ablation_20261105_r1/README.md) · [GPU operator prompt](proposals/lh_alignment_ablation_20261105_r1/GPU_OPERATOR_PROMPT.md) · [CPU checks](proposals/lh_alignment_ablation_20261105_r1/CHECKS.md). The new helper binds the completed H0/H1/A1 evidence and all352existing H1 review cases, profiles20fixed TRAIN events under identity/resize for H0/H1, and conditionally permits one three-epoch native-arm-C ablation. The scientific trainer and H1 config are unchanged; only response alignment is removed in the candidate. CPU tests pass; actual cluster diagnostics and candidate training remain pending. Maximum one GPU/one aggregate GPU-hour within the existing ledger; no new TEST evaluation or FairForge-2 review change.

## Short nuisance/sensitivity pilot — 4 November 2026

[Runnable four-arm proposal](proposals/lh_nuisance_pilot_20261104_r1/README.md) · [GPU operator prompt](proposals/lh_nuisance_pilot_20261104_r1/GPU_OPERATOR_PROMPT.md) · [Actual CPU checks](proposals/lh_nuisance_pilot_20261104_r1/CHECKS.md). New ResNet A0/H0/A1/H1 three-epoch cohort tests original versus matched mild processing, retaining original losses and initialization. Synthetic counterexamples demonstrate possible scale/texture shortcuts; trained-checkpoint diagnostics and production GPU runs are pending. Maximum two devices/three aggregate GPU-hours within the remaining study ledger. Publication does not launch jobs. No new benchmark TEST use or claim of disentangled features.

## FairForge-2 external baselines — 3 November 2026

[Proposal and readiness](proposals/ff2_baseline_suite_s17_20261103T110324Z/README.md) · [GPU operator prompt](proposals/ff2_baseline_suite_s17_20261103T110324Z/GPU_OPERATOR_PROMPT.md). User authorized72 additional aggregate GPU-hours and at most3 allocated GPUs. Matched native Xception BCE/DAG-FDD/DAW-FDD code, configs and CPU checks are prepared; real input/CUDA checks and training remain pending. PG-FDD/FairAdapter source is pinned but adapters remain conditional; RSEF lacks verified source. Existing Ding and plain-head work is preserved. Publish destination is the writable `hardiksharma6555` fork; this is not a training launch.

## Plain Xception head ablation — 3 November 2026

[Project memory and research blueprint](PROJECT_MEMORY.md) · [Two-arm proposal](proposals/plain_head_s17_20261103T101418Z/README.md) · [GPU operator prompt](proposals/plain_head_s17_20261103T101418Z/GPU_OPERATOR_PROMPT.md). New standalone Xception+GAP+linear source:20,809,001 total /6,790,433 exit-policy trainable parameters. CPU synthetic smoke and targeted checks pass; CUDA/input checks and both real-data runs are pending. Existing H and completed results are preserved. This publication launches no jobs.

## One hierarchical original-real risk pilot — 3 November 2026

[Current H process/results](results/hierarchical_real_risk_s17_20261103_r1/STATUS.md) · [Failure diagnosis](proposals/hierarchical_real_risk_s17_20261103_r1/FAILURE_DIAGNOSIS.md) · [Frozen single-candidate proposal](proposals/hierarchical_real_risk_s17_20261103_r1/README.md). E/F finished and rejected their fixed guards. H is seed17/FF++ only and changes only the joint-event real fairness summand; no new Ding, FairForge-2 or TEST computation. The retained bootstrap support requirement makes numerical success at best INCONCLUSIVE. Historical snapshots below remain preserved.


## Seed-17 support/error-control pilots — 3 November 2026

Two CES-FL-only candidates are actually training on FF++ TRAIN/validation after fresh CPU/CUDA, real-input and recipe-review gates. [Live process, metric, exposure and budget evidence](results/support_warmup_s17_20261103_r1/STATUS.md) · [Audit and actual execution report](proposals/support_warmup_s17_20261103_r1/AUDIT_AND_RUN_REPORT.md) · [Frozen proposal and decision rules](proposals/support_warmup_s17_20261103_r1/README.md). No new Ding-model computation, new FairForge-2 experiment or TEST inference. This is a separate branch; historical snapshots below remain preserved.


## Actual Ding reproduction and FairForge-2 launch — 2 November 2026

[Our archived Ding reproduction versus CES-FL](results/ding_reproduced_vs_cesfl_20261102/REPORT.md) uses measured local checkpoints rather than the paper's numbers. All four TEST memberships and FF++ TRAIN/VAL memberships match exactly. Full local Ding seed7 AUC97.44/56.05/71.22/72.54%; CES-FL B/C seed17 have higher AUC on all four datasets, with disclosed differences in preprocessing, seeds, optimization, checkpoint selection and archived GDA behavior. Full fairness metrics and native aggregate files are included.

[FairForge-2 original-only pretraining request and actual checks](proposals/ff2_originals_s17_20261102_r1/README.md): original public TRAIN218,266/VAL31,107, seed17, unchanged ten-epoch configuration. Fresh input and CUDA/real backward checks PASS. Actual launch evidence and state: [execution status](proposals/ff2_originals_s17_20261102_r1/STATUS.md). This releases P only; full Ding comparison and paired arms retain their review blockers. No TEST evaluation is attached.

## Ding-format Tables 1, 2 and 3 — 2 November 2026

[Table 1: FF++](results/ding_format_tables_s17_20261102/TABLE_1.md) · [Table 2: cross-domain](results/ding_format_tables_s17_20261102/TABLE_2.md) · [Table 3: ResNet-50](results/ding_format_tables_s17_20261102/TABLE_3.md) · [Combined PDF](results/ding_format_tables_s17_20261102/TABLES_1_2_3.pdf).

All published rows are followed by the applicable completed seed17 A/B/C and upgrade test results. [Arm definitions, thresholds, provenance and limitations](results/ding_format_tables_s17_20261102/README.md). The latest 50+50 source `58773bb` has validation results only and is shown separately; it has no TEST row. This reporting update launches no training or evaluation.

[Verified epoch budgets and numerical win counts](results/ding_epochs_wins_s17_20261102/REPORT.md): reported models used ten pretraining plus ten downstream epochs; Xception B/C each exceed Ding's displayed numbers on 17/40 unique cells at stored validation thresholds. Selected checkpoint epochs are not measured first-surpass epochs. Full fairness/AUC, dataset and ResNet breakdowns are included; no overall superiority claim or new run.

## Completed original-only comparison — 2 November 2026

[Verified P/BCE/fair completion and validation fairness comparison](results/manip_response_originals_s17_20261102/REPORT.md). Both downstream arms completed50epochs with native independent replay. BCE/fair validationAUC98.45/98.49%, but primaryF_FPR and FPRgap are worse for fair; no primary fairness win. Nochange/score/response remain blocked by reviewed manipulation quartets. Qwen224/224generated, no recorded pair approvals. No TEST or FairForge-2 run.

## P50 verified complete; matched original-only baselines launched

[Current per-arm state](status/manip_response_ssh_20261101T115720Z_live.md): P50 selected epoch29, validationAUC98.405989%, exact validation replay. BCE and fairness50epoch arms launched separately on gpu02 GPUs5/4 after actual CUDA backward PASS. Paired arms remain blocked by reviewed manipulation quartets. Qwen224/224generation complete, not independently validated. No test or FairForge-2 launch. This is a dated execution observation.

**[Shareable current supervisor handoff: progress, running jobs and complete code map](handoffs/SUPERVISOR_HANDOFF_2026-11-01_MANIPULATION_RESPONSE.md)** — verified snapshot 1 November 19:41 IST; refresh native/live-status evidence before acting.

## Direct SSH: new CUDA/input checks passed; P50 running

[Actual manipulation-response stage and arm status](status/manip_response_ssh_20261101T115720Z_live.md): all six CUDA software groups passed, all101,525TRAIN/VALimages decoded/hashed, configured backward PASS. Original-only P50 launched on gpu02GPU5 (training PID1393109); BCE/fair await P50, paired arms await reviewed quartets. Qwen continues separately on gpu01. No test or FairForge-2 launch. This is a dated observation.

## Priority manipulation-response experiment

[Actual per-arm status](status/manip_response_20261101T104925Z_live.md): exact source `58773bb`, new CUDA smoke **44963.mgmt01 queued with ngpus=1**; no new training yet. Original-only P/BCE/fair can proceed after their own checks. No-change/score/response are blocked by the matched manipulation bank. Existing Qwen32 continues separately. This is a timestamped launch observation; follow the linked status for updates.

## Authorized bounded Qwen quality pilot

[Executable 32-source quality diagnostic](proposals/qwen_bounded_quality_20261101_r2/README.md): six selection/isolation checks passed; 224 outputs planned, 4–6 generation hours after GPU binding. Training and full-bank generation remain stopped; no validity approvals are inferred. Actual launch and current state are recorded on its linked status page.

## Latest continuation: pilot review and bounded revision

[New audit and proposed 32-source quality diagnostic](proposals/qwen_bounded_quality_20261101_r1/README.md). All 14 pilot outputs reverified; zero recorded approvals. No new training or generation launched. Proposed generation cost 4.33 GPU-hours before review/retries; this does not replace the full training bank. Observed UTC: 2026-11-01T09:43:09.620570+00:00.

**Start here.** [Current timestamped status](STATUS.md) · [Review/change/run workflow](WORKFLOW.md) · [Prompt for a supervisor or another LLM](SUPERVISOR_PROMPT.md)

## Completed Qwen pilot — full scaling held, 1 November 2026

All 14 outputs completed for two FF++ TRAIN originals. Measured generation 69.5 seconds/output implies approximately 10,292 GPU-hours for the prescribed full bank. No-change controls and edits show substantial reconstruction/score-shift confounds; no candidate is human-approved and no fairness improvement is established. **REVISE / DO NOT SCALE the current recipe.** No full training or full generation was launched. [Report, visual-review paths, actual costs and failures](results/cf5050_registry_recovery_20261101T081523Z/REPORT_QWEN_PILOT.md). FairForge-2 remains DO NOT RUN under its independent review.

## Direct SSH continuation — 1 November 2026

The user superseded the queued PBS route. Job 44948 was cancelled before execution. The current immutable-source run is on direct SSH: seven CUDA software checks passed on gpu02, and full TRAIN-image hashing is running before the real-batch dry-run and separate two-source Qwen pilot. [Actual current status](status/cf5050_ssh_recovery_20261101T073243Z_live.md) · [CUDA proof, operational UUID recovery and exact wrappers](proposals/counterfactual_5050_20261101_r1/execution_cf5050_ssh_recovery_20261101T073243Z/README.md). This uses bounded shared VRAM and has no claimed PBS allocation. Full training and generation remain gated; FairForge-2 remains DO NOT RUN.

## New gated FF++ study and independent FF2 stop — 1 November 2026

The pinned counterfactual 50+50 handoff is now on the cluster. Source/configuration and declared input hashes passed; allocated milestone 1 job **44948.mgmt01** is queued behind the GPU user running-job limit. CUDA/real-data smoke and the two-source Qwen pilot remain pending. [Actual job/check status](status/cf5050_20261101T065603Z_live.md) · [Exact execution receipt](proposals/counterfactual_5050_20261101_r1/execution_cf5050_20261101T065603Z/README.md).

**FairForge-2 full comparison: DO NOT RUN.** [Independent supervisor review and required fixes](proposals/ff2_supervisor_review_20261101_r1/REVIEW.md). Its earlier 10-epoch draft is not an approved full comparison, and the FF++ 50+50 protocol does not close its blockers. No new full training or image generation has started.

## Completed controlled upgrade and separate FairForge-2 plan — 1 November 2026

The five approved Xception arms R0/R1/R2/RN/RD completed ten epochs each, with successful exits and verified checkpoint reloads. All 25 newly authorized frozen-checkpoint evaluations also completed across FF++, DFDC, Celeb-DF, DFD and public FairForge-2. Independent validation replay reproduced each selected checkpoint's recorded AUC exactly. [Results, fairness metrics and limitations](results/upgrade_tests_s17_20261101T040407Z/REPORT.md) · [Full aggregate metrics and confidence intervals](results/upgrade_tests_s17_20261101T040407Z/aggregate.json) · [Evaluation execution evidence](proposals/upgrade_tests_s17_20261101T040407Z/README.md).

Scientific model source remains pinned to 875bce025084871438ef799ca9086985c4375672. Execution used the authorized direct-SSH route, bounded shared VRAM and no new PBS allocation. Existing jobs were preserved. RN/RD remain six-source diagnostics; fairness outcomes are mixed and transfer performance is weak. No uniform fairness win or direct protocol equivalence with Ding is established.

The separate [FairForge-2 public in-domain plan](proposals/ff2_indomain_s17_20261101T042828Z/README.md) preserves 218,266 TRAIN / 31,107 VAL / 87,564 TEST images. Membership and recorded-contributor overlap checks passed. A seed 17 original-image BCE pretraining configuration is prepared; no new FF2 training is running. Paired FF2 arms additionally need verified matched controls for eight reviewed sources. Age/skin labels remain automated proxies.

## Authorized benchmark batch — 31 October 2026

Five jobs were submitted under `benchmark_s17_20261031T112401`: Xception A/B/C evaluation44847; ResNet P44848; ResNet B44849 and C44850 after successful P; ResNet A44851. Each requests ngpus=1. [Follow the dedicated batch monitor](https://github.com/hardikbha/CFES-BeatDing/blob/main/CES_FL/status/benchmark_s17_20261031T112401_live.md) for actual scheduler/work states and emerging results. [Request, protocol, preflight and submission evidence](proposals/benchmark_s17_20261031T112401/). The source is pinned to commit `a11927affe394daf877e37234c21fea88cc171be`; no existing experiment is restarted.

This authorized batch has a finite login-node observation/publication monitor, which stops on completion/failure or after24h and never launches extra work. It is separate from the general manual status helper. GPU/queue waits remain possible.

## Historical original pilot

The seed 17 Xception P/A/B/C experiment completed. B (detection-pretrained encoder plus BCE) reached **98.1949% FF++ validation AUC**. C (the same encoder plus demographic invariance) reached **98.0661% validation AUC**. Fairness outcomes are mixed; a consistent fairness benefit has not been established. **These are not test results and do not establish a win over Ding.** C uses 36 reviewed Qwen variants of only six TRAIN originals and does not test forensic sensitivity.

The paragraphs in this section describe the original P/A/B/C pilot. For the later five-arm upgrade, use the completed test report above. The separate benchmark batch has its own timestamped evidence; this update does not certify uninspected ResNet jobs. The earlier preregistered CLIP CESel-MA gate remains FAIL; this exploratory branch does not change it.

## Files to review

| Location | Purpose |
|---|---|
| [STATUS.md](STATUS.md) | Last observation, current tracked stages and GPU/scheduler query results. Always read its timestamp. |
| [src/ces_fl_diagram_model_v021.py](src/ces_fl_diagram_model_v021.py) | Exact standalone model/training/evaluation code used for the completed pilot; Xception classes are embedded. |
| [configs/completed_seed17/](configs/completed_seed17/) | Actual saved P/A/B/C configurations; historical, do not overwrite. Cluster paths are intentional. |
| [results/completed_seed17/](results/completed_seed17/) | Actual run verification, epoch histories, provenance, pair-use counts and group metrics; no binary checkpoints. |
| [provenance/](provenance/) | Tested source hash, target environment and historical CPU/CUDA receipts. |
| [proposals/](proposals/) | New training requests and reviewed configurations; pushing a request does not launch it. |
| [cluster/train_one_gpu.pbs](cluster/train_one_gpu.pbs) | New-run template with explicit ngpus=1 and immutable-source checks. Syntax checked; not submitted. |
| [scripts/refresh_status.py](scripts/refresh_status.py) | Reads the tracked cluster artifacts, writes current status and preserves a dated snapshot; never trains or pushes. |
| [status/tracking.json](status/tracking.json) | Explicit cluster experiment paths/arms/evaluation directories the refresh command observes. Update deliberately for a new experiment. |
| [tests/test_refresh_status.py](tests/test_refresh_status.py) | Software tests for truthful status/freshness/error handling. |
| [Dated research handoff](../docs/ces_fl/2026-10-31/HANDOFF_2026-10-31.md) | Detailed research state, protocol, hashes, blockers and conditional estimates. |
| [Ding Tables1–3 status](../docs/ces_fl/2026-10-31/TABLES_1_2_3_STATUS_2026-10-31.md) | Published reference values with current missing cells clearly identified. |

The exact initial review-source SHA256 is `0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de`. Historical results and smoke receipts apply to that source. The status helper flags if a later edited source differs.

## Reviewing and proposing changes

The supplied supervisor account `hardikbha` was verified through the GitHub API as having **admin** permission. No invitation or access change was needed. A different account needs its own collaborator access; do not share account credentials.

The supervisor can browse the files, create a branch, change the source, and add a request/config under `proposals/<request_id>/`. Review the diff and push or merge it. The cluster operator then fetches the exact commit and launches only that reviewed request from a detached worktree. See [WORKFLOW.md](WORKFLOW.md). GitHub access does not grant cluster access, and Git pushes do not automatically run training.

## Keeping progress current

Run this on the cluster, from the Git checkout. Without cluster access, read the last pushed snapshot instead of refreshing against missing local paths:

```bash
python CES_FL/scripts/refresh_status.py --resources
git diff -- CES_FL/STATUS.md CES_FL/status/
git add -- CES_FL/STATUS.md CES_FL/status/
git diff --cached --check
git commit -m "Update CES-FL verified status"
git push origin main
```

Run this after a launch, a failure, checkpoint selection, evaluation, review decision or code/config change. Include newly verified result files and request decisions in the same scoped update when needed. The refresh helper does not copy newly created arbitrary outputs into GitHub; select those explicitly after checking their contents. Its run state reflects only the experiments listed in tracking.json. The general helper is manual. The explicitly authorized benchmark above has a dedicated finite monitor that can publish while ChatGPT is inactive; it observes only that batch, cannot guarantee uninterrupted updates, and stops after completion/failure or24h. GitHub always shows the latest successful published observation, not direct scheduler access.

## Reproducibility and external artifacts

Core runtime dependencies: PyTorch, numpy, Pillow, scikit-learn; torchvision for ResNet. Exact target versions are in [environment_versions.json](provenance/environment_versions.json). The published script imports without other project Python files; pretrained weights, manifests, images, selected checkpoints and optional Qwen weights remain external cluster artifacts. No new dependency installation or model download was done for this setup.

Cluster execution environment: `/home/sachin.chaudhary/.conda/envs/aiface/bin/python`. Default weights and archived configurations contain absolute cluster paths; adapt them explicitly for another environment. The mutable working directory `/home/sachin.chaudhary/ces_fl_pipeline/` remains outside this Git checkout. **A git pull does not synchronize that working directory.** New runs should use the pinned repository source rather than assume synchronization.

See [AGENTS.md](AGENTS.md) for continuation rules. Preserve originals/checkpoints, distinguish validation/test and historical/current methods, and never upload datasets, face images, credentials or binary weights to this public repository.
