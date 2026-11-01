# CES-FL — supervisor review and training handoff

**Start here.** [Current timestamped status](STATUS.md) · [Review/change/run workflow](WORKFLOW.md) · [Prompt for a supervisor or another LLM](SUPERVISOR_PROMPT.md)

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
