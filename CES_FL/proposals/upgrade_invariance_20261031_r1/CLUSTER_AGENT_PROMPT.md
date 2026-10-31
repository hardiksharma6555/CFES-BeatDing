Use the exact commit in the supervisor handoff for branch
`upgrade/ces-fl-controlled-invariance-20261031`. Read CES_FL/AGENTS.md, README.md,
WORKFLOW.md and this proposal's README.md/request.json/software_checks.json.

1. Fetch the provided remote branch (or supplied Git bundle). Check its resolved
   commit against the handoff. Make a NEW detached worktree under
   /home/sachin.chaudhary/ces_fl_sources/. Preserve running-job sources and all
   historical results. Do not merge/pull over the cluster's mutable source tree.
2. Inspect existing jobs and output directories to avoid duplicate training.
   Verify the recorded external file/checkpoint hashes with the upgrade preflight.
   Do not substitute datasets, checkpoints, approvals or hash values on failure.
3. On a legitimately allocated GPU (ngpus=1 and verified CUDA binding), use the
   proposal's new launcher CES_FL/cluster/upgrade_one_gpu.pbs. First submit a
   smoke-only job with CES_SMOKE_ONLY=1 and R1_partial_bce.json. It runs preflight
   and the full upgrade CUDA software suite. CPU smoke passed locally; CUDA is
   still unverified. Inspect the actual exit status and CUDA_SMOKE_PASS artifact.
4. If GPU smoke and all real-input checks pass, you are authorized to submit the
   five configured Xception arms R0_frozen_bce, R1_partial_bce, R2_partial_dro,
   RN_partial_dro_nochange_pilot and RD_partial_dro_age_pilot using
   CES_SMOKE_ONLY=0, the same launcher, each matching configs/<arm>.json, and a
   UNIQUE external output directory per job. Schedule according to actual GPU
   allocations; do not kill unrelated jobs or use guessed device indices. The
   launcher also repeats CUDA smoke before each training job.
5. RN/RD deliberately use only six approved sources and are small-pool exposure
   diagnostics. They do not implement the future >=300-source coverage study.
   Do not invent new approvals, generate new images, lower a coverage guard, or
   represent the six-source results as evidence for the expanded study.
6. Keep the five exact configurations and original sampling/update budget fixed.
   Select by validation AUC only and retain the selected checkpoint's calibrated
   validation threshold. No test-based tuning or automatic test evaluation.
   No full-encoder unfreezing, forensic loss or map-consistency loss is authorized.
7. After launch and completion/failure, publish aggregate evidence: commit and
   config/source/input/checkpoint hashes, scheduler IDs, GPU allocation, paths,
   CPU/CUDA/preflight logs, history, validation subgroup metrics, pair exposure,
   q weights, image-forward counts and elapsed time. Check DONE/checkpoint reload
   and selection/calibration independently before marking a run verified. A DONE
   marker alone is not verification. No private images/row-level records/weights.
8. Report actual status and failures promptly. Do not claim new test scores or a
   win over Ding from synthetic tests or validation results. Frozen-protocol test
   evaluation and bootstrap uncertainty integration are subsequent explicit work.

Refer to README.md for the exact qsub form and environment variable names. Every
training job must use CES_FL/src/ces_fl_upgrade.py through the NEW launcher, not
the historical v0.2.1 launcher. The generated PNG/PDF is illustrative; executable
source/configurations are authoritative.
