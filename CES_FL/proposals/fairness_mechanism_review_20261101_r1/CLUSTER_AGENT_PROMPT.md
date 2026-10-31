# Read-only fairness mechanism audit — do not train

The user has paused the proposed class-conditional model upgrade until its fairness contribution is stronger. Fetch the exact reviewed commit supplied with this handoff into a new detached worktree. Do not pull over a running job's source.

Read CES_FL/AGENTS.md, WORKFLOW.md and this proposal's README.md/request.json. This package is an audit, not a training request. Do not start a new training arm, generate images, run test evaluation, change existing models/configs, or stop any existing job.

Use `/home/sachin.chaudhary/.conda/envs/aiface/bin/python`. On an appropriate CPU compute allocation first run:

```bash
CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
CES_REVIEW=CES_FL/proposals/fairness_mechanism_review_20261101_r1
"$CES_PY" -m unittest discover -s "$CES_REVIEW" -p test_mechanism.py -v
```

For GPU execution obtain a real `ngpus=1` allocation, verify job-specific CUDA binding, then run the same tests with `CES_AUDIT_DEVICE=cuda`. Low GPU usage or a host name alone does not establish allocation. This proposal does not extend previous direct-SSH sharing permissions. Do not install packages or download weights automatically.

Inspect the three `*_audit.json` files. They pin existing completed R0/R1/R2 checkpoint hashes and the original validation manifest. Verify files exist and match hashes; stop and report discrepancies. The audit verifies them again. Choose a new absolute cluster output root outside the immutable checkout; for example append the reviewed commit and an execution timestamp to `/home/sachin.chaudhary/ces_fl_pipeline/audits/fairness_mechanism_review_20261101_r1_`. Record the resolved path before running; preserve any prior outputs.

From the immutable checkout, after setting `CES_AUDIT_OUT` to that new root and `CES_AUDIT_DEVICE` to the verified compute device (`cpu` or `cuda`), run:

```bash
set -euo pipefail
: "${CES_AUDIT_OUT:?New absolute audit output root required}"
: "${CES_AUDIT_DEVICE:?Verified compute device required}"
for CES_ARM in R0_frozen_bce R1_partial_bce R2_partial_dro; do
  "$CES_PY" "$CES_REVIEW/audit_checkpoint.py" \
    --config "$CES_REVIEW/${CES_ARM}_audit.json" \
    --out "$CES_AUDIT_OUT/$CES_ARM" --device "$CES_AUDIT_DEVICE"
done
```

Publish only aggregate report JSONs and textual execution evidence under a new dated result folder on a separate publication branch. Include exact commit, input/source hashes, command, device allocation, test logs, observed time, output path and failures. No raw images, per-person predictions, checkpoints or credentials belong in GitHub.

Summarize: Did nominal validation AUC replay match? Are contribution shifts opposite within class? How did FPR/FNR gaps and AUC change under the same ±10% gain perturbations and unchanged threshold? Which groups have weak support? Explain that stress results are algebraic diagnostics, not measured corruption robustness, causal bias, or improvements from a new model.

Return the findings and stop. Real corruption diagnostics, cluster uncertainty estimates and a defensible novelty/fairness decision are still required before a separate model-training proposal. A passing software test or nonzero cancellation diagnostic does not satisfy those research gates automatically.
