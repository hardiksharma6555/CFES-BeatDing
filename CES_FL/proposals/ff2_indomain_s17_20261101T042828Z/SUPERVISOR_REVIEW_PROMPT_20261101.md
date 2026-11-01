# Supervisor review prompt — FairForge-2, 1 November 2026

Copy and send the following prompt. This dated prompt supersedes the older general handoff for this particular review.

Please review our CES-FL FairForge-2 experiment before we launch training.

Repository: https://github.com/hardikbha/CFES-BeatDing
Branch: main. Review reference: commit bc322b7; record the exact commit you review.

Start with CES_FL/README.md, CES_FL/AGENTS.md and CES_FL/WORKFLOW.md. Then inspect:

1. CES_FL/proposals/ff2_indomain_s17_20261101T042828Z/ — README.md, request.json, P_ff2_bce_s17.json, INPUT_VERIFICATION.json, DING_BASELINE_STATUS_20261101.md and FINAL_TABLE_TEMPLATE_20261101.md.
2. CES_FL/src/ces_fl_upgrade.py and ces_fl_diagram_model_v021.py — actual architecture, training policies, loss, checkpoint selection, calibration and evaluation.
3. CES_FL/results/upgrade_tests_s17_20261101T040407Z/REPORT.md and aggregate.json — completed five-arm FF++-trained study, including all 25 test evaluations and mixed fairness/weak transfer findings. This is separate from the proposed FF2 in-domain training.

The fixed public FF2 split contains 218,266 TRAIN, 31,107 validation and 87,564 TEST images. Seed 17 is the initial pilot. No matching public-only Ding retrain or current CES-FL in-domain result exists yet. The older 99.36% Ding AUC belongs to the larger internal split. All unmeasured cells must remain Pending.

Please assess:
- Whether the planned Ding-versus-CES-FL comparison is methodologically fair: data membership, Xception initialization, preprocessing, optimization/exposure budgets, validation selection, thresholds, demographic groups and uncertainty. Distinguish native Ding selection from any selection-matched variant.
- The archived Ding GDA fairness-gradient issue. Decide how to label the archived reference and whether a separately specified corrected-gradient baseline is needed. Do not silently fix or substitute it; disclose its hardcoded GDA seed 121.
- Whether CES-FL controls isolate encoder pretraining, BCE/DRO, editor no-change consistency and age consistency. FF2 has 16 visually approved age edits from eight TRAIN originals; matched controls remain unverified. This supports a small-pool diagnostic, not broad demographic coverage or validated forensic preservation.
- Whether the proposed tables adequately report AUC, worst-group AUC, group FPR/FNR, Ding-style F_FPR/F_DP/es-AUC and separate age/skin results. Age/skin labels are automated proxies; ids8 and gender_race are different fields.

Return PASS, PASS WITH FIXES, or DO NOT RUN; prioritized findings with file/line references; the exact recommended baseline variant and arm list; and remaining prerequisites. Separate measured evidence from assumptions. If GitHub lacks raw datasets, weights or live scheduler access, mark those checks unverified.

Review only: do not launch jobs, generate edits, change the preregistration, tune on test, overwrite artifacts or alter the current architecture/configurations. Provide proposed changes for review first. The historical CESel-MA gate remains FAIL. Improving overall AUC alone does not establish improved fairness.
