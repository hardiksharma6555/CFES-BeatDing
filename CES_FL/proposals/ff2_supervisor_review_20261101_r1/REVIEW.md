# FairForge-2 supervisor review — DO NOT RUN full comparison

Exact upstream commit reviewed: `f73c98d28f01604e45525b2acff49cd945b1cd33`.
This is an additive review record, not an amendment to the existing preregistration. No training or generation was launched. Raw cluster data, weights, GPU allocation and the historical raw gradient probe remain unverified through GitHub.

References below refer to that immutable commit. The subsequently published FF++ 50+50 script is a separate experiment and does not close these FF2 blockers.

## Prioritized findings

1. **P1: comparison specification incomplete.** Public split counts are 218266/31107/87564; published manifest receipts report no recorded contributor/path overlaps. `ff2_indomain_s17_20261101T042828Z/P_ff2_bce_s17.json:8` still specifies 10 epochs; exact public-split Ding and downstream configurations are absent/unbound (`request.json:218`). A 50+50 change needs a new versioned proposal, not an edit to the locked config. Match or disclose optimizer steps and image exposures; Ding concatenates separate real/fake batches (`DINGREPRO_LIN/code/Ding_mod/train_ori_model.py:134`), so nominal batch size is not total images/update. Historical internal-FF2 numbers are not public-split comparison scores.

2. **P1: define the Ding variant precisely.** Archived `DINGREPRO_LIN/code/Ding_mod/GDA.py:232-239` computes group predictions and fairness/total loss in no_grad. The published prior probe reports a detached fairness term and identical head updates across lambda settings; its raw run is unavailable here. Behavior depends on the archived dependency's grad-mode side effect; another environment may fail backward. Removing no_grad alone still aligns mixed real/fake two-class logits, unlike the class-conditioned probability distributions in paper Eq.6. Primary comparison should use an explicitly named, differentiable, class-conditioned public-split Ding adaptation; archive-faithful GDA should remain a separate reference. Minimal gradient-only repair is a distinct diagnostic, not a paper-faithful reproduction. Declare head-only optimization, updating BN buffers (`GDA.py:198`), and hardcoded GDA seed121 (`aux_funcs.py:68`). Tests must show nonzero fairness-only gradients on misaligned fixtures, lambda-dependent parameter updates with identical forwards/initialization, and independent class conditioning. SFD must rank TRAIN-only channels and pass that recorded mask; do not silently use GDA's fallback channel list.

3. **P1: no-change/age comparison is not ready.** Proposal README:17-21 reports only 16 age edits of eight sources (four real, four fake; fake sources all Editing), visual overrides and unverified matched controls. RN/RD remain blocked pending source-matched no-change and geometric controls. `CES_FL/src/ces_fl_upgrade.py:89-141,231-252` samples separate anchors with replacement: repeated epochs do not expand unique coverage or create edits for every ordinary training image. Even after controls pass, this is an eight-source age pilot, not broad age/gender/skin evidence. Review cannot prove identity or preservation of forgery artifacts from approval flags.

4. **P2: interpret controls and weighting correctly.** R0→R1 isolates partial fine-tuning after shared detection pretraining, not pretraining benefit. R1→R2 tests DRO, R2→RN generic editor consistency, RN→RD age versus no-change. In `ces_fl_upgrade.py:60-86`, absent groups contribute zero while q sums over all groups: observed q mass changes classification-loss scale. This may be intentional stochastic estimation; log it and preregister any scale-control diagnostic before running. An ImageNet control is needed only if claiming benefit from detection pretraining itself.

5. **P1: common score adapter missing.** `upgrade_tests_s17_20261101T040407Z/runtime/evaluate_frozen.py:85-101` loads CES scalar-logit models. Ding returns two logits (`DINGREPRO_LIN/code/Ding_mod/detectors/xception_detector.py:97`). The corresponding binary scalar is fake_logit minus real_logit, not fake_logit alone. Test class orientation, manifest order, preprocessing and equality to softmax fake probability; use identical metrics/eligibility/calibration for both methods.

6. **P2: selection is defensible; uncertainty is incomplete.** Existing validation-AUC selection and validation-real empirical FPR calibration (`ces_fl_diagram_model_v021.py:583-591,755-810`) keep test out of selection. Preserve both fixed0.5 and validation-calibrated reporting. `CES_FL/experiments/benchmark_seed17/uncertainty.py:49-96` produces aggregate axis CIs but not the promised individual-group CIs or paired method-difference CIs. Implement shared component bootstrap draws for paired differences; hold thresholds/checkpoints fixed and report eligible support, missing/invalid replicates and component counts. These intervals are conditional on one trained seed, not training-seed variability. CI overlap is not a paired superiority test.

7. **P2: groups and labels differ.** Training balancing/DRO uses label×ids8; main table reporting uses gender_race (`FINAL_TABLE_TEMPLATE_20261101.md:45`). Declare the mapping and Ding's grouping; do not assume the partitions are interchangeable. Age/skin labels are proxies (automatic age, ITA skin bins). Report missingness, exclusions, per-class support and manipulation families. Recorded contributor disjointness is not verified true-person/donor independence.

## Recommended preregistered experiments

- Public-split Ding classification; archived SFD/GDA reference; explicitly corrected class-conditioned Ding adaptation. Keep native-final checkpoint reporting distinct from common validation-selected comparisons.
- CES R0/R1/R2 with the same bound FF2 pretraining checkpoint and matched head seeds; RN/RD only after verified paired controls, labeled an eight-source diagnostic.
- Optional minimal-gradient GDA repair as a separate diagnostic. Full-coverage counterfactual 50+50 is a separate versioned study; the accompanying FF++ configuration must not be relabeled FF2.
- Freeze exact stage budgets, optimizer, image exposure, BN policy, seeds, mask provenance, threshold rules, label mapping and primary fairness endpoints before training/test evaluation. Multi-seed confirmation should be prespecified; never choose seeds or hyperparameters against test scores.

## Remaining blockers and release criteria

A new exact FF2 protocol plus all configurations; tested Ding objective variants and common score adapter; verified real paired controls for RN/RD; per-group and paired uncertainty implementation; actual allocated CUDA checks and data verification. Publish source/config/input hashes and evidence before full comparison. Completed tables correctly retain pending cells; no superiority over Ding has been established.

Primary paper: https://arxiv.org/html/2511.10150v5 (Section3.3, Eq.6).
All abbreviated proposal paths above are under `CES_FL/proposals/`. Completed evidence inspected: `CES_FL/results/upgrade_tests_s17_20261101T040407Z/{REPORT.md,aggregate.json,final_verification.json}`. Those FF2 results are FF++→FF2 transfer, not public-FF2 in-domain training.
