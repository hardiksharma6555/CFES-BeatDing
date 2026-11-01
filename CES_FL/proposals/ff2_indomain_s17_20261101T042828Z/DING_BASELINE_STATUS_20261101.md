# Ding baseline on the exact public FairForge-2 split

Observed UTC: 2026-11-01T05:40:33.391498+00:00.

**Answer: no completed Ding training/evaluation result was found on this exact public split.** Existing completed Ding FairForge-2 results belong to the larger internal/full split. No new training, inference, image generation, configuration edit or architecture change was performed for this check.

| Evidence | TRAIN | Validation/select | TEST | Ding status |
|---|---:|---:|---:|---|
| Frozen public FairForge-2, current CES-FL plan | 218,266 | 31,107 | 87,564 | Completed matching Ding retrain not found |
| Historical internal/full FairForge-2 | 502,035 | 74,974 | 182,884 | Completed local classification and SFD/GDA adaptation |

The actual historical run log names `data_ding_schema/fairforge2_full` on its first line. Both saved score archives contain 182,884 rows. CPU recomputation of the existing raw logits gives classification AUC **99.261922%** and final local SFD/GDA AUC **99.357254%**. These are local historical measurements, not published-paper FF2 numbers and not results on the 87,564-image public test split. This check did not perform new model inference or use those scores to tune anything.

## Direct evidence checked

- Historical launcher and defaults: `/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/code/run_task_ff2.sh`.
- Actual dataset/run evidence: `/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/DING_ff2_s7/run.log:1`.
- Saved classification scores: `/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/DING_ff2_s7/scores/base_ff2_test.npz`.
- Saved final scores: `/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/DING_ff2_s7/scores/ff2_test.npz`.
- Dataset schemas/counts: `/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/data_ding_schema/fairforge2_full/` and `fairforge2_public/`.
- Current public adapter/path verification: [INPUT_VERIFICATION.json](INPUT_VERIFICATION.json); public raw CSVs and CES-FL adapters differ in schema, with exact image/label membership checked separately.
- Result-directory inventory: DINGREPRO_LIN/results has the completed DING_ff2_s7 and stopped DING_ff2_s17_STOPPED_oct23; both inspected run logs name the full split. CFES/runs_ff2id is empty. CESEL_MA/runs_ff2public contains older CFES-D, phase1 and a partial B0, not a completed public Ding retrain.
- [Machine-readable status, current source hash, manifests and CPU score recomputation](ding_baseline_evidence_20261101.json).

This is a search of the checked local project archives, not proof that no unindexed external result exists. Scheduler names alone do not identify a matching completed baseline; the current jobs do not supply such an artifact here.

## Recommended comparison

Establish a separate local Ding baseline trained on the same public TRAIN membership, with public validation and TEST held out, alongside CES-FL on that split. Preserve the completed internal Ding run; this is a different dataset experiment, not a request to redo the old FF++ reproduction.

Keep the CES-FL architecture fixed initially. A missing baseline is not evidence that its architecture needs changing. The first comparison should include Ding initial classification, Ding SFD/GDA as an explicitly identified source variant, our original-image BCE/DRO controls, and our counterfactual arm when its matched controls are ready. Both methods use the Xception encoder family; their heads, channel treatment and training schedules are not automatically identical. Record those differences.

Use the same public image membership, label definition, demographic grouping/provenance, held-out evaluation implementation and threshold policy for a controlled comparison. Keep the inherited AUC-only validation selection rule and empirical validation-real target FPR 0.1 fixed. If the Ding reference retains its released final-epoch selection, report that as a separate reference protocol rather than claiming selection is identical. Specify exposure/update budgets before launch and report any native-method differences. Ding's hardcoded GDA seed 121 must be disclosed; do not label every stage seed 17 without an explicit source variant.

The paper claim should concern demographic errors and worst-group performance at useful detection quality, supported by uncertainty. A larger overall AUC alone does not establish better fairness. The existing FF2 age/skin labels remain automated proxies. The sixteen approved age pairs cover only eight originals; matched controls are pending. There is no guaranteed improvement over Ding and no justified direct comparison with its published FF++ numbers from this new FF2 dataset.

## Important implementation blocker for the Ding fairness baseline

Current archived GDA.py lines 122–124 freeze all model parameters except `backbone.last_linear`. Lines 232–235 compute both `fair_loss` and `total_loss` inside `torch.no_grad()`. This source still matches the prior executed stage probe hash:

`bef29fa69e46c729dc679356d0ac3904ebac6da4c00180350daf282fe94599db`.

Prior exact-source CPU/gradient evidence is `/home/sachin.chaudhary/ces_fl_pipeline/audits/encoder_controls_20261030T202634/ding_stage_probe.json` (PyTorch 2.10.0+cu128, GeomLoss 0.3.1). It recorded a detached fairness term and identical head updates for fairness weights 0, 0.005, 1000. GeomLoss's grad-mode side effect let the classification term backpropagate in that environment. Other dependency behavior can instead cause backward failure. BatchNorm buffers can still update even though most affine parameters are frozen. The source/hash was checked now; that gradient probe was not rerun now.

Therefore, a native archived-code baseline must carry this limitation. A gradient-corrected variant must be a separate, explicitly documented baseline with new software tests; it must not silently replace the reference or be described as a verified reproduction of the paper's unpublished training. No GDA source was patched for this status check.

## Next dependency

Prepare the exact public-split Ding baseline specification, including native versus gradient-corrected GDA labeling, initialization, selection, seed policy and compute budget, before CUDA smoke or training. Existing architecture and frozen science settings remain unchanged. This report proposes that comparison; it does not launch a job.
