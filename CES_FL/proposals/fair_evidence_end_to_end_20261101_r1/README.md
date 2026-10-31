# Standalone fair-evidence experiment — 1 November 2026

Request: implement and deliver a single Python file and GPU-operator prompt for the proposed fairness model. This new request authorizes implementation following the earlier research-only audit. It does not claim novelty or authorize this workstation to start cluster jobs. Historical code, completed results and protocols remain unchanged.

Entry point: `CES_FL/src/ces_fl_fair_evidence.py`. It has no project imports. Python 3.10+, PyTorch, NumPy, Pillow and scikit-learn are required; torchvision is needed only for the optional ResNet path. Production configurations use Xception. No model/package downloads occur. The existing, hash-pinned original-TRAIN detection pretraining checkpoint and the cluster manifests/images remain external inputs; this is downstream end-to-end training, not a replacement for that completed pretraining stage.

## Hypothesis and implementation

For class y and image condition c, aligning the joint distribution of the two actual evidence contributions across demographic groups may reduce disparities hidden by cancellation in the summed score. A second term penalizes the largest soft error disparity across three fixed conditions. Both claims require experiments; neither mathematical regularization nor a successful software test proves fair detection.

`F = encoder(x)`, `g = mean_hw(F)`, `z_g = global_MLP(g)`, `z_p = shared_patch_MLP(F)`. Image/patch sigmoid gates and positive softplus fusion weights give `c_g = a_g r_g z_g`, `M = a_p r_p z_p`, `c_l = mean_hw(M)`, and `z = c_g + c_l + b`. The local map explains only `c_l`; gates are learned weights, not calibrated reliability.

Each source image independently yields clean, JPEG quality 85, and resize-to-75%-then-restore variants after canonical 224px preprocessing. There are no chained transformations or generated demographic edits. These mild engineering choices are not independently certified to preserve forensic evidence.

For each class and each condition, a Gaussian MMD aligns each group's empirical contribution distribution with an equal-group mixture. The joint variant uses `[c_g,c_l]`; the score variant uses `z`; the marginal variant averages the two separate 1D penalties. All terms retain gradients through the mixture. The finite-sample biased MMD estimator is nonnegative up to floating-point error; small cells still produce noisy estimates. We use two distinct images per class/group per step, eight groups and both classes: 32 originals, 96 image forwards. Sampling is reproducible and identical across arms.

At the fixed training logit threshold 0, `q = sigmoid(z / 1.0)` approximates a positive decision. For each condition, compute the range of group means of q on real images and the range of group means of 1-q on fake images, then average these two ranges. The condition penalty is the maximum of that average across clean/JPEG/resize. It is a surrogate, not the measured equalized-odds gap at the final evaluation threshold.

The full objective is mean BCE across conditions + 0.1 × mean joint MMD across classes/conditions + 0.1 × worst-condition soft gap. Xception exit-flow parameters, heads, gates and fusion train; earlier encoder parameters and every BatchNorm running buffer are frozen. Total parameters: 21,594,159; trainable: 7,575,591. Encoder/head learning rates are 1e-5/1e-3 with AdamW. Ten epochs, seed/head seed 17. These are unoptimized, explicitly fixed pilot settings.

## Matched comparisons and selection

| Config | Alignment | Condition-gap penalty |
|---|---|---|
| bce.json | None | 0 |
| score.json | Summed-score MMD, weight 0.1 | 0 |
| marginal.json | Separate contribution MMDs, weight 0.1 | 0 |
| joint.json | Joint contribution MMD, weight 0.1 | 0 |
| full.json | Joint contribution MMD, weight 0.1 | 0.1 |

The initial launch prompt runs BCE and full sequentially on one allocated GPU. The other three exact configurations are prepared ablations, not automatically launched. All five share initialization, data, transformations, batch construction, update budget, optimizer and selection. Historical R1 differs in these factors and is only context, not the matched causal baseline.

This new study explicitly changes selection from the historical validation-AUC-only rule. Within each epoch, calibrate one conservative threshold using clean validation real images and a 10% empirical FPR budget. Reuse that threshold for all groups and conditions. A checkpoint is eligible only if every condition has overall validation AUC >= 0.95 and worst primary-group AUC >= 0.85. Among eligible checkpoints, minimize the maximum hard FPR/FNR group range over the three conditions; break ties by mean condition AUC. Fixed groups are TRAIN `gender_race` groups with required validation support. Save the clean-AUC comparator separately as `best_auc.pt`.

If no checkpoint qualifies, record `COMPLETED_NO_ELIGIBLE_CHECKPOINT` and exit 2. Do not relax floors after seeing results or silently use the comparator as primary. `best.pt` receives an independent reload and full validation replay recorded in `RUN_VERIFIED.json`. Test evaluation is an explicit separate command; never use it for selection or tuning. No direct apples-to-apples comparison to Ding is established by this new pilot.

## Outputs and checks

Training writes config/provenance, epoch histories with condition and group metrics, `last.pt`, selected `best.pt` when eligible, the separately labeled AUC comparator, summary and verification receipt. Exact epoch-boundary resume is supported using the same source/config and `--resume`; inconsistent partial artifacts are preserved and rejected. Evaluation uses the stored threshold even if test labels change. Cancellation audits report descriptive contribution offsets and hypothetical fusion perturbations; they are not causal proof. Signed map exports contain actual local/global/bias decomposition. No bootstrap intervals or multi-seed significance claims are implemented in this pilot.

Run `smoke --device cpu|cuda --out NEW_DIR` for numerical/gradient, sampling, model, training, replay, resume, evaluation, audit and export checks on temporary synthetic fixtures. Run `dry-run --config CONFIG --device cuda --out NEW_DIR` on the allocated GPU for one actual configured batch/backward without optimizer steps, checking that all parameters and buffers remain unchanged. Memory figures are for this dry run, not a reservation or training-throughput guarantee.

See `evidence/` for actual local test receipts. CUDA execution, the cluster's PyTorch 2.7 environment, real-data gradients and model performance remain pending until the operator records them. Current local software tests use PyTorch 2.15. The optional ResNet path has no dedicated end-to-end receipt and is outside the prepared Xception launch.

Use `GPU_OPERATOR_PROMPT.txt` after pinning the exact published commit. The proposal's `SHA256SUMS` pins the runnable source and all five configurations. Do not use the old `train-config` launcher with this standalone CLI.
