# FairForge-2 public: separate in-domain study, seed 17

Observed UTC: 2026-11-01T04:42:46.223256+00:00. **Prepared; no new FF2 training has started.**

Use the existing frozen public TRAIN/VAL/TEST release. Do not randomly reshuffle it. This is separate from FF++-trained transfer evaluation on the same public TEST manifest.

| Split | Images | Real | Fake | Absolute manifest |
|---|---:|---:|---:|---|
| train | 218,266 | 16,777 | 201,489 | `/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/train.csv` |
| val | 31,107 | 2,438 | 28,669 | `/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/val.csv` |
| test | 87,564 | 5,441 | 82,123 | `/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/test.csv` |

## Verified inputs

Exact image-path/label membership matches the frozen public release. There is zero cross-split overlap in canonical image paths and recorded identity_ids/source_ids, parsing all semicolon-separated contributors. The actual model manifest reader checked every one of the 336,937 original image paths is a file, binary labels, class coverage, duplicate paths and required provenance. All model source/configuration/template/weights/manifest hash checks passed. This is metadata/path validation, not decoding every image. A previous stratified decode checked 16 originals per split (48 total) and all 16 approved age-edit files. Hidden donors, duplicate content under different paths and true-person separation are not established automatically.

Existing 16 approved age pairs cover eight TRAIN originals (four real/four fake), each edited independently younger and older. Approval is the user visual approval with automatic thresholds bypassed on 30 October. No held-out original occurs in this pair pool. This small pool is not broad demographic or forgery-family coverage; the four fake sources are all Editing-family examples. Approval does not establish preservation of forensic evidence.

No-change and legacy resize files exist for these eight originals, but a complete control manifest matched to the actual Qwen conditioning geometry has not passed verification. The six-source Ding control manifest is a different pool and must not be reused. RN/RD FF2 arms remain blocked on this control audit. No new images were generated or approvals fabricated.

Age labels are automatic age estimates and skin labels use automatic ITA; ids8 is a FairFace-derived field. Retain this provenance and missingness. They are not human demographic ground truth. The current annotation review and hidden-contributor limitations remain relevant.

## Prepared protocol

1. Original-image BCE encoder pretraining P: seed 17 Xception, 10 epochs, batch size 16, 224 pixels, 4 workers, balanced label×ids8 sampling, head LR 0.001, encoder LR 0.0001, weight decay 0.0001, full encoder trainable, BN updates. This is exactly the previous original-TRAIN P template with only train/val paths and the obsolete FF++ overlap exception changed. No test field, pair pool or counterfactual loss is passed to P.
2. Select P by FF2 validation AUC only and store a threshold calibrated only on selected-checkpoint FF2 validation reals using the existing empirical-budget target 0.1. Never use test scores/labels to update weights, select epochs or calibrate. The same public TEST has already been used for the separate transfer report; do not describe it as never previously scored.
3. Only after P exists, bind downstream controlled arms to its actual selected checkpoint/hash, using identical head initialization and the approved controlled-arm protocol. The source explicitly rejects an encoder pretrained on different TRAIN/VAL manifests. Do not silently substitute the FF++ encoder or fabricate an FF2 hash.
4. Before any FF2 launch, verify the new configuration/input preflight, target CUDA/backward smoke and GPU binding. Paired arms additionally require complete control provenance and the actual eight-source approvals. No full forensic-selectivity or validated gender/skin invariance claim is supported by these age-only pairs.

The P configuration was frozen before current FF2 transfer scores were inspected. The settings are inherited controls, not FF2 test-tuned values. Class imbalance requires reporting AUC, group support and errors rather than accuracy alone.

## Files and commands

Original plan directory: `/home/sachin.chaudhary/ces_fl_pipeline/training_plans/ff2_indomain_s17_20261101T042828Z`.

[Exact P configuration](P_ff2_bce_s17.json) · [Request, provenance and hashes](request.json) · [Actual model-reader verification](INPUT_VERIFICATION.json) · [Verification log](input_verification.log).

Executed input-only command:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_pipeline/training_plans/ff2_indomain_s17_20261101T042828Z/verify_inputs.py
```

Actual output: SPLIT_PASS train 218266, val 31107, test 87564; FF2_INPUT_VERIFICATION_PASS. This command did not train or calculate test performance.

Prepared P command, **not executed**; run only after launch prerequisites and a legitimate verified GPU binding:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672/CES_FL/src/ces_fl_diagram_model_v021.py train-config \
  --config /home/sachin.chaudhary/ces_fl_pipeline/training_plans/ff2_indomain_s17_20261101T042828Z/P_ff2_bce_s17.json \
  --out /home/sachin.chaudhary/ces_fl_pipeline/training_plans/ff2_indomain_s17_20261101T042828Z/proposed_P_training --device cuda
```

No proposed_P_training output directory or FF2 job is created by preparing this command. Preserve all existing historical/internal FF2 runs and the current transfer outputs. The exact P checkpoint and final downstream run directories cannot be reported until they exist.

## Ding comparison status checked on 1 November

No completed Ding retrain was found on this exact public split. The historical 99.3573% AUC belongs to the larger internal/full split. [Verified evidence, archived GDA limitation and recommended matched baseline](DING_BASELINE_STATUS_20261101.md). CES-FL architecture/configurations remain unchanged; no new training was launched.

[Proposed final comparison tables and reporting rules](FINAL_TABLE_TEMPLATE_20261101.md): unmeasured cells remain Pending; preparing a reporting template does not launch training or change configurations.

[Copy-and-send supervisor review prompt](SUPERVISOR_REVIEW_PROMPT_20261101.md), covering the public-split Ding baseline, code, controls and proposed tables.
