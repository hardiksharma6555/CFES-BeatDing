# Proposed final FairForge-2 public comparison tables

Reporting draft prepared 2026-11-01T05:45:12.583953+00:00. This is a table template, not experimental results or a change to the locked model/configuration/preregistration. No training was launched.

## Dataset and interpretation

Train on 218,266 public TRAIN images; select/calibrate using 31,107 public validation images; report on 87,564 public TEST images. Exact paths, counts, source-provenance limitations and hashes are in [the prepared plan](README.md) and [input verification](INPUT_VERIFICATION.json). Preserve the existing public split.

Ding rows mean a **local public-split retrain of the archived Xception method**. They are not Ding's published paper numbers. The earlier 99.3573% result belongs to the larger internal split and must not be inserted in this table. The current FF++→FF2 transfer results likewise answer a separate question.

## Main comparison: detection and demographic fairness

All numeric cells remain Pending until the corresponding public-TRAIN experiment and frozen-checkpoint TEST evaluation actually complete. Arrows indicate preferred direction. AUC and fairness values will be reported as percentages; F_FPR is a sum of absolute FPR deviations, not the max-minus-min FPR gap.

| Method | Test AUC ↑ | Worst gender×race AUC ↑ | Gender F_FPR ↓ | Race F_FPR ↓ | Gender×race F_FPR ↓ |
|---|---:|---:|---:|---:|---:|
| Ding classification stage, Xception | Pending | Pending | Pending | Pending | Pending |
| Ding SFD+GDA, archived implementation* | Pending | Pending | Pending | Pending | Pending |
| CES-FL R0: frozen encoder + BCE | Pending | Pending | Pending | Pending | Pending |
| CES-FL R1: partial encoder + BCE | Pending | Pending | Pending | Pending | Pending |
| CES-FL R2: partial encoder + group DRO | Pending | Pending | Pending | Pending | Pending |
| CES-FL RN: R2 + no-change consistency | Pending | Pending | Pending | Pending | Pending |
| CES-FL RD: R2 + reviewed age consistency | Pending | Pending | Pending | Pending | Pending |

*The archived GDA fairness term is detached in the previously executed source/environment probe. This remains an implementation limitation, not a repaired fairness baseline. A corrected-gradient implementation, if separately specified and tested, needs its own clearly labeled row. Do not silently replace the archived reference. [Verified baseline status and exact source evidence](DING_BASELINE_STATUS_20261101.md).

CES-FL downstream arms must use their own selected public-FF2 P encoder, with the actual checkpoint/hash bound after P exists. R1/R2/RN/RD retain the controlled partial-encoder policy; R0 freezes that encoder. Current approval does not make these drafted public-FF2 downstream bindings exist. RN/RD additionally require matched control provenance for the eight reviewed FF2 originals; they remain small-pool diagnostics. No full forensic-sensitivity objective is represented.

## Additional demographic panels

For each method, provide the following four blocks, including individual groups, real/fake support and 95% uncertainty intervals. Age/skin labels are automated proxies and must be marked as such; they are not human ground truth.

| Demographic panel | Reported quantities |
|---|---|
| Gender | Per-group AUC, FPR, FNR; F_FPR; F_DP; es-AUC |
| Race | Per-group AUC, FPR, FNR; F_FPR; F_DP; es-AUC |
| Gender×race | Per-group AUC, FPR, FNR; worst-group AUC; F_FPR; F_DP; es-AUC |
| Age and skin, separately | Per-group AUC, FPR, FNR; worst-group AUC; max-minus-min FPR/FNR gaps; missingness and label provenance |

This is an FF2 in-domain table family. Ding's original Tables1–3 use their own datasets/backbones/protocols and cannot be filled with FF2 numbers under the original dataset headings.

## Fixed evaluation and remaining execution decisions

- Use the same original TRAIN/VAL/TEST membership and demographic definitions across compared methods. Synthetic variants stay in TRAIN and retain source lineage.
- Report overall AUC on continuous fake-class scores. Use gender_race for the displayed intersection; do not silently substitute the FairFace-derived ids8 field used by the sampler.
- Apply the existing shared evaluator and min_support 20 per class for subgroup eligibility; show excluded/missing groups and support. Do not erase small-group uncertainty by treating frame rows as independent identities.
- Report both the fixed 0.5 operating point and the existing validation-only empirical-real FPR-budget rule, requested target 0.1. Each selected checkpoint has its own validation threshold; freeze it before TEST. Report achieved TEST FPR/FNR; 10% is not guaranteed on TEST.
- Retain the current CES-FL AUC-only validation-selection policy. The archived Ding chain uses final-epoch selection. Before launching a Ding public-split retrain, specify whether it is a native-selection reference or a separately labeled selection-matched comparison. The table alone does not resolve this difference.
- Freeze training exposure, resolution, optimization schedule, initialization and seed handling before launch. Same Xception family and same data do not imply identical heads or compute. Ding's released GDA uses hardcoded seed 121; disclose it rather than silently claiming seed 17 throughout.
- Initial runs remain single-seed pilots. Show recorded-contributor component-bootstrap CIs where supported; these do not measure between-seed variability. Additional training seeds come later if authorized.
- Compare RN versus R2 to assess the editor-control effect; compare RD versus both R2 and RN to assess whether an age effect exceeds the control. Review detection trade-offs, worst-group metrics and errors together. One favorable metric does not establish a consistent fairness gain.

The immediate goal is a valid matched dataset comparison. No architecture modification, test tuning, extra synthetic generation or promise of beating Ding follows from preparing this table.
