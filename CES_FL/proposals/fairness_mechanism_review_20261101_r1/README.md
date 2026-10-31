# Fairness mechanism review — 1 November 2026

**State: HOLD_MODEL_TRAINING.** The user asked to improve the fairness contribution before proceeding with the low-novelty score-alignment model. This package implements a synthetic mechanism probe and a read-only checkpoint audit, not a new detector trainer. No training configuration or automatic launch is included. Historical source, results, protocols and the CESel-MA FAIL gate remain unchanged.

The earlier generated diagram describes the score-alignment baseline. It does not yet depict the revised research hypothesis below.

## Decision and novelty assessment

The basic proposal, BCE plus class-conditional score MMD, is **3/10 for novelty** in my subjective assessment. It is an appropriate baseline, but should not be advertised as a new fairness principle. The more specific direction below is **5/10 as a research hypothesis**, not an established publication contribution. No search can certify novelty; a renamed loss or extra module does not establish it.

Primary-source review:

| Prior work | Relevant overlap | Consequence for our claim |
|---|---|---|
| [Pfohl et al., An Empirical Characterization of Fair Machine Learning for Clinical Risk Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC7871979/), §2.1.5 | Gaussian MMD penalties for conditional prediction parity/equalized odds | Class-conditional score MMD is established. |
| [Ding et al., Decoupling Bias, Aligning Distributions](https://arxiv.org/html/2511.10150v5), GDA | Class-separated subgroup-to-global prediction distribution alignment, alongside structural decoupling | Substituting MMD for transport is insufficient novelty. Compare against the intended differentiable method as well as any archived-code reproduction. |
| [FairForensics](https://arxiv.org/html/2608.01661), §III-C, equations 9–13 | Within-class subgroup prototype alignment plus real/fake separation | Preserving class separation while aligning groups is also established. |
| [FairReL](https://arxiv.org/html/2608.28777v1), §3 | Targets spatial wavelet structure and fine-tuning residual features with demographic supervision; aligns subgroup residual means within class | Component-specific fairness supervision alone is not a distinctive contribution. |
| [FairFD, Thinking Racial Bias in Fair Forgery Detection](https://arxiv.org/html/2407.14367), bias-offset discussion | Aggregation across forgery methods can hide opposing biases | Do not claim discovery of bias cancellation in general; report results by manipulation family. |
| [Practical Compositional Fairness](https://research.google/pubs/practical-compositional-fairness-understanding-fairness-in-multi-component-recommender-systems/) | Studies how component fairness relates to system fairness | Fairness of components and their composition is an existing research area. |

These are substantive overlaps. The proposed distinction is narrower: **audit demographic differences in the joint distribution of the detector's exact additive contributions, then test whether preventing cancellation makes error parity more stable under changes in evidence balance.** It remains possible that this direction is too incremental or irrelevant to our actual failures.

## Candidate mechanism

The existing model exports exact contributions:

```
c_global = a_g * r_g * z_g
c_local  = mean(a_p * r_p * z_p)
z        = c_global + c_local + bias
e        = [c_global, c_local]
```

The two branches may carry opposing group-dependent shifts. Identical final scores can therefore coexist with different joint contribution distributions. This is a distributional observation, not proof that either branch contains causally harmful demographic information. Different forgery evidence, image quality or manipulation mixtures could legitimately change contributions.

Candidate training objective, **not yet integrated into a trainer**:

```
L = BCE(z, y) + lambda_joint * L_joint
L_joint = (1/2) sum_y (1/G_y) sum_g MMD²(P(e | y,g), Q_y)
Q_y = (1/G_y) sum_g P(e | y,g)
```

Use the joint two-dimensional vector, not two independently aligned marginal distributions. Separate marginal alignment can miss group-dependent correlations. The probe includes a counterexample. Group weights are equal within each class; neither class is aligned with the other. All score/reference paths remain differentiable. Gates remain learned weights, and spatial maps remain signed local contributions, not forensic ground truth.

Why this could help: exact equality of conditional joint contribution distributions implies equality of the score distributions under every common deterministic fusion function, including common branch-gain changes. Hence a shared threshold has equal class-conditional error rates under that idealized equality. This is an elementary pushforward property, **not a new theorem**. Small finite-sample Gaussian MMD does not itself guarantee a small hard-threshold error gap, and source equality does not guarantee target-domain equality.

The diagnostic perturbation is:

```
z_rho = (1 + rho) * c_global + (1 - rho) * c_local + bias
rho in {-0.1, 0, +0.1}
```

All groups use the same rho and the checkpoint's unchanged validation threshold. These are algebraic sensitivity tests. They are not substitutes for experiments showing that actual compression, resizing or unseen manipulation methods create relevant evidence shifts. No group-dependent inference branch or threshold is introduced.

## What has actually been tested

`mechanism_probe.py` constructs identical nominal scores for two groups, with opposing contribution offsets. Score MMD is zero and has zero offset gradient; joint MMD detects the difference and has a nonzero gradient. A 200-step optimization of this **known synthetic scalar offset**, not a detector, removes the constructed sensitivity without changing nominal scores. Numerical finite-difference gradient checks pass.

The initial toy has zero nominal FPR/FNR gaps and 0.50 gaps under rho=0.1. After offset optimization, those toy gaps are zero. These are deliberately constructed counterexamples, **not measured CES-FL improvements, performance targets, or evidence of SOTA**. See the dated JSON receipts.

`audit_checkpoint.py` verifies pinned input/source hashes, restricts inputs to the checkpoint's original TRAIN/validation manifest hashes, strictly reloads the model and performs inference only. It exports aggregate contribution shifts, fixed-bandwidth kernel diagnostics, subgroup supports and gain-stress error rates. It replays nominal validation AUC, preserves calibration and never updates parameters, writes checkpoints, trains, or reads a test split. Its tests exercise a real Xception forward/reload path with synthetic images and random weights. Production cluster weights and CUDA remain untested locally.

## Required evidence before making a new training request

1. Run the read-only audit on completed R0, R1 and R2 checkpoints, using the three pinned configs in this directory. R1 is the primary partially trained BCE reference; the others check whether observations depend on training policy. The declared 163 train/validation contributor overlaps remain a benchmark exception, not identity independence.
2. Confirm that nominal validation AUC replay matches the recorded checkpoint before interpreting stress results. Report disagreement; do not silently retune thresholds or choose another epoch.
3. Inspect group/class contribution shifts, including the descriptive cancellation amount `|delta_global| + |delta_local| - |delta_global + delta_local|` relative to an equal-group class mean. A large value alone does not establish unfairness or justify suppressing features.
4. Check whether cancellation relates to actual error disparities under predeclared, label-preserving validation corruptions. Stratify by manipulation family and account for minority sample support and shared source/identity clusters. The provided audit is descriptive: cluster uncertainty intervals and corruption audits are still pending. Do not present its MMD magnitudes as statistical significance or directly compare one- versus two-dimensional kernel values.
5. Proceed to a bounded training pilot only if a practically relevant failure exists and is not explained solely by composition or measurement noise. Otherwise reject this mechanism; do not add it merely to increase a novelty rating.

If supported, the controlled training comparison should be BCE, BCE+score alignment, BCE+separate marginal alignment, and BCE+joint contribution alignment. Use identical encoder initialization, parameter budget, class/group batch draws, optimizer steps and selection policy across all four. Score alignment is an essential comparator; a win over BCE alone does not establish value beyond existing fairness alignment. Choose one primary demographic partition consistently for sampling, loss and primary reporting (gender × race for Ding-style comparisons), with other axes reported separately.

Success requires reduced FPR/FNR disparities at a single threshold, preserved overall and worst-group AUC, and consistent results across multiple seeds and manipulation families. A future protocol must specify numerical utility floors, hyperparameter budget and confidence procedure before launching. This review intentionally supplies no ready-to-run detector training config. Test sets remain locked until that protocol is fixed.

Failure cases to actively check: both branches collapse to constants; one branch becomes unused; branch evidence changes without improving prediction errors; discriminatory cues move within the patch map while its mean aligns; kernel bandwidth gives vanishing gradients; subgroup labels are noisy; minority identities are repeatedly sampled; legitimate class-conditional evidence differs across demographics. Aligning internal evidence is a stronger restriction than outcome parity and can harm detection.

## State and artifacts

- Current decision: **research audit first; full model implementation/training held**.
- Current novelty rating: 3/10 baseline, 5/10 revised hypothesis; neither is a claim of CVPR acceptance readiness.
- `CLUSTER_AGENT_PROMPT.md`: read-only operator handoff; explicitly prohibits new training.
- `request.json`: audit scope and unresolved research gates.
- `software_checks.json`: actual local checks, source/config hashes and environment.
- Original diagram/model code and completed results are unchanged.
