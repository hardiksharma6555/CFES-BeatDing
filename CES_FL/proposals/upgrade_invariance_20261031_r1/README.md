# Controlled CES-FL upgrade — 31 October 2026

This request adds a separate exploratory training module while preserving the
completed v0.2.1 source, configurations and results. No improvement over Ding is
claimed. No training, generation or scheduler action is launched by this commit.
The CPU checks exercise synthetic software fixtures; their numerical AUC values
are not research results. CUDA smoke and real-input preflight must run on the
cluster before training.

## Hypothesis and ablations

The frozen encoder may restrict forensic adaptation. Updating only the final
stage tests that hypothesis. Group DRO subsequently tests emphasis on high-loss
label × ids8 groups. Finally, matched no-change/age consistency arms distinguish
editor exposure from the additional age intervention. Group DRO is a training
loss surrogate, not an equalized-odds/FPR guarantee. Age consistency is not proof
of demographic fairness or preservation of forgery evidence.

| Config | Encoder | Original-image loss | Pair consistency |
|---|---|---|---|
| R0_frozen_bce | frozen | mean BCE | none |
| R1_partial_bce | Xception exit stage | mean BCE | none |
| R2_partial_dro | Xception exit stage | group DRO | none |
| RN_partial_dro_nochange_pilot | Xception exit stage | group DRO | no-change |
| RD_partial_dro_age_pilot | Xception exit stage | group DRO | age |

R0→R1 changes the encoder training policy; R1→R2 changes classification weighting;
R2→RN adds reconstruction consistency; RN→RD changes only the pair variant.
All configurations import the same hash-pinned detection-pretrained Xception
encoder, reset identical heads at seed17, use the same original-image sampler,
10 epochs, batch16, head learning rate0.001 and weight decay0.0001. Partial-stage
learning rate is0.00001. BatchNorm running buffers stay frozen; affine parameters
in the selected stage may train. Xception trains block12/conv3/bn3/conv4/bn4;
ResNet support trains layer4, although no new ResNet production config is bound
in this request. Global/local heads, gates and positive fusion are unchanged.

Group DRO uses exponentiated group weights with eta0.01 and original-image group
means. Absent minibatch groups have zero loss for that update. Groups require
at least20 TRAIN records; otherwise preflight rejects rather than dropping them.
Unknown metadata, if present, remains its own training group and is recorded.
The original sampler is balanced over label × ids8 in every arm. This stochastic
objective is not a population worst-group guarantee; review actual per-group
validation results and regularization before drawing conclusions.

RN/RD sample8 additional anchors per step,4 real and4 fake, independently of the
original batch RNG. Groups then sources are sampled uniformly within each label.
Age variants cycle per source. Both arms use the same anchors/direction schedule.
BCE/DRO applies only to the ordinary original batch. Pair anchors and variants
receive the symmetric asinh consistency gradient with lambda1. No teacher is
detached. Heads and selected encoder stage receive both gradients. Extra pair
forwards are logged; equal update counts do not imply equal compute.

## Pair scope and dependencies

**RN/RD are six-source diagnostics only.** Their exact input is the existing
approved manifest (12 age variants of six TRAIN originals). Sampling8 anchors
with replacement increases exposure, not independent coverage. This is not the
recommended expanded study. A separate request with at least300 reviewed TRAIN
sources, older AND younger directions for each, associated no-change/resize
controls and new exact input hashes is required for that study. The sampler
enforces `minimum_pair_sources`; raising it to300 with the current pool fails.
Do not lower that guard and call the expanded study complete.

Controls verify source/edit hashes and independent-from-original provenance.
They do not certify forensic preservation. Review generation-induced artifact
changes before using edits to support an invariance claim. No labels or approvals
are generated here. No-change reconstruction is the RN loss input; resize controls
remain diagnostic. Independent matched-control evaluation is still required for
research interpretation; this patch does not automatically produce that study.

The exact old manifests contain declared TRAIN/VAL contributor overlaps. These
are preserved and reported, not described as an identity-disjoint benchmark.
All external files, including weights and the pretraining checkpoint, are hashed
in each config. Production execution rejects missing or changed inputs. No model
or dataset is downloaded automatically.

## Local / allocated-GPU smoke

From the repository root in an environment with torch, torchvision, numpy,
Pillow and scikit-learn:

```bash
python -m unittest discover -s CES_FL/tests -v
CES_UPGRADE_DEVICE=cuda python -m unittest discover -s CES_FL/tests -p test_upgrade.py -v
```

The CUDA command fails when CUDA is unavailable; it never substitutes CPU.
Tests cover both backbone partial policies, frozen early parameters/BN buffers,
global/local score decomposition, group-DRO gradients and persistence, both
consistency paths, deterministic balanced pair/control schedules, invalid input
rejection, five tiny synthetic training runs, checkpoint reload and frozen
threshold after evaluation-label changes. These use random weights only.

## Pull and launch on the cluster

Fetch the exact commit supplied in the handoff, then create a fresh detached
worktree. Do not pull over any source used by an active job.

```bash
git fetch https://github.com/hardiksharma6555/CFES-BeatDing.git upgrade/ces-fl-controlled-invariance-20261031
CES_COMMIT=$(git rev-parse FETCH_HEAD)
CES_SOURCE_DIR=/home/sachin.chaudhary/ces_fl_sources/upgrade_${CES_COMMIT}
git worktree add --detach "$CES_SOURCE_DIR" "$CES_COMMIT"
cd "$CES_SOURCE_DIR"
CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
"$CES_PY" CES_FL/src/ces_fl_upgrade.py preflight \
  --config CES_FL/proposals/upgrade_invariance_20261031_r1/configs/R1_partial_bce.json
```

If GitHub push is unavailable, transfer the supplied Git bundle and replace the
fetch command with `git fetch /path/to/bundle upgrade/ces-fl-controlled-invariance-20261031`.
Verify the resulting commit matches the handoff before creating the worktree.

Publication uses the supervisor fork because the authenticated account cannot
push to hardikbha/CFES-BeatDing (HTTP403). The upstream main branch is unchanged.
Fetch the explicit fork URL above; fetching the upgrade branch from upstream
origin will not work until the owner merges/publishes it there.

The new `CES_FL/cluster/upgrade_one_gpu.pbs` requests ngpus=1 and runs external
input preflight plus CUDA smoke before training. It defaults to smoke-only.
The operator must inspect current jobs, avoid duplicates, and establish actual
GPU allocation/binding. Selecting an idle GPU index is not a reservation.

Submission form (fill the actual allocation-dependent values):

```text
qsub -v CES_SOURCE_DIR=<detached-worktree>,CES_TRAIN_CONFIG=CES_FL/proposals/upgrade_invariance_20261031_r1/configs/R1_partial_bce.json,CES_OUTPUT_DIR=<new-absolute-output>,CUDA_VISIBLE_DEVICES=<verified-binding>,CES_SMOKE_ONLY=1 <detached-worktree>/CES_FL/cluster/upgrade_one_gpu.pbs
```

After the smoke-only job succeeds, submit with `CES_SMOKE_ONLY=0` and a new output
directory to train that arm. Recommended order is R0/R1, then R2, then the RN/RD
diagnostic pair. All arms are separate jobs; no launcher automatically starts
the next experiment. Exact same configurations for their matched comparisons
must be retained even when one result disappoints. Preserve outputs and failures.

Checkpoint selection remains validation AUC only; calibration remains the
selected checkpoint's validation-real empirical FPR budget. No additional
validation fairness criterion is silently introduced. Review subgroup support,
fairness and AUC; any new selection criterion needs a separate prespecified study.
Test evaluation is separate, using `ces_fl_upgrade.py evaluate` with the selected
checkpoint and locked manifest. This exporter retains fixed0.5 and calibrated
metrics but does not yet add the benchmark bootstrap CIs. Matched test protocols,
repeated seeds and uncertainty estimates are prerequisites for a win claim.

## Sources informing the proposal

- Sagawa et al., group DRO and regularization: https://arxiv.org/abs/1911.08731
- Jung et al., counterfactual consistency need not imply group fairness:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/65b4660c1fee9842472e1557b582c940-Abstract-Datasets_and_Benchmarks_Track.html

The generated architecture figure is illustrative; executable source and exact
configs are authoritative. Performance remains unvalidated after software smoke.
