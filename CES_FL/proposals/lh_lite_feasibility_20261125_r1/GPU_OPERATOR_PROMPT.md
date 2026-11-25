# Lightweight student pilot — bounded execution instructions

This is a NEW study. Do not use the historical FairForge-2 72-hour budget. A request for up to four additional GPU-hours on one H100 is pending; a source publication or these commands are not a launch approval.

Use branch `experiment/lh-lite-feasibility-20261125-r1` and the exact reviewed commit recorded in the handoff. Create a detached worktree; preserve all current sources, jobs, completed results, and private data. Read `CES_FL/AGENTS.md`, this proposal, and the actual source. Do not modify a running checkout or choose models using TEST scores.

## Bound inputs

The three exact configurations are `configs/bce.json`, `configs/kd.json`, and `configs/fair_response.json`. Each uses the same full-width, ReLU MobileNetV2 with one binary head, ImageNet trunk initialization, seed17, 224×224 original crops, ten epochs, batch32, AdamW learning rate 1e-4 and weight decay 1e-4. All student layers train; BatchNorm running buffers are frozen. The initialization SHA is `7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f` and exists at the bound cluster path. It is an ordinary public ImageNet checkpoint; the CLI never downloads weights.

The completed Xception L_H teacher is pinned to SHA `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0`. Its native validation cache, manifests, original preflight inventory and qualified procedural recipe are bound and checked. This pilot is FF++ TRAIN/validation only. It does not launch Ding, FairForge-2, ResNet, TEST evaluation, or image generation.

The BCE control retains inverse class-by-ids8 weighting. KD adds ordinary sample-mean binary teacher distillation, temperature2 and coefficient0.5. The fairness/response arm uses class/group-balanced KD, all original native H penalties, and real-source matched response KD with coefficient0.1. Thus the last arm tests an objective package; it does not isolate each added component. Teacher parameters remain frozen. The student converts native RGB [-1,1] tensors to the ImageNet normalization expected by its trunk. Procedural pairs use the original native geometry before student normalization.

## Required checks before the GPU launch

1. Confirm the new compute allowance and actual scheduler allocation. Do not guess a free card or reuse an unrelated lease.
2. Run the local numerical/lifecycle tests in the supported environment. Recorded local tests used torch2.8.0/torchvision0.23.0; cluster metadata identifies torch2.7.0/torchvision0.22.0, so require actual cluster checks as well.
3. Prepare all three configs and complete a zero-update production TRAIN backward check. The supplied `check_production_cpu.py --out NEW_PRIVATE_ROOT` performs this. A fresh prepared receipt must match the exact trainer, native source, and compact-model source hashes. Existing receipts can be reused only when their hashes and bound inputs agree.
4. Read actual failures. A short environment-query command timed out during cluster import; this was not GPU training. The independently launched bounded CPU-check process has separate PID, log and exit-code evidence. Do not launch a duplicate if that process is still live.

## GPU stage

After approval and CPU PASS, submit `CES_FL/cluster/lite_three_arm_one_gpu.pbs`. It requests one GPU, eight CPU cores, 32GB host memory, and a four-hour walltime. Set these values explicitly:

```text
CES_LITE_SOURCE=<detached immutable source checkout>
CES_LITE_COMMIT=<exact reviewed commit>
CES_LITE_PREPARED=<CPU-check output root with CPU_PRODUCTION_CHECKS.json>
CES_LITE_RUNS=<new absolute output root>
CUDA_VISIBLE_DEVICES=<site-verified device assignment for this PBS job>
```

The job runs three fresh CUDA backward checks and the three arms sequentially. It records PBS, source and CUDA metadata. Each arm has a 3,600-second runtime guard; a mid-epoch budget stop preserves a partial checkpoint and refuses automatic resume. A scheduler timeout may also leave an incomplete run. Preserve that state and report it; do not call it a completed ten-epoch experiment. Do not expand budgets or alter hyperparameters to chase validation scores.

## CPU postprocessing

After verified GPU completion, use a separate CPU allocation (four cores, sufficient memory, no GPU):

```sh
python CES_FL/proposals/lh_lite_feasibility_20261125_r1/postprocess.py \
  --prepared <same prepared root> --runs <completed GPU root> --out <new CPU root>
```

This exports QNNPACK INT8 with 1,024 class/group-stratified TRAIN calibration images, evaluates the actual reloaded artifact on validation, and computes paired connected-contributor bootstrap comparisons. It also rebuilds and evaluates an FP32 CPU artifact so INT8-minus-FP32 comparisons use the same host/preprocessing path. Each deployment format receives one global validation-real threshold; fixed0.5 and float-threshold INT8 diagnostics are also reported. No TEST data are accepted by this helper. Do not label the TorchScript files as ESP32/ESP-DL artifacts.

## What to return

Publish source/config hashes, allocation/accounting, completion state, selected epochs, exported byte sizes, overall AUC/FPR/FNR, gender/race/joint/age/skin metrics, worst-group AUC and every bootstrap usable count. Compare each student with the same frozen teacher and compare the full package with BCE and ordinary KD. Keep face images, private score caches, initialization weights and trained binaries off the public repository. Sparse bootstrap groups, calibration changes, and single-seed limitations remain explicit. Under10MB alone is not completion of the accuracy or MCU-deployment claim.
