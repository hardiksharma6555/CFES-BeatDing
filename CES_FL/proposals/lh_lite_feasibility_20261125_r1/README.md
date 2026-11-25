# Lightweight CES-FL feasibility — 25 November 2026

The under-10-MB storage requirement is demonstrated with actual serialized, executable candidate architectures. Similar detection/fairness performance and ESP32 deployment remain unverified. This is a separate deployment study; the existing CVPR paper and completed experimental results are unchanged.

## Verified local resource results

All weights are randomly initialized. Calibration is synthetic. These are resource/software results, not detector accuracy or fairness results. MB means 1,000,000 bytes. Export is frozen TorchScript FP32 or genuinely quantized QNNPACK INT8; no ZIP compression is used to meet the size limit.

| Candidate | Input | Parameters | FP32 file MB | INT8 file MB | Conv/linear MACs M |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 ReLU, width 1.0 | 224×224 | 2,225,153 | 8.867696 | 2.317544 | 299.495552 |
| MobileNetV2 ReLU, width 0.5 | 160×160 | 688,961 | 2.754034 | 0.765992 | 48.905280 |
| Sequential depthwise CNN | 96×96 | 123,145 | 0.501000 | 0.150267 | 7.190688 |

The native plain Xception reference has 20,809,001 parameters and 4,550.833120 M counted MACs at 224×224. Its raw FP32 parameter storage is 83.236004 MB, not a measured serialized file size. The full-width compact candidate therefore has about 89.3% fewer parameters and 15.2× fewer counted MACs. This does not establish the same factor in runtime.

Forward/backward checks passed for all three architectures. FP32 BatchNorm folding preserves outputs within tolerance. Both serialized formats were reloaded and their outputs compared with the corresponding in-memory model. All live calibration observers were checked. Torchvision has unused skip observers in blocks with no residual connection; they are explicitly identified, not mistaken for uncalibrated live paths. The actual report is in `evidence/RESOURCE_REPORT.json`. The published profile log has trailing whitespace removed; the original local log is preserved outside the repository.

Four numerical/gradient tests check binary distillation against an independent two-class KL calculation, equal-cell weighting, teacher detachment, and matched-response behavior. The profile is pinned to PyTorch 2.8.0 / torchvision 0.23.0; the eager quantization API used for this host check is deprecated and is not the chosen MCU production runtime.

## Recommended model and study

Start with **the 224×224 full-width candidate**, which already satisfies the size limit in FP32. INT8 provides substantial storage headroom without initially discarding high-frequency information through aggressive spatial downsampling. The half-width and micro candidates are later hardware tradeoff tests.

Use a completed, frozen, validation-selected native teacher for each dataset. Do not select an oracle teacher from the best-per-TEST tables. The original FF++ L_H and later FairForge-2 cohorts are different experiments; their checkpoints, input normalization, selection policies and scores cannot be interchanged. The upstream branch `results/ff2-fully-matched-interim-20261110` at `43c2272448` reports completed seven-method, three-downstream-seed cohorts, but checkpoint bytes and current private availability have not been inspected here.

The proposed training direction is **fairness-aware preservation of forensic responses under compression**:

1. Keep ground-truth classification on original TRAIN images. Match a plain compact BCE control and preserve sampling/split roles.
2. Distill teacher binary scores. Compare ordinary KD with group-balanced, class-conditional KD; the included primitive averages over supplied class-by-group cells. Absent/sparse groups need the existing support policy.
3. Keep direct student fairness control so the student is not merely required to reproduce teacher biases. Native H is a candidate baseline objective, not a proven fairness guarantee.
4. Add paired response distillation on the same sham/treated TRAIN source and operator/strength condition. Match `asinh(z_treated - z_sham)`, with detached teacher targets. The implemented function is a tested loss primitive, not yet an integrated training run. Different teacher/student resolutions must use the same source-space intervention and documented preprocessing.
5. Quantize using representative TRAIN-only calibration. Measure the exported INT8 model directly. If group or manipulation-family degradation is material, test target-compatible quantization-aware fine-tuning, using controls that isolate its contribution.

Teacher logits/responses can be cached under source, checkpoint, preprocessing and intervention hashes to save repeated teacher forward passes. Do not cache a different stochastic view and silently pair it with a student input. Initial paired development should use seed17; multiple seeds follow only if the controlled result warrants the cost. This proposal has no new GPU allocation and does not reuse the separate historical FF2 budget.

The included `study_plan.json` is **a research specification, not a runnable trainer config**. Its candidate learning rate, loss weights and noninferiority tolerances are engineering proposals, not tuned or accepted results. A practical proposed target is at most one percentage-point loss in overall and worst-group AUC, with subgroup error constraints at a common pooled-FPR target. Final tolerances must be frozen before evaluation. Accuracy alone is insufficient: compare AUC, es-AUC, F_FPR, F_DP, subgroup FNR, worst-group performance and uncertainty on the same data. Classify unsupported uncertainty as inconclusive rather than relaxing the criterion.

## ESP32 feasibility boundary

An under-10-MB file does not imply that a microcontroller can execute it. The [ESP32-S3 datasheet](https://documentation.espressif.com/esp32-s3_datasheet_en.pdf) lists 512 KB SRAM; PSRAM depends on the chip/module. An [ESP32-S3-EYE](https://documentation.espressif.com/esp-who/master/docs/en/get-started/ESP32-S3-EYE_Getting_Started_Guide.md) has 8 MB PSRAM and 8 MB flash. Firmware, OTA partitions, camera buffers, decoder/cropper, activations and kernel scratch also consume resources.

For our profiles, the largest single hypothetical INT8 layer output is 1,204,224 bytes for full MobileNetV2, 307,200 bytes for the half-width model, and 55,296 bytes for the micro model. These are **not** peak inference-memory estimates. Allocation lifetimes and kernel scratch must be measured with the actual runtime; flash storage may differ from working RAM.

[Espressif provides a MobileNetV2 deployment example](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_mobilenetv2.html), supporting this as a plausible architecture family. Its [quantization scheme](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html) is platform-specific. The files produced here use QNNPACK and cannot simply be copied to ESP-DL. A genuine ESP claim needs ESP-PPQ export, supported-op verification, actual `.espdl` bytes, measured memory/latency and output agreement on the selected board. No USB serial device matching common MCU interfaces was visible during this check; no physical board execution occurred. No ESP32 FPS or energy result is claimed.

The original ESP32 is a more restrictive target than the S3. Prefer the S3 with PSRAM for an initial MCU test, subject to the user's hardware choice. A lower-resolution or sequential-patch strategy is a research option, not evidence that the tiny model will preserve forensic accuracy. MCUNet shows why architecture/runtime co-design matters for microcontrollers ([primary paper](https://arxiv.org/abs/2007.10319)).

All prototypes accept an already cropped RGB face. An on-device face detector, decoding and crop pipeline must be included in end-to-end deployment measurements. Detecting a digitally manipulated file and detecting a screen recaptured by an ESP32 camera are different distributions and potentially different tasks; benchmark them separately.

## Novelty and the TIM target

MobileNet plus knowledge distillation and INT8 conversion is not a novel contribution by itself. Lightweight quantized deepfake detection already has direct prior work, including [Tao et al., IJCAI 2025](https://ijcai.org/proceedings/2025/59). A defensible hypothesis is that matching group-conditional **intervention responses**, while directly constraining student errors during target-aware quantization, better preserves fairness across capture and manipulation conditions than ordinary KD/PTQ. A focused novelty review and controlled evidence are still needed; no novelty rating or success probability is established.

[IEEE TIM's scope](https://ieee-ims.org/publication/ieee-tim) centers on instrumentation and measurement. Its [author guidance](https://ieee-ims.org/publication/ieee-tim/information-authors) identifies ML/classification work without demonstrated measurement improvement as a scope concern. A smaller deepfake classifier alone is therefore a weak TIM submission. Define an actual camera-based measurement/integrity application and show how the detector improves that system's measurement performance. Report acquisition conditions, reference truth, calibration, uncertainty, error propagation, repeatability, end-to-end latency, flash/RAM, and measured energy. Speed/power tables alone do not establish an I&M contribution or guarantee acceptance.

## Reproduce the present software checks

From the repository root, with the recorded PyTorch/torchvision environment:

```sh
python -m unittest discover -s CES_FL/tests -p test_lite_distillation.py -v
python CES_FL/src/ces_fl_lite_feasibility.py profile --out /tmp/cesfl_lite_NEW_OUTPUT
```

Use a new output directory. The command writes explicitly named `RANDOM_*` artifacts and must not be used to produce claimed detector predictions. No model downloads occur. Local production training/evaluation, target conversion, face-pipeline integration, and board measurements remain required to establish the user's complete performance/deployment objective.
