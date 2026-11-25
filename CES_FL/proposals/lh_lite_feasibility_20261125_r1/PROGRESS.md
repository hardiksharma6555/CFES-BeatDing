# Goal progress: lightweight deployment model

Status: resource feasibility verified; performance and hardware validation pending.

This first goal turn made concrete progress: three implemented architectures, six measured exports with reload/forward checks, finite-gradient checks, independent loss tests, a native Xception operation-count reference, and current primary-source feasibility/scope research. No task-accuracy result or on-device result exists for the students. No production training or GitHub push was performed.

The original goal remains active. A near-teacher detection/fairness result is not established. Required next work: bind teacher and TRAIN/validation data, integrate the candidate training/evaluation pipeline, run controlled paired validation (including actual target quantization), and confirm board/memory/latency if ESP32 deployment is required. No prior no-progress/blocker turns exist for this new goal.

Optional hardware preference was asked asynchronously; no answer was available when this proposal was recorded. Working priority is under 10 MB at 224×224, with ESP32-S3/PSRAM as the additional hardware path.
