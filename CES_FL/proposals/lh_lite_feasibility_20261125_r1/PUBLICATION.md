# Reviewed lightweight pilot handoff

Current runnable code/config commit: `1d43db947b6d04f8373ab115eee61c0a331ef7e4`.

Publish repository: `https://github.com/hardiksharma6555/CFES-BeatDing`.
Branch: `experiment/lh-lite-feasibility-20261125-r1`.
The main `hardikbha` repository remains read-only for the current authenticated account; the fork has verified push access.

The existing production CPU preparations at `/home/sachin.chaudhary/ces_fl_runs/lite_cpu_checks_20261125_391892433106` match the scientific source files in the runnable commit. All three preparations and the real-input full-objective CPU backward passed. No student optimization or GPU run has begun. Four additional GPU-hours on one H100 were requested; approval is pending and must not be inferred from this publication.

After approval, fetch the runnable commit into a detached checkout:

```sh
git fetch https://github.com/hardiksharma6555/CFES-BeatDing.git experiment/lh-lite-feasibility-20261125-r1
git worktree add --detach /home/sachin.chaudhary/ces_fl_sources/lite_1d43db9 1d43db947b6d04f8373ab115eee61c0a331ef7e4
```

Follow `GPU_OPERATOR_PROMPT.md`; use the prepared one-GPU PBS launcher and site-verified CUDA binding. The source checkout and all new output directories must be separate from existing runs. CPU postprocessing is separate from the GPU allocation and measures both actual FP32 and INT8 deployment artifacts.

This supersedes the earlier launcher: incomplete epoch-boundary stops now fail the dependent-stage check even when training returns exit0. The scientific model/training source and prior CPU checks are unchanged. Eight completion checks passed.
