# Reviewed lightweight pilot handoff

Runnable code/config commit: `b67400c3588dc79d1d85a8fe5fa1baf4497c5c60`.

Publish repository: `https://github.com/hardiksharma6555/CFES-BeatDing`.
Branch: `experiment/lh-lite-feasibility-20261125-r1`.
The main `hardikbha` repository remains read-only for the current authenticated account; the fork has verified push access.

The existing production CPU preparations at `/home/sachin.chaudhary/ces_fl_runs/lite_cpu_checks_20261125_391892433106` match the scientific source files in the runnable commit. All three preparations and the real-input full-objective CPU backward passed. No student optimization or GPU run has begun. Four additional GPU-hours on one H100 were requested; approval is pending and must not be inferred from this publication.

After approval, fetch the runnable commit into a detached checkout:

```sh
git fetch https://github.com/hardiksharma6555/CFES-BeatDing.git experiment/lh-lite-feasibility-20261125-r1
git worktree add --detach /home/sachin.chaudhary/ces_fl_sources/lite_b67400c b67400c3588dc79d1d85a8fe5fa1baf4497c5c60
```

Follow `GPU_OPERATOR_PROMPT.md`; use the prepared one-GPU PBS launcher and site-verified CUDA binding. The source checkout and all new output directories must be separate from existing runs. CPU postprocessing is separate from the GPU allocation and measures both actual FP32 and INT8 deployment artifacts.
