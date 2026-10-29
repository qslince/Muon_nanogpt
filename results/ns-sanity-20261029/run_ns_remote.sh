#!/usr/bin/env bash
set -u
umask 077
run_dir="$1"
trap 'printf "%s\n" "$?" > "$run_dir/runner.exit"' EXIT
cd "$HOME/workspace/project" || exit 1
source "$HOME/venvs/project/bin/activate" || exit 1
git rev-parse HEAD > "$run_dir/commit.txt"
git status --short > "$run_dir/git-status.txt"
python -m pip freeze > "$run_dir/requirements-frozen.txt"
printf 'Running CPU control\n'
python scripts/check_newton_schulz.py --device cpu --dtypes fp64 fp32 bf16 fp16 --output-dir "$run_dir/cpu"
cpu_rc=$?
printf '%s\n' "$cpu_rc" > "$run_dir/cpu.exit"
if test "$cpu_rc" -ne 0; then
    printf 'CPU control failed; GPU run not started\n'
    exit "$cpu_rc"
fi
printf 'Submitting CUDA check to the shared GPU queue\n'
gpu-run python scripts/check_newton_schulz.py --device cuda --dtypes fp64 fp32 bf16 fp16 --output-dir "$run_dir/cuda"
gpu_rc=$?
printf '%s\n' "$gpu_rc" > "$run_dir/cuda.exit"
exit "$gpu_rc"
