#!/usr/bin/env bash
# Запуск исходного train_gpt2.py «как есть» на одной GPU сервера.
# Код не меняется: правятся только значения в Hyperparameters, в копии скрипта внутри каталога прогона.
#
#   bash scripts/run_asis.sh                      # дефолты ниже
#   DEVICE_BS=4 NUM_ITERS=200 bash scripts/run_asis.sh
#   DISABLE_COMPILE=1 bash scripts/run_asis.sh
#
# Ожидает данные в source_code/data/fineweb10B (python source_code/data/cached_fineweb10B.py 11).
set -euo pipefail

# В WSL nvidia-smi лежит в /usr/lib/wsl/lib, а этот каталог не всегда есть в PATH.
[ -d /usr/lib/wsl/lib ] && export PATH="$PATH:/usr/lib/wsl/lib"

DEVICE_BS=${DEVICE_BS:-8}
NUM_ITERS=${NUM_ITERS:-500}
WARMDOWN=${WARMDOWN:-150}
VAL_EVERY=${VAL_EVERY:-50}
DISABLE_COMPILE=${DISABLE_COMPILE:-0}
NPROC=${NPROC:-1}

REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC="$REPO/source_code"
RUN_NAME=${RUN_NAME:-asis-bs${DEVICE_BS}-it${NUM_ITERS}-$(date +%Y%m%d-%H%M%S)}
RUN_DIR="$HOME/results/$RUN_NAME"

mkdir -p "$RUN_DIR"
cp "$SRC/train_gpt2.py" "$RUN_DIR/train_gpt2.py"
ln -sfn "$SRC/data" "$RUN_DIR/data"

python - "$RUN_DIR/train_gpt2.py" "$DEVICE_BS" "$NUM_ITERS" "$WARMDOWN" "$VAL_EVERY" <<'EOF'
import re, sys
p, *vals = sys.argv[1:]
s = open(p).read()
for name, val in zip(['device_batch_size', 'num_iterations', 'warmdown_iters', 'val_loss_every'], vals):
    s, n = re.subn(rf'({name}\s*:\s*int\s*=\s*)\d+', rf'\g<1>{val}', s)
    assert n == 1, name
open(p, 'w').write(s)
EOF

{
    echo "run: $RUN_NAME"
    echo "git: $(git -C "$REPO" rev-parse HEAD) $(git -C "$REPO" status --porcelain | wc -l) uncommitted"
    echo "device_batch_size=$DEVICE_BS num_iterations=$NUM_ITERS warmdown_iters=$WARMDOWN val_loss_every=$VAL_EVERY disable_compile=$DISABLE_COMPILE nproc=$NPROC"
    python -c 'import torch, sys; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'
    nvidia-smi || echo "nvidia-smi not found"
} > "$RUN_DIR/env.txt" 2>&1

cd "$RUN_DIR"
export TORCHDYNAMO_DISABLE=$DISABLE_COMPILE
gpu-run torchrun --standalone --nproc_per_node="$NPROC" train_gpt2.py 2>&1 | tee "$RUN_DIR/stdout.log"
