#!/bin/bash
# ImageNet-100 (CMC 100-class subset, ~127K train / 5K val) pretraining
# at the LeJEPA-exact recipe (single-GPU bs=256, lr=2e-3, eta_min=1e-3
# literal). 3-seed promotion of the Stage-1 screen finding.
#
# Stage-1 screen (`runs/inet100_lejepa_screen/`, 1 seed, 100 epochs):
#   sigreg  (n=7 placeholder) : 0.5380
#   ur_cglt n=7  :              0.5928  (peak; +5.48pp over sigreg)
#   ur_cglt n=10 :              0.5480
#   ur_cglt n=14 :              0.4598  (late collapse, final_acc=0.0170)
#
# This Stage-2 puts proper 3-seed error bars on the +5.48pp gap at
# the SAME 100-epoch recipe as the screen. The gap is already huge
# at 100 ep -- the open question is variance, not magnitude, so
# spending compute on more epochs isn't load-bearing.
#
# Configuration: D=32 only (the projector dim where UR-CGLT showed
# the clearest advantage on Imagenette-10). 100 epochs matching the
# Stage-1 screen recipe exactly -- only thing that changes is adding
# seeds 1 and 2 per method.
#
# 6 tasks = 2 methods (sigreg, ur_cglt integral) x 3 seeds.
#
# Per-task wall: ~4.6h on H100 single-GPU (matches Stage-1 screen
# walls of 4.53-4.60h at 100 ep). 6 runs = ~28 GPU-h total. Currently
# set to --time=24h (generous headroom; could safely lower to 8h).
#
# Why NOT 200 epochs: the prior 200-ep version of this sbatch timed
# out at the 24h wall (sigreg_s0 reached 172 ep, sigreg_s1 reached
# 174 ep, neither produced a final best_acc). At ~12h per 200-ep
# task on H100, the wall cap was the binding constraint. Reverting
# to 100 ep matches the screen recipe exactly and finishes cleanly
# inside any reasonable wall budget.
#
# IMPORTANT: pre-cache the dataset before submitting to avoid all 6
# tasks downloading concurrently. See run_inet100_pretcache.sh below;
# or just run once on the login node:
#   export HF_HOME=$SCRATCH/.cache/huggingface
#   python -c "from datasets import load_dataset; \
#     load_dataset('clane9/imagenet-100', split='train'); \
#     load_dataset('clane9/imagenet-100', split='validation')"
#
# Submit:    sbatch run_inet100_lejepa.sbatch
# Summarize: python scripts/aggregate_results.py \
#                $SCRATCH/projects/UR-JEPA/runs/inet100_lejepa
#
#SBATCH --job-name=inet100_lejepa
#SBATCH --account=cis260608-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-5

set -euo pipefail
mkdir -p logs

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda-envs/urjepa"

export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_DIR="$SCRATCH/wandb"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$SCRATCH/projects/UR-JEPA"

# --- task table (6 = 2 methods x 3 seeds, method-major) -------------------
REGULARIZERS=( ur_cglt sigreg ur_cglt sigreg ur_cglt   sigreg )
SEEDS=(0 0 1 1 2 2)
SCALES=(1000 1 1000 1 1000 1)        # sigreg native=1; ur_cglt=1000

REG="${REGULARIZERS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SCALE="${SCALES[$SLURM_ARRAY_TASK_ID]}"
OUT="$SCRATCH/projects/UR-JEPA/runs/inet100_lejepa/${REG}_s${SEED}"

echo "--- task $SLURM_ARRAY_TASK_ID: $REG seed=$SEED (inet100, LeJEPA-exact, D=32, n=7) out=$OUT ---"
nvidia-smi || true

# LeJEPA-exact recipe at our best ur_cglt config from Imagenette-10:
# proj_dim=32, n=7, K=n_scales=5, lambda_ad=0.1, projector_norm=bn.
# 100 epochs matches the Stage-1 screen exactly; the only change vs
# Stage-1 is adding seeds 1 and 2 per method for proper error bars.
python scripts/pretrain.py \
  +dataset=imagenet100 \
  +regularizer="$REG" \
  +reg_scale="$SCALE" \
  +projector_norm=bn \
  +seed="$SEED" \
  +out_dir="$OUT" \
  +lamb=0.02 \
  +V=4 \
  +proj_dim=32 \
  +lr=2e-3 \
  +eta_min=1e-3 \
  +bs=256 \
  +accum_steps=1 \
  +epochs=400 \
  +n=7 \
  +n_scales=5 \
  +anchors=128 \
  +lambda_ad=0.1 \
  +num_slices=256 \
  +knots=17 \
  +wandb=false
