#!/bin/bash
# SIGReg baseline at proj_dim=32 under the LeJEPA-exact recipe
# (single-GPU bs=256, lr=2e-3, eta_min=1e-3 literal, 800 epochs).
#
# Anchors the proj_dim=32 comparison. Without this, the recent
# UR-CGLT @ D=32, n=7 result (0.9141 +/- 0.0014) has no
# matched-recipe SIGReg baseline at the same projector dim, so we
# can only compare it against SIGReg @ D=16 (0.9130 +/- 0.0008),
# which leaves the projector-dim variable confounded with the
# regularizer variable.
#
# 3 tasks = sigreg x 3 seeds, paired to the existing
# `runs/projdim32_n_sweep_lejepa/ur_cglt_n7_s{0,1,2}` cells so the
# downstream aggregator can produce a clean side-by-side at D=32.
#
# Per-seed wall: ~3-4h on H100 single-GPU.
# 3 runs -> ~10-12 GPU-h total, ~3-4h elapsed on a 4-GPU node.
#
# Submit:    sbatch run_sigreg_projdim32_lejepa.sbatch
# Summarize: python scripts/aggregate_results.py \
#                $SCRATCH/projects/UR-JEPA/runs/sigreg_projdim32_lejepa
#
#SBATCH --job-name=sigreg_pd32
#SBATCH --account=cis260608-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-2

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

# --- task table (3 = 3 seeds) ---------------------------------------------
SEEDS=(0 1 2)

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUT="$SCRATCH/projects/UR-JEPA/runs/sigreg_projdim32_lejepa/sigreg_s${SEED}"

echo "--- task $SLURM_ARRAY_TASK_ID: sigreg seed=$SEED proj_dim=32 (LeJEPA-exact) out=$OUT ---"
nvidia-smi || true

# Single-GPU plain python (no torchrun) so pretrain.py takes the
# non-DDP code path -- the regularizer sees the 256-sample batch
# directly, exactly as in LeJEPA's MINIMAL.md. Only +proj_dim differs
# from run_lejepa_exact.sbatch's sigreg arm. The ur_cglt-only kwargs
# (n, lambda_ad, anchors) are inert for sigreg but kept for parity
# with the sister sbatches.
python scripts/pretrain.py \
  +regularizer=sigreg \
  +reg_scale=1 \
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
  +epochs=800 \
  +n=8 \
  +n_scales=5 \
  +anchors=128 \
  +lambda_ad=0.1 \
  +num_slices=256 \
  +knots=17 \
  +wandb=false
