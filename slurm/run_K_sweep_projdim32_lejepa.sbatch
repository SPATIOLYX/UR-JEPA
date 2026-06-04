#!/bin/bash
# K-ablation (K = n_scales) at the D=32 optimum n=7, LeJEPA-exact
# recipe (single-GPU bs=256, lr=2e-3, eta_min=1e-3 literal, 800
# epochs). Trimmed K values: keeps only the well-performing K cells
# from the D=16 sweep and adds the right-shoulder values 10, 11, 13,
# 14 that the D=16 K_sweep_v2 is currently filling in.
#
# Dropped from the original 17-task design:
#   K=1  -- degenerate at D=16 (0.9057, single (Delta log theta)^2
#           term too coarse). Unlikely to behave better at D=32.
#   K=5  -- local minimum at D=16 (0.9090, ~0.0035 below plateau peak).
#   K=7  -- mild dip at D=16 (0.9101).
#
# Kept (good D=16 plateau cells):
#   K=2  (0.9125 +/- 0.0024 at D=16)
#   K=3  (0.9126 +/- 0.0015 at D=16)
#   K=9  (0.9127 +/- 0.0017 at D=16)
#   K=12 (0.9139 1-seed at D=16; now 3 seeds at D=32 for clean read)
#
# Added (right-shoulder cells, untested at D=16 until K_sweep_v2 lands):
#   K=10, K=11, K=13, K=14 (3 seeds each)
#
# 24 tasks total = 8 K values x 3 seeds. ~3-4h per task on H100;
# ~80 GPU-h total, ~20h elapsed on a 4-GPU node.
#
# Submit:    sbatch run_K_sweep_projdim32_lejepa.sbatch
# Summarize: python scripts/aggregate_results.py \
#                $SCRATCH/projects/UR-JEPA/runs/k_sweep_projdim32_lejepa
#
#SBATCH --job-name=K_sweep_pd32
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
#SBATCH --array=0-23

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

# --- task table (24 tasks, K-major, 3 seeds per K) -------------------------
# Only "good candidates" for K based on D=16 results: drop K=1, 5, 7
# (degenerate or local minima) and keep K=2, 3, 9, 12 from the proven
# plateau, plus K=10, 11, 13, 14 as new right-shoulder cells.
KS=(    2 2 2  3 3 3  9 9 9  10 10 10  11 11 11  12 12 12  13 13 13  14 14 14)
SEEDS=( 0 1 2  0 1 2  0 1 2  0  1  2   0  1  2   0  1  2   0  1  2   0  1  2)

K="${KS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUT="$SCRATCH/projects/UR-JEPA/runs/k_sweep_projdim32_lejepa/ur_cglt_n7_K${K}_s${SEED}"

echo "--- task $SLURM_ARRAY_TASK_ID: ur_cglt n=7 K=$K seed=$SEED proj_dim=32 out=$OUT ---"
nvidia-smi || true

# LeJEPA-exact recipe (matches run_lejepa_exact.sbatch and
# run_K_sweep.sbatch). Only +proj_dim differs (16 -> 32); n=7 is the
# D=32 optimum confirmed by run_projdim32_n_sweep_lejepa, K varies.
python scripts/pretrain.py \
  +regularizer=ur_cglt \
  +reg_scale=1000 \
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
  +n=7 \
  +n_scales="$K" \
  +anchors=128 \
  +lambda_ad=0.1 \
  +num_slices=256 \
  +wandb=false
