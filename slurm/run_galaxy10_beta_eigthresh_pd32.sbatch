#!/bin/bash
# 3-seed promotion of the winning eigval-threshold from the 1-seed sweep
# (run_galaxy10_beta_eigthresh_sweep.sbatch). The bare 1-seed winner
# already lands at best=0.7993 / final=0.7773 on Galaxy10, within ~1.3 pp
# of the 400-ep UR-CGLT verify (0.8124 / 0.7981) and ~1 pp ABOVE the
# original Spatiolyx published headline (0.7894 +/- 0.0034 at 200 ep).
# This sbatch puts a real 3-seed error bar on that number so it can
# replace the "beta-number collapses" negative result in the paper.
#
# BEFORE SUBMITTING:
#   1. Inspect the 1-seed sweep results (see comment at end of
#      run_galaxy10_beta_eigthresh_sweep.sbatch) to pick the winning tau.
#   2. Edit the TAU= line below to that value.
#   3. (Optional) Edit the PD= line below if running at a different
#      projector dim than the default 32. Output paths and the python
#      +proj_dim flag both derive from PD.
#
# Recipe: identical to the sweep at the chosen tau, with --array=0-2.
# Per-seed wall on H100: ~3.4h. Total compute: ~10 GPU-h on H100.
#
# Submit:    sbatch run_galaxy10_beta_eigthresh_3seed.sbatch
# Aggregate: python scripts/aggregate_results.py \
#                $SCRATCH/projects/UR-JEPA/runs/galaxy10_beta_eigthresh_3seed_pd${PD}
#
#SBATCH --job-name=galaxy10_beta_eigthresh_3seed
#SBATCH --account=cis260608-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

# === EDIT THESE LINES =================================================
TAU=1.0   # one of {0.5, 1.0, 2.0, 5.0}; set to the sweep winner
PD=32     # projector dim; UR variants peak here on Imagenette
# =====================================================================

SEED="${SLURM_ARRAY_TASK_ID}"
TAU_TAG="$(echo "$TAU" | tr '.' 'p')"
OUT="$SCRATCH/projects/UR-JEPA/runs/galaxy10_beta_eigthresh_pd${PD}/ur_beta_tau${TAU_TAG}_s${SEED}"

echo "--- array task ${SLURM_ARRAY_TASK_ID}: ur_beta seed=$SEED eigval_threshold=$TAU proj_dim=$PD (galaxy10 beta+log-trace+eig-thresh 3-seed promotion: 400 ep) out=$OUT ---"
nvidia-smi || true

python scripts/pretrain.py \
  +dataset=galaxy10 \
  +backbone=resnet18_lowres \
  +regularizer=ur_beta \
  +reg_scale=1000 \
  +projector_norm=bn \
  +seed="$SEED" \
  +out_dir="$OUT" \
  +lamb=0.02 \
  +V=4 \
  +proj_dim=${PD} \
  +lr=2e-3 \
  +eta_min=1e-3 \
  +bs=512 \
  +accum_steps=1 \
  +epochs=400 \
  +n=7 \
  +n_scales=5 \
  +anchors=128 \
  +lambda_ad=0 \
  +gamma_logtrace=0.1 \
  +eigval_threshold="$TAU" \
  +num_slices=256 \
  +wandb=false
