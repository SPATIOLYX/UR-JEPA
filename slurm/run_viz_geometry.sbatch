#!/bin/bash
# Geometry-aware visualization of the projector output distribution for
# matched-recipe LeJEPA(SIGReg) vs UR-JEPA(CGLT) checkpoints.
#
# Loads ckpt-final.pt for each method, runs the backbone+projector on the
# validation split, and writes a 2x2 diagnostic figure
# (eigenvalue spectrum, per-dim Shapiro-Wilk W, Q-Q plot, marginal
# histogram) plus a raw .npz dump to OUT.
#
# Default targets are the ImageNet-100 Stage-2 single-seed (seed 0)
# checkpoints. Switch DATASET / CKPT_* to compare Galaxy10 or Inet10
# pairs.
#
# Single GPU is plenty: the script touches each ckpt for a few hundred
# val batches and then the heavy work (PCA + per-dim Shapiro) is on CPU.
# Wall time is dominated by ckpt forward passes, ~5-10 min per ckpt at
# bs=128 / max_samples=5000.
#
# Submit:    sbatch run_viz_geometry.sbatch
# Inspect:   ls $SCRATCH/projects/UR-JEPA/runs/viz_geometry/inet100
# Override:  sbatch --export=ALL,DATASET=galaxy10,CKPT_SIGREG=...,CKPT_URCGLT=...,OUT=... run_viz_geometry.sbatch
#
#SBATCH --job-name=viz_geometry
#SBATCH --account=cis260608-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda-envs/urjepa"

export HF_HOME="$SCRATCH/.cache/huggingface"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$SCRATCH/projects/UR-JEPA"

# --- defaults (override via --export=ALL,VAR=value) ----------------------
: "${DATASET:=imagenet100}"
: "${CKPT_SIGREG:=$SCRATCH/projects/UR-JEPA/runs/inet100_lejepa/sigreg_s0/ckpt-final.pt}"
: "${CKPT_URCGLT:=$SCRATCH/projects/UR-JEPA/runs/inet100_lejepa/ur_cglt_s0/ckpt-final.pt}"
: "${MAX_SAMPLES:=5000}"
: "${BS:=128}"
: "${IMG_SIZE:=128}"
: "${OUT:=$SCRATCH/projects/UR-JEPA/runs/viz_geometry/${DATASET}}"

echo "--- viz_geometry (dataset=$DATASET, max_samples=$MAX_SAMPLES, bs=$BS) ---"
echo "    sigreg ckpt: $CKPT_SIGREG"
echo "    urcglt ckpt: $CKPT_URCGLT"
echo "    out dir:     $OUT"
nvidia-smi || true

python scripts/viz_geometry.py \
  +ckpt_sigreg="$CKPT_SIGREG" \
  +ckpt_urcglt="$CKPT_URCGLT" \
  +dataset="$DATASET" \
  +max_samples="$MAX_SAMPLES" \
  +bs="$BS" \
  +img_size="$IMG_SIZE" \
  +out_dir="$OUT"
