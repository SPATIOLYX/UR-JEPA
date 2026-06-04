#!/bin/bash
# Matched-recipe SIGReg baseline on Galaxy10 SDSS, mirroring
# run_galaxy10_verify.sbatch exactly except for the regularizer
# (sigreg instead of ur_cglt) and the corresponding reg_scale (native
# semantics for sigreg = 1, vs 1000 for ur_cglt).
#
# Why this exists:
#   The current Galaxy10 headline (UR-CGLT 0.8142 ± 0.0017 from
#   runs/galaxy10_verify_h100/) compares against LeJEPA's published
#   Table 5 numbers, which use a different recipe (proj_dim=512, V=8,
#   img_size=224, lr=3e-3, lamb=0.05, 1000 slices). Reviewers will
#   want a matched-recipe SIGReg vs UR-CGLT comparison on this
#   dataset. This sbatch produces that baseline: SIGReg at exactly
#   the same recipe that produced the 0.8142 headline.
#
# Recipe (bit-for-bit match to run_galaxy10_verify.sbatch except
# regularizer + reg_scale):
#   * +dataset=galaxy10           Galaxy10 SDSS, 50/50 stratified split
#   * +backbone=resnet18_lowres   conv1=3x3 stride=1, no maxpool
#   * +regularizer=sigreg         (vs ur_cglt in verify)
#   * +reg_scale=1                native sigreg semantics
#                                 (vs 1000 for ur_cglt's small magnitude)
#   * +proj_dim=32                our empirical optimum from Imagenette
#   * +bs=512                     same as verify (H100 80GB)
#   * +V=4                        our convention
#   * +img_size=128 (default)     RandomResizedCrop upscales from 69
#   * +epochs=400                 same as verify
#   * +lr=2e-3 +eta_min=1e-3      LeJEPA MINIMAL.md literal
#   * +lamb=0.02                  our default
#   * +num_slices=256 +knots=17   LeJEPA MINIMAL.md sigreg defaults
#
# Output dir is runs/galaxy10_sigreg_h100/ to parallel the verify's
# runs/galaxy10_verify_h100/ naming convention; per-seed subdirs are
# sigreg_s{SEED}/ to parallel verify's ur_cglt_s{SEED}/.
#
# Per-seed wall on H100 (estimate from verify): ~3.3h. SIGReg's
# per-step cost differs from ur_cglt's only in the regularizer
# kernel (slicing + characteristic-function on a sample subset vs
# CGLT integration); backbone forward dominates, so wall should be
# essentially identical. +1.5h headroom -> --time=08:00:00.
#
# Submit:    sbatch run_galaxy10_sigreg.sbatch
# Re-run a subset (e.g. just seeds 1, 2):  sbatch --array=1-2 run_galaxy10_sigreg.sbatch
# Check:     for s in 0 1 2; do
#              jq '{seed: '$s', best_acc, final_acc, epochs: (.epochs | length)}' \
#                $SCRATCH/projects/UR-JEPA/runs/galaxy10_sigreg_h100/sigreg_s$s/results.json
#            done
#
# Expected outcome:
#   * If SIGReg also reaches ~0.81: UR-CGLT's headline advantage
#     was the recipe (bs=512 + 400 ep), not the regularizer.
#     Honest finding; still publishable as "this recipe is strong".
#   * If SIGReg lands at ~0.78-0.80: UR-CGLT wins on Galaxy10 by
#     1-3pp at matched recipe. Strong paired-comparison claim,
#     complements the +5.48pp ImageNet-100 screen finding.
#   * If SIGReg lands below 0.78: UR-CGLT advantage on Galaxy10
#     is larger than the +3.62pp vs LeJEPA published numbers
#     suggested. Strongest possible outcome for the paper.
#
#SBATCH --job-name=galaxy10_sigreg
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

SEED="${SLURM_ARRAY_TASK_ID}"
OUT="$SCRATCH/projects/UR-JEPA/runs/galaxy10_sigreg_h100/sigreg_s${SEED}"

echo "--- array task ${SLURM_ARRAY_TASK_ID}: sigreg seed=$SEED (galaxy10 matched-recipe sigreg baseline on H100: resnet18_lowres, proj_dim=32, bs=512, V=4, 400 ep) out=$OUT ---"
nvidia-smi || true

# Matched-recipe SIGReg counterpart to run_galaxy10_verify.sbatch.
python scripts/pretrain.py \
  +dataset=galaxy10 \
  +backbone=resnet18_lowres \
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
  +bs=512 \
  +accum_steps=1 \
  +epochs=400 \
  +n=8 \
  +num_slices=256 \
  +knots=17 \
  +wandb=false
