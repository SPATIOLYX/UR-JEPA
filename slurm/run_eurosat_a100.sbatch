#!/bin/bash
# EuroSAT RGB in-domain pretraining: SIGReg vs UR-CGLT head-to-head.
#
# Second non-natural domain (Earth-observation remote sensing) alongside
# Galaxy10 SDSS (astronomy), to test whether the "in-domain SSL beats
# foundation-model transfer" finding generalizes beyond a single domain.
# Recipe is bit-for-bit the Galaxy10 matched-recipe headline
# (see slurm/run_galaxy10_verify.sbatch): resnet18_lowres,
# proj_dim=32, bs=512, V=4, 400 ep, lr=2e-3, eta_min=1e-3, lamb=0.02,
# drop_path=0.1, ur_cglt n=7 K=5 lambda_ad=0.1. Only the dataset changes,
# so the EuroSAT result is directly comparable to Galaxy10 cell-for-cell.
#
#   * +dataset=eurosat            timm/eurosat-rgb, 16,200 train / 5,400 val
#   * +backbone=resnet18_lowres   conv1=3x3 stride=1, no maxpool (64px tiles)
#   * +proj_dim=32                our empirical optimum
#   * +img_size=128 (default)     RandomResizedCrop upscales from 64
#   * +bs=256                     A100-40GB-safe; see MEMORY NOTE
#
# MEMORY NOTE: bs=512 (the Galaxy10 H100 headline batch) OOMs on Anvil's
# A100-40GB nodes -- job 17814276 died in resnet18 layer1 BN, 8 GiB short.
# The resnet18_lowres mod (stride-1 conv1, no maxpool) keeps full 128x128
# spatial resolution through layer1, so activations are large despite the
# small param count; the original A100 Galaxy10 headline used bs=200 for
# the same reason. We use bs=256 here (fits ~17/40 GiB). PARITY CAVEAT:
# this is NOT bs-matched to the bs=512 Galaxy10 *H100* headline (0.8142);
# it IS comparable to the bs=200-class A100 Galaxy10 cell (0.7894). For an
# exact bs=512 cross-domain match, run on the H100 (ai) partition instead.
#
# 6 tasks = 2 methods (sigreg, ur_cglt integral) x 3 seeds (0, 1, 2),
# method-major: 0-2 = sigreg s0-2, 3-5 = ur_cglt s0-2.
#
# Submit:    sbatch run_eurosat_a100.sbatch
# Subset:    sbatch --array=3-5 run_eurosat_a100.sbatch   # just ur_cglt
# Check:     for d in runs/eurosat_a100/*/; do
#              jq '{run: "'$d'", best_acc, final_acc, epochs: (.epochs|length)}' "$d/results.json"
#            done
# Aggregate: python scripts/aggregate_results.py $SCRATCH/projects/UR-JEPA/runs/eurosat_a100
#
#SBATCH --job-name=eurosat_a100
#SBATCH --account=cis260608-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

# --- task table (6 = 2 methods x 3 seeds, method-major) --------------
METHODS=(sigreg sigreg sigreg ur_cglt ur_cglt ur_cglt)
SEEDS=(0 1 2 0 1 2)
REG="${METHODS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
# SIGReg uses its native reg_scale=1; UR-CGLT uses the canonical 1000.
if [ "$REG" = "ur_cglt" ]; then REG_SCALE=1000; else REG_SCALE=1; fi
OUT="$SCRATCH/projects/UR-JEPA/runs/eurosat_a100/${REG}_s${SEED}"

echo "--- array task $SLURM_ARRAY_TASK_ID: $REG seed=$SEED (eurosat in-domain: resnet18_lowres, proj_dim=32, bs=256, V=4, 400 ep, reg_scale=$REG_SCALE) out=$OUT ---"
nvidia-smi || true

python scripts/pretrain.py \
  +dataset=eurosat \
  +backbone=resnet18_lowres \
  +regularizer="$REG" \
  +reg_scale="$REG_SCALE" \
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
  +wandb=false
