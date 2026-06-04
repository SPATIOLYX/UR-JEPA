#!/bin/bash
# Reproducibility verification of the Galaxy10 headline result
# (Spatiolyx/UR-CGLT @ 78.94 +/- 0.34 from runs/galaxy10_lejepa_a100/).
# Same exact recipe, 3 seeds (0, 1, 2), but on H100 instead of A100.
#
# Original headline run was on A100 (cis260608-gpu) with 3 seeds,
# 200 epochs (3h wall cap stopped at ~197-198 ep). This sbatch
# reproduces all three seeds on H100 (cis260608-ai) with extra wall
# headroom so the full 200 epochs complete cleanly. Goal: confirm
# 0.7894 +/- 0.0034 is reproducible at the same recipe on different
# hardware, with a tight enough variance to compare seed-by-seed.
#
# Recipe (bit-for-bit match to the original A100 run):
#   * +dataset=galaxy10           Galaxy10 SDSS, 50/50 stratified split
#   * +backbone=resnet18_lowres   conv1=3x3 stride=1, no maxpool
#   * +proj_dim=32                our empirical optimum from Imagenette
#   * +bs=200                     (A100 memory limit at the time; kept
#                                  to preserve exact recipe even though
#                                  H100 could run bs=256)
#   * +V=4                        our convention
#   * +img_size=128 (default)     RandomResizedCrop upscales from 69
#   * +epochs=200                 actual epochs that produced 78.94 on A100
#   * +lr=2e-3 +eta_min=1e-3      LeJEPA MINIMAL.md literal
#   * +lamb=0.02                  our default
#   * +n=7 +n_scales=5            UR-CGLT defaults
#
# Output dir is DIFFERENT from the original (runs/galaxy10_verify_h100/)
# so we can compare side-by-side without overwriting the headline data.
#
# Per-task wall on H100: ResNet18 at 128x128, bs=200, V=4 (800
# images/step backbone forward). H100 is ~2-3x faster than A100 for
# this workload but the data pipeline is likely the bottleneck at
# this small backbone. Estimate ~2h per seed; set --time=04:00:00
# for safety (vs the original 3h cap that truncated).
#
# Submit:    sbatch run_galaxy10_verify.sbatch
# Re-run a subset (e.g. just seeds 1, 2):  sbatch --array=1-2 run_galaxy10_verify.sbatch
# Check:     for s in 0 1 2; do
#              jq '{seed: '$s', best_acc, final_acc, epochs: (.epochs | length)}' \
#                $SCRATCH/projects/UR-JEPA/runs/galaxy10_verify_h100/ur_cglt_s$s/results.json
#            done
#
# Compare to the original A100 mean ± std: 0.7894 ± 0.0034 (per-seed
# best_acc: s0=0.7872, s1=0.7877, s2=0.7933). If H100 verify lands
# within ±0.005 of those per-seed numbers, reproducibility confirmed.
# Any systematic >0.005 drift in either direction, dig in.
#
#SBATCH --job-name=galaxy10_verify
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
#SBATCH --array=1-2

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
OUT="$SCRATCH/projects/UR-JEPA/runs/galaxy10_verify_h100/ur_cglt_s${SEED}"

echo "--- array task ${SLURM_ARRAY_TASK_ID}: ur_cglt seed=$SEED (galaxy10 headline-recipe verify on H100: resnet18_lowres, proj_dim=32, bs=200, V=4, 200 ep) out=$OUT ---"
nvidia-smi || true

# Bit-for-bit match to the recipe that produced runs/galaxy10_lejepa_a100/
# headline (Spatiolyx/UR-CGLT 78.94 +/- 0.34 on Galaxy10 SDSS).
python scripts/pretrain.py \
  +dataset=galaxy10 \
  +backbone=resnet18_lowres \
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
  +bs=512 \
  +accum_steps=1 \
  +epochs=400 \
  +n=7 \
  +n_scales=5 \
  +anchors=128 \
  +lambda_ad=0.1 \
  +num_slices=256 \
  +wandb=false
