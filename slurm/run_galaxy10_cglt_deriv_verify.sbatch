#!/bin/bash
# UR-CGLT scale-derivative variant (Eq. (1.5) of CGLT 2014) verify on
# Galaxy10. Same recipe as run_galaxy10_verify.sbatch (the dyadic-
# difference variant verify that hit best=0.8124 / final=0.7981 at
# 400 ep, 1 seed) except:
#   * +regularizer=ur_cglt_deriv  (was +regularizer=ur_cglt)
#
# Hypothesis being tested: the scale-derivative characterization of UR
# in Theorem 1.2 of Chousionis-Garnett-Le-Tolsa is provably equivalent
# to the dyadic-difference characterization of Theorem 1.1, but offers
# two practical advantages that may translate to better training:
#   1. One per-anchor quantity per scale (vs. one per pair of adjacent
#      scales), so the gradient signal is not cancelled by cross-scale
#      subtractions.
#   2. Half the memory footprint per anchor (one log density per
#      scale instead of two).
#
# Decision tree on the result vs the 400-ep dyadic-difference verify
# (best=0.8124, final=0.7981, 1 seed):
#   * >= 0.81  -> derivative variant matches or exceeds dyadic-diff.
#                 Both characterizations are training-equivalent in
#                 our regime; recommend deriv as default for its
#                 memory advantage.
#   * 0.78-0.80 -> derivative variant within seed noise of dyadic-diff.
#                  No clear winner; report both.
#   * < 0.78    -> derivative variant underperforms. Discrete approx
#                  has a property we're not capturing in the
#                  continuous limit (perhaps it benefits from the
#                  cross-scale gradient cancellation as a stabilizer?).
#
# 3-seed array (0, 1, 2) for mean ± std parallel to the dyadic-diff
# verify's 0.8142 ± 0.0017 headline. Per-task wall on H100: ~3.4h
# (same as the dyadic-diff verify; only the regularizer kernel
# differs, backbone forward dominates). 8h wall headroom.
#
# Submit:    sbatch run_galaxy10_cglt_deriv_verify.sbatch
# Re-run a subset (e.g. just seeds 1, 2):  sbatch --array=1-2 run_galaxy10_cglt_deriv_verify.sbatch
# Check:     for s in 0 1 2; do
#              jq '{seed: '$s', best_acc, final_acc, epochs: (.epochs | length)}' \
#                $SCRATCH/projects/UR-JEPA/runs/galaxy10_cglt_deriv_verify/ur_cglt_deriv_s$s/results.json
#            done
#
#SBATCH --job-name=galaxy10_cglt_deriv_verify
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
OUT="$SCRATCH/projects/UR-JEPA/runs/galaxy10_cglt_deriv_verify/ur_cglt_deriv_s${SEED}"

echo "--- array task ${SLURM_ARRAY_TASK_ID}: ur_cglt_deriv seed=$SEED (galaxy10 Eq.(1.5) variant on H100: resnet18_lowres, proj_dim=32, bs=512, V=4, 400 ep, lambda_ad=0.1) out=$OUT ---"
nvidia-smi || true

# Same recipe as run_galaxy10_verify.sbatch except +regularizer=ur_cglt_deriv.
python scripts/pretrain.py \
  +dataset=galaxy10 \
  +backbone=resnet18_lowres \
  +regularizer=ur_cglt_deriv \
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
