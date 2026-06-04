#!/bin/bash
# UR-CGLT literal Eq.~(1.5) scale-derivative variant (raw form, no log
# transformation) verify on Galaxy10. Companion to
# run_galaxy10_cglt_deriv_verify.sbatch (which uses the LOG derivative
# t * d/dt log theta_t = <d^2/t^2 - n>_w).
#
# The raw form is the literal CGLT 2014 Theorem 1.2 statement:
#   tilde Delta = (1/(N t^n)) sum_j exp(-d_j^2/(2t^2)) * (d_j^2/t^2 - n)
#                = theta_t(x) * <d^2/t^2 - n>_w
# The theta_t(x) prefactor couples the loss to the local density level,
# which is exactly the failure mode the log variant was engineered to
# avoid. Pair with the AD anchor (lambda_ad=0.1) so the cloud is pushed
# toward AD-regularity --- the precondition Theorem 1.2 needs.
#
# The CGLTDerivRaw class uses the t_max-normalized prefactor
# 1 / (N (t/t_max)^n) instead of the literal Eq.~(1.5) prefactor
# 1 / (N t^n). This keeps the loss magnitude in the same O(10)
# range as the log variants (smoke test: ~27 on isotropic Gaussian),
# while still blowing up at point-mass collapse (~5e7 in the same
# smoke test). reg_scale=1000 like the other variants; no special
# retune needed.
#
# Decision tree vs the log-derivative verify
# (run_galaxy10_cglt_deriv_verify.sbatch):
#   * matches log-deriv within seed noise -> two forms equivalent in
#                  practice; recommend log form as default (smaller
#                  reg_scale, batch-size invariant magnitude).
#   * raw beats log-deriv -> the local-density coupling is actually
#                  informative, not parasitic. Reconsider the §5.1
#                  "Why the log-increment" argument.
#   * raw underperforms -> log form vindicated for training; raw is
#                  theoretically literal but practically inferior.
#
# 3-seed array (0, 1, 2) for mean ± std parallel to the dyadic-diff
# verify's 0.8142 ± 0.0017 headline. Per-task wall on H100: ~3.4h.
# 8h wall headroom.
#
# Submit:    sbatch run_galaxy10_cglt_deriv_raw_verify.sbatch
# Re-run a subset (e.g. just seeds 1, 2):  sbatch --array=1-2 run_galaxy10_cglt_deriv_raw_verify.sbatch
# Check:     for s in 0 1 2; do
#              jq '{seed: '$s', best_acc, final_acc, epochs: (.epochs | length)}' \
#                $SCRATCH/projects/UR-JEPA/runs/galaxy10_cglt_deriv_raw_verify/ur_cglt_deriv_raw_s$s/results.json
#            done
#
#SBATCH --job-name=galaxy10_cglt_deriv_raw_verify
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
OUT="$SCRATCH/projects/UR-JEPA/runs/galaxy10_cglt_deriv_raw_verify/ur_cglt_deriv_raw_s${SEED}"

echo "--- array task ${SLURM_ARRAY_TASK_ID}: ur_cglt_deriv_raw seed=$SEED (galaxy10 rescaled Eq.(1.5) raw form on H100: resnet18_lowres, proj_dim=32, bs=512, V=4, 400 ep, reg_scale=1000, lambda_ad=0.1) out=$OUT ---"
nvidia-smi || true

# Same recipe as run_galaxy10_cglt_deriv_verify.sbatch except:
#   * +regularizer=ur_cglt_deriv_raw  (was ur_cglt_deriv)
python scripts/pretrain.py \
  +dataset=galaxy10 \
  +backbone=resnet18_lowres \
  +regularizer=ur_cglt_deriv_raw \
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
