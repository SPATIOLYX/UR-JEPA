#!/bin/bash
# β-number + log-trace anti-collapse (Option B) on Galaxy10 SDSS,
# 2× A100 DDP — H100-equivalent run at the same effective batch (512)
# but using the abundant A100 budget instead of the scarce H100 budget.
#
# Loss:
#     L = β²(x, r)  -  γ · mean_a log trace(S_r(x))
#
# Why this exists (and supersedes the earlier single-GPU bs=200 A100
# variant whose 3-seed result lives at runs/galaxy10_beta_logtrace_a100/):
#   * The earlier single-A100 bs=200 run got 0.7970 ± 0.0037 on Galaxy10
#     — competitive at A100 bs=200, but NOT matched-recipe to the new
#     H100-extended headline (UR-CGLT integral 0.8142 ± 0.0017 at H100
#     bs=512, 400 ep). The recipe asymmetry confounds the β-vs-CGLT
#     comparison: β was bs=200, CGLT was bs=512.
#   * To close that asymmetry without burning H100 SU, we run 2× A100
#     DDP with bs=256 per rank → effective bs=512, identical to the
#     H100 single-GPU recipe.
#   * scripts/pretrain.py handles DDP fully:
#       (1) all_gather_with_grad() reassembles the projector outputs
#           across ranks before the regularizer evaluates — so SIGReg
#           / CGLT / β² see the true effective bs=512, bit-identical
#           to a single-GPU bs=512 run (rel diff = 0.00e+00 verified).
#       (2) SyncBatchNorm.convert_sync_batchnorm() under DDP keeps
#           the projector's BN consistent with the H100 single-GPU
#           BatchNorm1d behavior.
#       (3) Per-rank invariance + probe losses; DDP averages grads.
#
# Recipe (matched to run_galaxy10_beta_logtrace.sbatch H100 except for
# the 2-GPU DDP launcher and bs split):
#   * +backbone=resnet18_lowres
#   * +regularizer=ur_beta  +reg_scale=1000
#   * +lambda_ad=0  +gamma_logtrace=0.1   (β² + log-trace anti-collapse)
#   * +bs=256 per rank × 2 ranks = effective bs=512
#   * +V=4  +proj_dim=32  +img_size=128 (default)
#   * +epochs=400  +lr=2e-3  +eta_min=1e-3  +lamb=0.02
#   * +n=7  +n_scales=5  +anchors=128  +num_slices=256
#
# Per-seed wall on 2× A100 DDP: ~4-5h (single H100 takes ~3.3h at the
# same effective batch; A100 is ~0.5× H100 fp16 throughput per GPU,
# 2 A100s in parallel ≈ 1.0× H100 raw, with ~20% DDP all-reduce
# overhead → ~4h with some headroom). Setting --time=06:00:00 for
# safety; the single-GPU bs=200 A100 variant took ~6h per seed at
# 400 ep, this should be faster or equal.
#
# Per-seed SU on 2× A100: ~4h × 2 GPUs = ~8 A100 SU. Total 3-seed cost:
# ~24 A100 SU (vs ~10 H100 SU for the strict-H100 alternative).
# Plenty of headroom in the 1647-SU A100 budget.
#
# Output dir distinct from the bs=200 single-GPU variant
# (runs/galaxy10_beta_logtrace_a100/) to avoid overwriting that data.
#
# Submit:    sbatch run_galaxy10_beta_logtrace_a100.sbatch
# Re-run a subset (e.g. just seeds 1, 2):  sbatch --array=1-2 run_galaxy10_beta_logtrace_a100.sbatch
# Check:     for s in 0 1 2; do
#              jq '{seed: '$s', best_acc, final_acc, epochs: (.epochs | length)}' \
#                $SCRATCH/projects/UR-JEPA/runs/galaxy10_beta_logtrace_2a100/ur_beta_s$s/results.json
#            done
#
# Expected outcome vs the H100-recipe variants:
#   * If ≥ 0.81  -> β + log-trace at effective bs=512 matches or beats
#                   UR-CGLT integral (0.8142). The +0.76pp β advantage
#                   from the A100 bs=200 comparison generalizes to
#                   the extended H100-equivalent recipe.
#   * 0.80-0.81  -> β + log-trace lands between SIGReg matched
#                   (0.8105) and the deriv variants; competitive
#                   at matched recipe.
#   * < 0.80     -> β + log-trace underperforms at the extended
#                   recipe; the A100 bs=200 +0.76pp win was recipe-
#                   dependent and doesn't transfer.
#
#SBATCH --job-name=galaxy10_beta_logtrace_2a100
#SBATCH --account=cis260608-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
# Random-ish port to avoid clashes if two array tasks land on the same node.
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID + RANDOM % 1000))

cd "$SCRATCH/projects/UR-JEPA"

SEED="${SLURM_ARRAY_TASK_ID}"
OUT="$SCRATCH/projects/UR-JEPA/runs/galaxy10_beta_logtrace_2a100/ur_beta_s${SEED}"

echo "--- array task ${SLURM_ARRAY_TASK_ID}: ur_beta seed=$SEED (galaxy10 beta+log-trace on 2x A100 DDP: resnet18_lowres, proj_dim=32, bs=256/rank x 2 = eff 512, V=4, 400 ep, gamma_logtrace=0.1, lambda_ad=0) out=$OUT ---"
nvidia-smi || true

# 2× A100 DDP at bs=256/rank — effective batch 512, matching the H100
# single-GPU recipe (run_galaxy10_beta_logtrace.sbatch).
torchrun --standalone --nproc-per-node=2 scripts/pretrain.py \
  +dataset=galaxy10 \
  +backbone=resnet18_lowres \
  +regularizer=ur_beta \
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
  +epochs=400 \
  +n=7 \
  +n_scales=5 \
  +anchors=128 \
  +lambda_ad=0 \
  +gamma_logtrace=0.1 \
  +num_slices=256 \
  +wandb=false
