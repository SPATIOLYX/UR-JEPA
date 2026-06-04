#!/bin/bash
# SIGReg at proj_dim=${PD} on Galaxy10 SDSS — the "LeJEPA-loss at
# LeJEPA's canonical projector dim" experiment. Mirrors
# run_galaxy10_sigreg.sbatch in every knob except +proj_dim=${PD}
# (vs sigreg's 32). Set PD on the variable line below; output paths
# and the python +proj_dim flag both derive from it.
#
# Motivation:
#   The matched-recipe SIGReg run (runs/galaxy10_sigreg_h100/,
#   0.8105 ± 0.0050 at D=32) gave us a SIGReg vs UR-CGLT comparison
#   at OUR optimum projector dim (D=32, where UR-CGLT peaks). But
#   on Imagenette-10, SIGReg's optimum is D=16, NOT D=32:
#
#     SIGReg @ D=16 (runs/lejepa_exact/sigreg_*):    0.9130 ± 0.0008
#     SIGReg @ D=32 (runs/sigreg_projdim32_lejepa):  0.9058 ± 0.0019
#     Δ:                                          −0.72 pp (degrades)
#
#   So the matched-recipe Galaxy10 comparison at D=32 implicitly
#   favors UR-CGLT (its optimum) and disfavors SIGReg (above its
#   optimum). To get the honest each-at-its-own-optimum-D
#   comparison, we need SIGReg @ D=16 on Galaxy10.
#
#   Mechanism: SIGReg's
#   num_slices=256 covers the S^(D-1) Gaussian-direction sphere more
#   sparsely as D grows; D=32 doubles the sphere's volume but keeps
#   the slice count the same, so coverage drops per-dimension.
#
# Two competing predictions:
#   Prediction A (Imagenette extrapolates): SIGReg lands ~0.818 on
#     Galaxy10 D=16 (0.8105 + 0.007 ≈ 0.818). At or above UR-CGLT's
#     D=32 headline (0.8142). The regularizer-attributable advantage
#     on Galaxy10 collapses; the honest comparison would be each
#     method at its own optimum D.
#   Prediction B (Galaxy10 is different): SIGReg flat or lower at
#     D=16 on Galaxy10 (small dataset, non-natural domain). The D=32
#     comparison is fine.
#
# Decision tree:
#   * >= 0.815  -> Prediction A holds; UR-CGLT-vs-SIGReg gap on
#                  Galaxy10 disappears at each-method-its-own-D.
#                  Honest paper framing has to acknowledge this.
#   * 0.805-0.814 -> SIGReg flat or marginally better at D=16; UR-CGLT
#                    D=32 still trends above but within seed noise.
#   * < 0.805   -> Prediction B holds; D=32 was already SIGReg's best
#                  on Galaxy10; UR-CGLT advantage at D=32 stands.
#
# Trust calibration:
#   LeJEPA's published Galaxy10 result (resnet18, 0.7532) is BELOW
#   our SIGReg at D=32 + our recipe (0.8105) — confirming our
#   recipe is genuinely stronger than LeJEPA's published recipe
#   (independent of regularizer family). We trust their published
#   numbers; the recipe gain is real.
#
# Recipe differences vs run_galaxy10_sigreg.sbatch (D=32):
#   * +proj_dim=${PD} (vs +proj_dim=32) — ONLY difference
#   * Output dir runs/galaxy10_lejepa_pd${PD}/sigreg_s{SEED}
#   * Job name galaxy10_lejepa
#
# All other knobs identical: backbone=resnet18_lowres, bs=512, V=4,
# 400 ep, lr=2e-3, eta_min=1e-3, lamb=0.02, num_slices=256, knots=17,
# regularizer=sigreg, reg_scale=1, projector_norm=bn.
#
# Per-seed wall on H100: ~3.3h (D=16 projector is even smaller than
# D=32, so wall ≤ sigreg sbatch's 3.3h). 8h wall headroom.
#
# Submit:    sbatch run_galaxy10_lejepa.sbatch
# Re-run a subset (e.g. just seeds 1, 2):  sbatch --array=1-2 run_galaxy10_lejepa.sbatch
# Check:     PD=16; for s in 0 1 2; do
#              jq '{seed: '$s', best_acc, final_acc, epochs: (.epochs | length)}' \
#                $SCRATCH/projects/UR-JEPA/runs/galaxy10_lejepa_pd${PD}/sigreg_s$s/results.json
#            done
#
# Cost: 3 seeds × ~3.3h ≈ 10 H100 SU.
#
#SBATCH --job-name=galaxy10_lejepa
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
PD=16
OUT="$SCRATCH/projects/UR-JEPA/runs/galaxy10_lejepa_pd${PD}/sigreg_s${SEED}"

echo "--- array task ${SLURM_ARRAY_TASK_ID}: sigreg seed=$SEED (galaxy10 SIGReg at LeJEPA-canonical D=${PD} on H100: resnet18_lowres, proj_dim=${PD}, bs=512, V=4, 400 ep) out=$OUT ---"
nvidia-smi || true

# Same as run_galaxy10_sigreg.sbatch except +proj_dim=${PD} (vs 32).
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
  +proj_dim=${PD} \
  +lr=2e-3 \
  +eta_min=1e-3 \
  +bs=512 \
  +accum_steps=1 \
  +epochs=400 \
  +n=8 \
  +num_slices=256 \
  +knots=17 \
  +wandb=false
