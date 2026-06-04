#!/bin/bash
# Geometry-aware visualization on EuroSAT RGB, proj_dim=32, all 3 seeds of
# both matched-recipe methods overlaid in a single panel:
#   SIGReg  s0,s1,s2   (LeJEPA baseline, isotropic Gaussian projector)
#   UR-CGLT s0,s1,s2   (geometric loss; anisotropic covariance cliff)
#
# In-domain SSL ckpts from runs/eurosat_a100 (resnet18_lowres, bs=256,
# 400 ep, img_size=128 upscaled from 64x64). Shows whether the SIGReg
# isotropy vs UR-CGLT cliff signature is seed-stable on a second
# non-AD-regular domain.
#
# Submit:    sbatch run_viz_geometry_eurosat_6way.sbatch
# Inspect:   ls $SCRATCH/projects/UR-JEPA/runs/viz_geometry/eurosat_6way
#
#SBATCH --job-name=viz_geom_eurosat_6way
#SBATCH --account=cis260608-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
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

RUNS="$SCRATCH/projects/UR-JEPA/runs"
: "${DATASET:=eurosat}"
: "${MAX_SAMPLES:=5000}"
: "${BS:=128}"
: "${IMG_SIZE:=128}"
: "${OUT:=$RUNS/viz_geometry/${DATASET}_6way}"

E="$RUNS/eurosat_a100"
for s in 0 1 2; do
  for m in sigreg ur_cglt; do
    f="$E/${m}_s${s}/ckpt-final.pt"
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
  done
done

echo "--- viz_geometry eurosat 6-way (max_samples=$MAX_SAMPLES, bs=$BS, img=$IMG_SIZE) ---"
echo "    out: $OUT"
nvidia-smi || true

python scripts/viz_geometry.py \
  +dataset="$DATASET" \
  +max_samples="$MAX_SAMPLES" \
  +bs="$BS" \
  +img_size="$IMG_SIZE" \
  +out_dir="$OUT" \
  +ckpts.sigreg_s0="$E/sigreg_s0/ckpt-final.pt" \
  +ckpts.sigreg_s1="$E/sigreg_s1/ckpt-final.pt" \
  +ckpts.sigreg_s2="$E/sigreg_s2/ckpt-final.pt" \
  +ckpts.ur_cglt_s0="$E/ur_cglt_s0/ckpt-final.pt" \
  +ckpts.ur_cglt_s1="$E/ur_cglt_s1/ckpt-final.pt" \
  +ckpts.ur_cglt_s2="$E/ur_cglt_s2/ckpt-final.pt" \
  '+labels.sigreg_s0="LeJEPA($\mathcal{L}^{\mathrm{SIGReg}}$) s0"' \
  '+labels.sigreg_s1="LeJEPA($\mathcal{L}^{\mathrm{SIGReg}}$) s1"' \
  '+labels.sigreg_s2="LeJEPA($\mathcal{L}^{\mathrm{SIGReg}}$) s2"' \
  '+labels.ur_cglt_s0="UR-JEPA($\mathcal{L}^{\mathrm{CGLT}}$) s0"' \
  '+labels.ur_cglt_s1="UR-JEPA($\mathcal{L}^{\mathrm{CGLT}}$) s1"' \
  '+labels.ur_cglt_s2="UR-JEPA($\mathcal{L}^{\mathrm{CGLT}}$) s2"'
