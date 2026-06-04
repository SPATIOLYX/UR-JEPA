# SLURM recipes

These eleven sbatches reproduce every result table and figure in the paper.
They are the recipes used in our submission, written for the [Anvil cluster](https://www.rcac.purdue.edu/anvil)
at Purdue (NSF ACCESS allocation), but adapt to any SLURM site with a one-block edit.

## Quick map: sbatch → paper table

| Result | sbatch |
|---|---|
| **Table 1** Inet10 UR&ndash;JEPA(L<sup>CGLT</sup>) headline (D=32, n=7, K=5; the K=5 array tasks give the +0.83&nbsp;pp cell) | `run_K_sweep_projdim32_lejepa.sbatch` |
| **Table 1** Inet10 LeJEPA(L<sup>SIGReg</sup>) matched-D=32 baseline | `run_sigreg_projdim32_lejepa.sbatch` |
| **Table 5** Galaxy10 UR&ndash;JEPA(L<sup>CGLT</sup>) headline (0.8142 &plusmn; 0.0017) | `run_galaxy10_verify.sbatch` |
| **Table 5** Galaxy10 LeJEPA(L<sup>SIGReg</sup>) @ D=32 | `run_galaxy10_sigreg.sbatch` |
| **Table 5** Galaxy10 LeJEPA(L<sup>SIGReg</sup>) @ D=16 (LeJEPA-canonical) | `run_galaxy10_lejepa.sbatch` |
| **Table 5** Galaxy10 UR&ndash;JEPA(L<sup>CGLT,&part;log</sup>) | `run_galaxy10_cglt_deriv_verify.sbatch` |
| **Table 5** Galaxy10 UR&ndash;JEPA(L<sup>CGLT,&part;</sup>) raw form | `run_galaxy10_cglt_deriv_raw_verify.sbatch` |
| **Table 5** Galaxy10 UR&ndash;JEPA(L<sup>&beta;,&gamma;</sup>) &beta;&nbsp;+&nbsp;log-trace | `run_galaxy10_beta_logtrace_a100.sbatch` |
| **Table 5** Galaxy10 UR&ndash;JEPA(L<sup>&beta;,&gamma;,&tau;</sup>) &tau;=1.0 | `run_galaxy10_beta_eigthresh_pd32.sbatch` |
| **Table 7** Inet100 Stage-2 single-seed convergence trajectory | `run_inet100_lejepa.sbatch` |
| **Table 8** EuroSAT~RGB matched-recipe head-to-head (3 seeds) | `run_eurosat_a100.sbatch` |
| **Figures 4&ndash;6** Projector-geometry diagnostics (Inet10, Galaxy10, Inet100) | `run_viz_geometry.sbatch` |
| **Figure 7 + Table 10** EuroSAT 3-seed projector-geometry overlay | `run_viz_geometry_eurosat_6way.sbatch` |

## Adapting to your cluster

Every sbatch starts with a block like:

```bash
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
```

Three things you will likely need to change:

1. **`--account` and `--partition`**: replace with your cluster's values (often something like `--account=<your-allocation> --partition=<your-gpu-queue>`). The `cis260608-ai` value is our Anvil allocation; the `cis260608-gpu` variant (in the A100 recipes) targets Anvil's A100 partition.
2. **`SCRATCH` path**: each sbatch does `cd "$SCRATCH/projects/UR-JEPA"` and writes outputs under `$SCRATCH/projects/UR-JEPA/runs/`. Either export `SCRATCH=<your-fast-scratch-dir>` before submitting, or replace the path with your absolute project directory.
3. **`conda activate "$SCRATCH/conda-envs/urjepa"`**: replace with your conda env path (or substitute your preferred environment manager, e.g., `source <venv>/bin/activate`).

Everything else (recipe flags, array task counts, wall-time estimates documented in the header comments) is portable.

## Non-cluster runs

Every sbatch is just a shell wrapper around a single `python scripts/pretrain.py ...` invocation (or `python scripts/viz_geometry.py ...` for the viz one). To run locally, copy the `python` line from inside the sbatch and execute it directly with the array-task-specific values substituted in.

For example, the Inet10 UR&ndash;JEPA(L<sup>CGLT</sup>) headline cell (D=32, n=7, K=5, single seed) reduces to:

```bash
python scripts/pretrain.py \
  +dataset=imagenette \
  +regularizer=ur_cglt \
  +reg_scale=1000 \
  +proj_dim=32 \
  +n=7 \
  +n_scales=5 \
  +seed=0 \
  +out_dir=runs/inet10/ur_cglt_n7_K5_s0 \
  +epochs=800 \
  +lr=2e-3 +eta_min=1e-3 +bs=256 +V=4 +lamb=0.02 \
  +projector_norm=bn +anchors=128 +lambda_ad=0.1 +num_slices=256
```

(The full set of flags actually used in each headline run is at the bottom of each sbatch.)

## Compute budget summary

Recipes are sized for the budget we had on Anvil; values below are the upper bound observed in practice. Adjust `--time` and `--mem` per your cluster's policy.

| sbatch | per-task wall | per-task GPU | array size | total GPU-h |
|---|---|---|---|---|
| `run_K_sweep_projdim32_lejepa.sbatch` | ~4h | 1× H100 | 24 | ~96 |
| `run_sigreg_projdim32_lejepa.sbatch` | ~4h | 1× H100 | 3 | ~12 |
| `run_galaxy10_verify.sbatch` | ~3h | 1× H100 | 3 | ~9 |
| `run_galaxy10_sigreg.sbatch` | ~3h | 1× H100 | 3 | ~9 |
| `run_galaxy10_lejepa.sbatch` | ~3h | 1× H100 | 3 | ~9 |
| `run_galaxy10_cglt_deriv_verify.sbatch` | ~3.4h | 1× H100 | 3 | ~10 |
| `run_galaxy10_cglt_deriv_raw_verify.sbatch` | ~3.4h | 1× H100 | 3 | ~10 |
| `run_galaxy10_beta_logtrace_a100.sbatch` | ~5h | 2× A100 DDP | 3 | ~30 (A100-h) |
| `run_galaxy10_beta_eigthresh_pd32.sbatch` | ~3.4h | 1× H100 | 3 | ~10 |
| `run_inet100_lejepa.sbatch` | ~17h (400 ep) | 1× H100 | 3 | ~51 |
| `run_eurosat_a100.sbatch` | ~3.4h | 1× A100-40GB | 6 | ~20 (A100-h) |
| `run_viz_geometry.sbatch` | <1h | 1× H100 | 1 | ~1 |
| `run_viz_geometry_eurosat_6way.sbatch` | <1h | 1× H100 | 1 | ~1 |

Approximate total: ~250 H100-h + ~30 A100-h to reproduce all headline tables and figures from scratch at 3 seeds.

## Pre-cache datasets first

Most pretraining sbatches will lazily download the dataset on first run. To avoid having a dozen tasks downloading the same dataset concurrently, pre-cache once on a login node:

```bash
bash scripts/precache_datasets.sh          # Imagenette, ImageNet-100, CIFAR, Flowers, Food
python scripts/galaxy10_sdss.py            # Galaxy10 SDSS (HDF5 from Zenodo, ~200 MB)
bash scripts/precache_downstream.sh        # downstream-transfer datasets (Aircraft, DTD, etc.)
```
