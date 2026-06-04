# UR&ndash;JEPA

**Uniform Rectifiability as a Regularizer for Joint-Embedding Predictive Architectures**

A geometric-measure-theoretic alternative to LeJEPA's isotropic-Gaussian target for JEPA embeddings. UR&ndash;JEPA replaces the full-dimensional Gaussian target with a *uniformly n-rectifiable* measure: the canonical geometric-measure-theory notion of "quantitatively n-dimensional at every location and scale." We operationalize this through a Carleson-type square-function loss built on prior work: Chousionis&ndash;Garnett&ndash;Le&ndash;Tolsa, *Square functions and uniform rectifiability* (TAMS, 2016).

📄 **Paper**: [arXiv:2606.01443](https://arxiv.org/abs/2606.01443) &nbsp; · &nbsp; 🌐 **Project page**: https://spatiolyx.ai/research/ur-jepa &nbsp; · &nbsp; 🏛️ **License**: Apache-2.0

---

## Headline results

- **+18 pp over IJEPA&ndash;IN22K foundation-model transfer on Galaxy10 SDSS** (81.4% vs 62.9% linear-probe). In-domain UR&ndash;JEPA on a 21K-image astronomical dataset substantially exceeds a 630M-parameter foundation model pretrained on 22M images.
- **+0.83 pp over matched-recipe LeJEPA on ImageNet-10** (3 seeds, paired-*t* = +15.5, *p* &lt;&lt; 0.001) with ~30% lower seed variance.
- **Statistically tied with LeJEPA at convergence on Galaxy10 SDSS, ImageNet-100, and EuroSAT (remote sensing).** Sample efficiency and lower seed variance are the practical differentiators at convergence; UR&ndash;JEPA holds $\sim\!2\times$ tighter seed variance on EuroSAT at 3 seeds.

- **EuroSAT vs foundation-model transfer.** In-domain UR&ndash;JEPA and LeJEPA at $11$M parameters reach $96.0$ to $96.1\%$ on EuroSAT RGB, competitive with large remote-sensing foundation models (matching Scale-MAE ViT-L at a $25\times$ smaller backbone).
- **Downstream transfer (ImageNet-100 &rarr; 5 datasets).** UR&ndash;JEPA leads LeJEPA on 4 of 5 transfer datasets (Aircraft, DTD, Flowers, Food) at the 800-epoch checkpoint, with a mean &Delta; of +0.32 pp (single seed).
- **Geometrically distinct representation.** UR&ndash;JEPA produces an effectively low-rank covariance structure where LeJEPA yields near-isotropic projections. The covariance cliff sits at index ~20&ndash;25 across all four datasets (single-seed on Inet10/Galaxy10/Inet100; three-seed seed-stable confirmation on EuroSAT), consistent with Pope et al. (ICLR 2021) intrinsic-dimension estimates for natural images.

---

## Install

```bash
git clone https://github.com/SPATIOLYX/UR-JEPA.git
cd UR-JEPA
pip install -r requirements.txt
pip install -e .
```

Requires Python &ge; 3.10 and a recent PyTorch build (tested on PyTorch 2.x with CUDA 12).

---

## Quick start

```python
import torch
from ur_jepa import URJEPA

# 1-line drop-in replacement for SIGReg in a LeJEPA-style training loop.
ur = URJEPA(n=7, variant="cglt", n_scales=5, lambda_ad=0.1)

# Z: (V, N, D) projector outputs from V augmented views, N items, D channels.
Z = torch.randn(4, 256, 32, requires_grad=True)
loss = ur(Z)
loss.backward()
```

The `URJEPA` module accepts four `variant` strings. The β family expands further via kwargs into three rehabilitation modes, yielding the six UR&ndash;JEPA losses studied in the paper. The matched-recipe LeJEPA baseline `L`<sup>SIGReg</sup> (last row, accessed via `baselines.sigreg.SIGReg`) is listed for reference.

| Paper notation | `variant` | Additional kwargs | Description | Reference |
|---|---|---|---|---|
| L<sup>CGLT</sup> *(headline)* | `"cglt"` | &mdash; | Gaussian-kernel smoothed dyadic density-difference (Carleson-type), integral form. | Chousionis&ndash;Garnett&ndash;Le&ndash;Tolsa, Thm. 1.1 |
| L<sup>CGLT, &part;log</sup> | `"cglt_deriv"` | &mdash; | Log scale-derivative form of the same characterization. | Chousionis&ndash;Garnett&ndash;Le&ndash;Tolsa, Thm. 1.2 |
| L<sup>CGLT, &part;</sup> | `"cglt_deriv_raw"` | &mdash; | Raw Eq.&nbsp;(1.5) integrand (literal Thm. 1.2 discretization, less stable). | Chousionis&ndash;Garnett&ndash;Le&ndash;Tolsa, Eq. (1.5) |
| L<sup>&beta;</sup> | `"beta"` | &mdash; | Jones &beta;-number via local PCA (bare). Collapses without an anti-collapse term. | Jones (1990); David&ndash;Semmes (1991) |
| L<sup>&beta;, &gamma;</sup> | `"beta"` | `gamma_logtrace=0.1` | &beta; paired with the intrinsic `&minus;&gamma; log tr(S)` anti-collapse penalty (no AD anchor needed). | this work |
| L<sup>&beta;, &gamma;, &tau;</sup> | `"beta"` | `gamma_logtrace=0.1`, `eigval_threshold=1.0` | Adds the adaptive eigenvalue threshold &tau;: tangent count is selected per anchor as `{i : &lambda;<sub>i</sub> > &tau; &middot; tr(S) / D}` instead of fixed top-n. | this work |
| L<sup>SIGReg</sup> *(baseline)* | (use `baselines.sigreg.SIGReg`) | &mdash; | LeJEPA's sliced characteristic-function statistic, included for matched-recipe comparison. | Balestriero &amp; LeCun, 2025 |

The `cglt*` variants additionally take a `lambda_ad` weight on the Ahlfors&ndash;David anchor term &mdash; canonical pairing for the CGLT family (paper default `lambda_ad=0.1`). For the β family the log-trace penalty replaces AD, so `lambda_ad=0` is the canonical setting.

Example: headline CGLT config (the row in Table 1 with the +0.83&nbsp;pp Inet10 result and the +18&nbsp;pp Galaxy10 vs IJEPA&ndash;IN22K result):

```python
ur = URJEPA(
    n=7, variant="cglt",
    n_scales=5,                # K = 5 dyadic scales
    lambda_ad=0.1,             # canonical AD-anchor weight for the CGLT family
)
```

Example: β rehabilitation (β + log-trace + eigenvalue threshold):

```python
ur = URJEPA(
    n=7, variant="beta",
    n_scales=5,
    gamma_logtrace=0.1,        # intrinsic anti-collapse
    eigval_threshold=1.0,      # adaptive tangent selection
    lambda_ad=0.0,             # AD anchor disabled for the β family
)
```

Example: SIGReg baseline (LeJEPA, for head-to-head comparison):

```python
from baselines.sigreg import SIGReg

sigreg = SIGReg(knots=17, num_slices=256)  # LeJEPA MINIMAL.md defaults

# Same call shape as URJEPA: forward accepts (V, N, D) or (N, D).
loss = sigreg(Z)
loss.backward()
```

`SIGReg.forward` has the same input shape as `URJEPA.forward`, so swapping the regularizer in a training loop is a one-line change. There is no `n` (intrinsic dimension) parameter &mdash; the SIGReg target is a full-dimensional isotropic Gaussian on the projector outputs.

Drop into a LeJEPA-style training step by replacing `sigreg(proj)` with `ur(proj)` &mdash; see [`scripts/pretrain.py`](scripts/pretrain.py) for the complete ViT / ResNet pretraining loop used in the paper.

---

## Reproducing the paper

Eleven sbatches under [`slurm/`](slurm/) reproduce every result table and figure in the paper. Each is self-documenting (header comments describe the recipe, expected wall time, GPU budget, and decision tree). The most commonly used ones:

| Result | Recipe |
|---|---|
| **ImageNet-10 headline** (Table&nbsp;1: UR&ndash;JEPA D=32, n=7, K=5, +0.83&nbsp;pp over SIGReg) | `sbatch slurm/run_K_sweep_projdim32_lejepa.sbatch` *(K=5 array tasks)* |
| **ImageNet-10 SIGReg matched-D=32 baseline** | `sbatch slurm/run_sigreg_projdim32_lejepa.sbatch` |
| **Galaxy10 SDSS UR&ndash;JEPA(L<sup>CGLT</sup>) headline** (Table&nbsp;5: 0.8142 &plusmn; 0.0017) | `sbatch slurm/run_galaxy10_verify.sbatch` |
| **Galaxy10 SDSS matched-recipe 6-variant set** (rest of Table&nbsp;5) | 5 more sbatches under `slurm/` &mdash; see [`slurm/README.md`](slurm/README.md) for the full mapping |
| **ImageNet-100 Stage-2** (Table&nbsp;7: 800-ep convergence) | `sbatch slurm/run_inet100_lejepa.sbatch` |
| **Projector geometry diagnostics** (Figures&nbsp;4&ndash;6: eigenvalue cliff) | `sbatch slurm/run_viz_geometry.sbatch` |

See [`slurm/README.md`](slurm/README.md) for the complete sbatch-to-paper mapping, the per-sbatch compute-budget table, and cluster-adaptation notes (the recipes ship with our Anvil `--account` / `--partition` / `SCRATCH` defaults; adapting to another SLURM site is a one-block edit).

For non-cluster runs, every `sbatch` file is just a shell wrapper around `python scripts/pretrain.py ...` with explicit Hydra flags; you can copy the python line and run it locally.

---

## Repository layout

```
ur_jepa/                Core library (CGLT, CGLT-deriv, β-number, AD anchor)
├── base.py             URTest interface, dyadic scale ladder
├── cglt.py             CGLT, CGLTDeriv, CGLTDerivRaw, ADRegularity
├── beta_number.py      BetaNumber (with log-trace and eig-threshold options)
└── ur_jepa.py          URJEPA: combined regularizer module (the public API)

baselines/sigreg.py     LeJEPA's SIGReg, included as a reference baseline

scripts/
├── pretrain.py         Main pretraining loop (Hydra-configured)
├── eval_downstream.py  Frozen-backbone linear-probe transfer
├── viz_geometry.py     Projector covariance + marginal-Gaussianity diagnostics
├── plot_trajectory.py  Per-epoch test-acc / loss plots
├── galaxy10_sdss.py    Galaxy10 SDSS dataset wrapper
└── aggregate_*.py      Multi-seed aggregation utilities

slurm/                  SLURM recipes for headline results (see above)
figures/                Figures referenced in the paper
```

---

## Citation

If you use UR&ndash;JEPA in academic work, please cite:

```bibtex
@misc{le2026urjepa,
  title={UR-JEPA: Uniform Rectifiability as a Regularizer for Joint-Embedding Predictive Architectures},
  author={Le, Triet M.},
  year={2026},
  eprint={2606.01443},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

The CGLT square-function characterization that the `cglt*` losses operationalize:

```bibtex
@article{cglt2014,
  title={Square functions and uniform rectifiability},
  author={Chousionis, Vasileios and Garnett, John and Le, Triet and Tolsa, Xavier},
  journal={Transactions of the American Mathematical Society},
  volume={368},
  number={9},
  pages={6063--6102},
  year={2016}
}
```

---

## Acknowledgments

This work was supported by an [NSF ACCESS](https://access-ci.org/) allocation on the [Anvil](https://www.rcac.purdue.edu/anvil) cluster at Purdue University's Rosen Center for Advanced Computing. We thank the LeJEPA authors ([Balestriero & LeCun, 2025](https://arxiv.org/abs/2511.08544)) for releasing the SIGReg implementation that this work builds on and compares against.

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE) for full text.

Copyright &copy; 2026 Spatiolyx LLC. UR&ndash;JEPA is open-source research software; the underlying paper, figures, and theoretical results are © Spatiolyx LLC and the authors.
