"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

SSL pretraining harness: paired comparison of regularizers across datasets.

Same backbone, augmentations, optimizer, schedule, and online probe across
all runs — the only thing that varies is the regularizer applied to the
projector outputs. Originally written for SIGReg vs UR-JEPA on Imagenette
(hence the legacy ``compare_inet10`` name); now generalized to the full
LeJEPA Table 5 dataset roster and the UR-JEPA variant catalogue.

Datasets (``+dataset=<key>``): see ``PRETRAIN_DATASETS`` below. Currently
``imagenette``, ``imagenet100``, ``galaxy10``, ``flowers102``, ``cifar10``,
``cifar100``, ``food101``. Loaders unify HuggingFace, torchvision, and the
custom Galaxy10 SDSS source behind a single ``{"image", "label"}`` schema.

Backbones (``+backbone=<key>``): see ``BACKBONES``. ``vit_s8_128`` is the
default for ckpt-compatibility; ``resnet18_lowres``, ``resnet18``, and
``resnext26ts`` cover the LeJEPA Table 5 architecture sweep.

Regularizers (``+regularizer=<name>``): ``sigreg``, ``ur_beta``,
``ur_cglt``, ``ur_cglt_deriv``, ``combined`` (SIGReg + UR-CGLT additive).

Features beyond a minimal launcher:
  * ``+seed=int`` (the minimal launcher hardcodes 0)
  * Checkpointing + per-epoch JSON results, written to ``+out_dir``
  * **Resume from checkpoint** via either:
      - ``+resume_from=<path>`` -- explicit path to a prior checkpoint
        (loads model + optimizer + scheduler + scaler state, continues
        from the next epoch).
      - ``+resume=true`` -- auto-discover an existing checkpoint in
        ``+out_dir``. Prefers ``ckpt-final.pt`` (run completed) then
        falls back to ``ckpt-latest.pt`` (run was in progress); if
        neither exists, trains from scratch. Makes the same sbatch
        re-runnable: rerun after an interrupt continues; rerun of a
        completed run exits cleanly without re-training.
    Periodic ``ckpt-latest.pt`` is written every ``+ckpt_every=N``
    epochs (default 50). Both resume modes support extending epochs:
    if you resume a 400-ep run with ``+epochs=800``, the scheduler is
    rebuilt for the 800-ep total and advanced to the saved step count.
  * Optional Wandb logging via ``+wandb=true``
  * Gradient accumulation (``+accum_steps=N``) for large-effective-batch
    runs on memory-bound GPUs
  * ``+n_schedule={fixed,step}`` for annealing the intrinsic-dim target
    of UR variants across epochs
  * **DDP support with projector all-gather.** When launched via torchrun
    (``LOCAL_RANK`` set), runs in distributed data-parallel mode and
    all-gathers projector outputs before the regularizer so the
    batch-statistic regularizers (SIGReg, CGLT) see the true effective
    batch — not just per-rank micro-batches. This is what lets us match
    LeJEPA's effective bs=512 recipe on a pair of GPUs without
    grad-accum's regularizer mismatch.

Usage (single GPU):
    python scripts/pretrain.py +regularizer=ur_beta +dataset=galaxy10 ...

Usage (2 GPUs, DDP):
    torchrun --nproc-per-node=2 scripts/pretrain.py +regularizer=ur_cglt ...
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import v2
import timm
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

from ur_jepa import URJEPA
from baselines import SIGReg


# ----------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------
def setup_dist():
    """Initialize torch.distributed if launched under torchrun/srun.

    Returns ``(rank, world_size, local_rank)``. Single-process fallback
    returns ``(0, 1, 0)`` so the script keeps working under plain
    ``python`` invocation.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def all_gather_with_grad(t: torch.Tensor, dim: int) -> torch.Tensor:
    """All-gather along ``dim`` with gradient through the local slice only.

    Uses the SimCLR-style pattern: non-differentiable all_gather to fetch
    all ranks' tensors, then splice the local tensor (which carries grad)
    back in. Each rank's autograd only flows through its own slice; DDP's
    parameter-grad all-reduce then averages the per-rank partial
    derivatives, recovering the true full-batch gradient exactly.

    Verified bit-identical to single-process full-batch backward on a
    URJEPA-CGLT smoke test (rel diff = 0.00e+00). The differentiable
    ``torch.distributed.nn.all_gather`` is intentionally NOT used here
    because its backward sums grads across ranks (W× too large).

    No-op when not distributed.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t
    ws, rank = dist.get_world_size(), dist.get_rank()
    gathered = [torch.empty_like(t) for _ in range(ws)]
    dist.all_gather(gathered, t.contiguous())
    gathered[rank] = t  # restore local grad linkage
    return torch.cat(gathered, dim=dim)

# Wandb is optional — only used if `+wandb=true`.
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


PROJECTOR_NORMS = {
    "bn": nn.BatchNorm1d,    # LeJEPA MINIMAL.md default
    "ln": nn.LayerNorm,      # per-sample alternative
    "none": None,            # no hidden-layer normalization
}


class CombinedReg(nn.Module):
    """Additive combination of SIGReg + URJEPA(cglt) with per-head weights.

    .. math::
       L = \\alpha_{\\text{sigreg}} \\cdot L_{\\text{sigreg}}(z)
          + \\alpha_{\\text{ur}} \\cdot s_{\\text{ur}} \\cdot L_{\\text{UR-CGLT}}(z)

    The internal ``ur_scale`` (default 1000) compensates URJEPA's small
    native magnitude so the alpha weights live on comparable scales:
    ``alpha_sigreg = alpha_ur = 1.0`` puts both losses at roughly
    equal effective magnitude.

    Hypothesis (worth testing): SIGReg enforces marginal Gaussianity
    (a weak full-D structural constraint via 1-D sliced characteristic
    function tests), while UR-CGLT enforces an n-dim rectifiability
    target. The two are theoretically in tension when n < D (one
    wants full-D spread, the other wants n-D concentration), but
    empirically they may be partially complementary: Diaconis--Freedman
    says most 1-D projections of any high-dim measure look Gaussian
    anyway, so SIGReg may act as a marginal-cleanup prior rather than
    a hard full-D enforcer when paired with CGLT.
    """

    def __init__(self, sigreg, urjepa,
                 alpha_sigreg: float = 1.0,
                 alpha_ur: float = 1.0,
                 ur_scale: float = 1000.0):
        super().__init__()
        self.sigreg = sigreg
        self.urjepa = urjepa
        self.alpha_sigreg = float(alpha_sigreg)
        self.alpha_ur = float(alpha_ur)
        self.ur_scale = float(ur_scale)

    def forward(self, proj):
        return (self.alpha_sigreg * self.sigreg(proj)
                + self.alpha_ur * self.ur_scale * self.urjepa(proj))


# ----------------------------------------------------------------------
# Backbone registry. Each entry tells ViTEncoder how to create the timm
# model and (for small-image SSL like Galaxy10 SDSS at 69x69 or CIFAR
# at 32x32) whether to apply the standard low-resolution conv1
# modification (kernel=3 stride=1, no maxpool) that SimCLR/BYOL/LeJEPA
# all use for sub-100px inputs.
#
# All backbones target a 512-d embedding via timm's `num_classes=512`
# head, which doubles as a learned linear bridge into the MLP projector.
# Keep this invariant so eval_downstream.py loads any ckpt uniformly.
# ----------------------------------------------------------------------
BACKBONES = {
    "vit_s8_128": {
        # Our existing default; do NOT change behavior -- ckpts depend on it.
        "timm_name": "vit_small_patch8_224",
        "img_size": 128,
        "low_resolution": False,
    },
    "resnet18_lowres": {
        # For Galaxy10 SDSS (69x69) and CIFAR-class (32x32) inputs.
        # Standard SSL trick: replace conv1 with 3x3 stride=1 + drop the
        # maxpool, so the receptive field is appropriate for sub-100px
        # inputs and we don't lose half the spatial info in the first
        # block. Matches LeJEPA / stable-pretraining's `low_resolution`
        # flag for resnet on CIFAR-style data.
        "timm_name": "resnet18",
        "img_size": 69,   # default; per-dataset transforms override
        "low_resolution": True,
    },
    "resnet18": {
        # Standard ResNet18 for higher-res inputs (224x224, etc.).
        "timm_name": "resnet18",
        "img_size": 224,
        "low_resolution": False,
    },
    "resnext26ts": {
        # Tiny ResNeXt-26 ("ts" = tiny stem variant in timm). 8M params.
        # LeJEPA Table 5's resnext26ts row reaches 82.19% on flowers102,
        # their strongest small-architecture cell there. Standard 224 input.
        "timm_name": "resnext26ts",
        "img_size": 224,
        "low_resolution": False,
    },
}


class ViTEncoder(nn.Module):
    """Backbone + 3-layer MLP projector. Name kept ``ViTEncoder`` for
    backwards compatibility with the saved-ckpt state_dict (the legacy
    code wrapped a ViT here; any backbone in ``BACKBONES`` is now
    supported via ``+backbone=<key>``).

    ``projector_norm`` controls the norm layer used *between* hidden
    linear layers of the projector. The final ``Linear(2048, proj_dim)``
    output is unconstrained regardless — torchvision's ``MLP`` puts the
    norm/activation after every Linear except the last.

    Default ``backbone='vit_s8_128'`` reproduces the exact pre-registry
    behavior, so existing sbatches keep working without modification.
    """

    def __init__(
        self,
        proj_dim: int = 128,
        projector_norm: str = "bn",
        backbone: str = "vit_s8_128",
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if projector_norm not in PROJECTOR_NORMS:
            raise ValueError(
                f"projector_norm must be one of {list(PROJECTOR_NORMS)}, "
                f"got {projector_norm!r}"
            )
        if backbone not in BACKBONES:
            raise ValueError(
                f"backbone must be one of {list(BACKBONES)}, got {backbone!r}"
            )
        spec = BACKBONES[backbone]

        create_kwargs = dict(
            pretrained=False,
            num_classes=512,
            drop_path_rate=float(drop_path_rate),
        )
        # img_size is a ViT-specific kwarg; conv backbones infer from input.
        if "vit" in spec["timm_name"]:
            create_kwargs["img_size"] = spec["img_size"]
        self.backbone = timm.create_model(spec["timm_name"], **create_kwargs)
        if spec["low_resolution"]:
            self._apply_low_resolution(self.backbone)

        self.proj = MLP(
            512, [2048, 2048, proj_dim],
            norm_layer=PROJECTOR_NORMS[projector_norm],
        )

    @staticmethod
    def _apply_low_resolution(model: nn.Module) -> None:
        """In-place modification: replace conv1 (kernel=7 stride=2) with
        (kernel=3 stride=1) and drop the maxpool. Standard practice for
        SSL on sub-100px inputs (CIFAR, Galaxy10, etc.)."""
        if hasattr(model, "conv1"):
            old = model.conv1
            model.conv1 = nn.Conv2d(
                old.in_channels, old.out_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            )
        if hasattr(model, "maxpool"):
            model.maxpool = nn.Identity()

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


# ----------------------------------------------------------------------
# Pretrain dataset registry. Each entry tells HFDataset how to load the
# HuggingFace dataset and the online probe how many classes to size.
# ----------------------------------------------------------------------
PRETRAIN_DATASETS = {
    "imagenette": {
        "type": "hf",
        "hf_name": "frgfm/imagenette",
        "hf_config": "160px",
        "n_classes": 10,
        "train_split": "train",
        "val_split": "validation",
    },
    "imagenet100": {
        # CMC ImageNet-100 (100-class random subset of ImageNet-1k),
        # 126,689 train / 5,000 val, 160px short side, parquet, no auth.
        # https://huggingface.co/datasets/clane9/imagenet-100
        "type": "hf",
        "hf_name": "clane9/imagenet-100",
        "hf_config": None,
        "n_classes": 100,
        "train_split": "train",
        "val_split": "validation",
    },
    "galaxy10": {
        # Galaxy10 SDSS (the original 21,785-image 69x69 dataset used by
        # LeJEPA's Table 5). Loaded via scripts/galaxy10_sdss.py since
        # it's distributed as a Zenodo HDF5, not on HuggingFace.
        # We apply a deterministic 50/50 stratified split (random_state=42)
        # that approximates LeJEPA's unpublished 11,008-train split.
        "type": "galaxy10_sdss",
        "n_classes": 10,
        "train_split": "train",
        "val_split": "test",
    },
    "eurosat": {
        # EuroSAT RGB (Sentinel-2 land-cover, 10 classes, 27,000 tiles at
        # 64x64). Earth-observation remote sensing -- a second non-natural
        # domain alongside Galaxy10 for the "in-domain SSL beats
        # foundation-model transfer" claim. timm-maintained RGB build with
        # the standard 16,200 / 5,400 / 5,400 train/validation/test split
        # (no auth, parquet, {"image": PIL, "label": int} schema).
        # https://huggingface.co/datasets/timm/eurosat-rgb
        # Like Galaxy10 (69x69), pretrain with +backbone=resnet18_lowres;
        # the default img_size=128 RandomResizedCrop upscales from 64.
        "type": "hf",
        "hf_name": "timm/eurosat-rgb",
        "hf_config": None,
        "n_classes": 10,
        "train_split": "train",
        "val_split": "validation",
    },
    "flowers102": {
        # Oxford Flowers102 via torchvision. 1,020 train / 1,020 val /
        # 6,149 test, 102 fine-grained flower species. Matches LeJEPA
        # Table 5's "1020 #train samples" cell. We pretrain on the
        # train split (matching their convention) and use test as the
        # online-probe eval split (larger -> tighter probe accuracy).
        "type": "torchvision",
        "tv_class": "Flowers102",
        "tv_split_kwargs": {
            "train": {"split": "train"},   # 1020 imgs
            "val":   {"split": "test"},    # 6149 imgs (larger for stable probe)
        },
        "n_classes": 102,
        "train_split": "train",
        "val_split": "val",
    },
    "cifar10": {
        # CIFAR-10: 50K train / 10K test, 10 general object classes,
        # native 32x32 RGB. LeJEPA Table 5 cifar10 column. Note: their
        # recipe upscales 32x32 -> 224 via RandomResizedCrop -- we do
        # the same to match.
        "type": "torchvision",
        "tv_class": "CIFAR10",
        "tv_split_kwargs": {
            "train": {"train": True},
            "val":   {"train": False},
        },
        "n_classes": 10,
        "train_split": "train",
        "val_split": "val",
    },
    "cifar100": {
        # CIFAR-100: 50K train / 10K test, 100 general object classes,
        # native 32x32 RGB. Same upscaling caveat as cifar10.
        "type": "torchvision",
        "tv_class": "CIFAR100",
        "tv_split_kwargs": {
            "train": {"train": True},
            "val":   {"train": False},
        },
        "n_classes": 100,
        "train_split": "train",
        "val_split": "val",
    },
    "food101": {
        # Food-101: 75,750 train / 25,250 test, 101 food categories,
        # native 512 max-side. Largest dataset in LeJEPA Table 5 --
        # significant per-step wall budget required.
        "type": "torchvision",
        "tv_class": "Food101",
        "tv_split_kwargs": {
            "train": {"split": "train"},
            "val":   {"split": "test"},
        },
        "n_classes": 101,
        "train_split": "train",
        "val_split": "val",
    },
}


class HFDataset(torch.utils.data.Dataset):
    """Pretrain dataset wrapper. Dispatches between HuggingFace
    ``load_dataset`` and our custom Galaxy10 SDSS loader via the
    ``type`` field in ``PRETRAIN_DATASETS``. All sources expose the same
    ``{"image": PIL.Image, "label": int}`` schema downstream of this
    class, so the SSL transform stack is identical."""

    def __init__(self, dataset: str, split: str, V: int = 1, img_size: int = 128):
        if dataset not in PRETRAIN_DATASETS:
            raise ValueError(
                f"Unknown pretrain dataset {dataset!r}. "
                f"Choose from {list(PRETRAIN_DATASETS)}"
            )
        spec = PRETRAIN_DATASETS[dataset]
        self.V = V
        ds_type = spec.get("type", "hf")  # back-compat: missing "type" => hf
        if ds_type == "hf":
            self.ds = self._load_hf(spec, split)
        elif ds_type == "galaxy10_sdss":
            # Import lazily so we don't pay the h5py + sklearn import cost
            # for HF-only runs.
            from galaxy10_sdss import Galaxy10SDSS
            self.ds = Galaxy10SDSS(split=split)
        elif ds_type == "torchvision":
            self.ds = self._load_torchvision(spec, split)
        else:
            raise ValueError(
                f"Unknown dataset type {ds_type!r} for dataset {dataset!r}"
            )

        # Shared SSL aug + eval transform stack. img_size is configurable
        # so LeJEPA's 224x224 recipe (their default for Table 5
        # in-domain pretraining) can be selected via +img_size=224. The
        # default 128 preserves existing sbatch behavior.
        # Resize-to-(img_size * 1.14) on eval matches the standard
        # 256/224 ratio used in ImageNet evaluation pipelines.
        self.img_size = int(img_size)
        eval_resize = int(round(self.img_size * 256.0 / 224.0))
        self.aug = v2.Compose([
            v2.RandomResizedCrop(self.img_size, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test = v2.Compose([
            v2.Resize(eval_resize),
            v2.CenterCrop(self.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _load_hf(spec, split):
        if spec["hf_config"] is not None:
            return load_dataset(spec["hf_name"], spec["hf_config"], split=split)
        return load_dataset(spec["hf_name"], split=split)

    @staticmethod
    def _load_torchvision(spec, split):
        """Load a torchvision dataset and wrap its (img, label) tuples in
        the HuggingFace-style ``{"image": PIL, "label": int}`` schema so
        ``HFDataset.__getitem__`` works uniformly regardless of source.

        Different torchvision datasets use different split-selection
        APIs: Flowers102/Food101 use ``split="train"/"val"/"test"``,
        while CIFAR10/CIFAR100 use ``train=True/False``. To handle both
        uniformly the spec stores ``tv_split_kwargs`` -- a dict mapping
        logical split names ('train', 'val') to the kwargs dict that
        torchvision's constructor expects for that split.

        Honours ``$SCRATCH/.cache/torchvision`` if SCRATCH is set
        (Anvil/AWS bootstrap convention)."""
        import os
        from torchvision import datasets as tvds
        scratch = os.environ.get("SCRATCH")
        data_root = str((Path(scratch) if scratch else Path.home())
                        / ".cache" / "torchvision")
        Path(data_root).mkdir(parents=True, exist_ok=True)
        kwargs = spec["tv_split_kwargs"][split]
        cls = getattr(tvds, spec["tv_class"])
        tv_ds = cls(root=data_root, download=True, **kwargs)
        return _TVDictWrapper(tv_ds)


class _TVDictWrapper:
    """Wrap a torchvision Dataset whose __getitem__ returns ``(img, label)``
    so it returns ``{"image": img, "label": int(label)}`` instead --
    matching the schema HFDataset.__getitem__ assumes for HuggingFace
    and Galaxy10SDSS sources."""

    def __init__(self, tv_ds):
        self.tv_ds = tv_ds

    def __len__(self):
        return len(self.tv_ds)

    def __getitem__(self, i):
        img, label = self.tv_ds[i]
        return {"image": img, "label": int(label)}


def build_regularizer(cfg) -> nn.Module:
    """Construct the regularizer that ``cfg.regularizer`` selects.

    For UR variants the initial ``n`` is ``cfg.n`` (the schedule's start
    value when ``n_schedule != 'fixed'``).
    """
    name = cfg.regularizer
    if name == "sigreg":
        # LeJEPA MINIMAL.md uses num_slices=256, knots=17.
        return SIGReg(knots=cfg.get("knots", 17), num_slices=cfg.get("num_slices", 256))
    if name == "ur_beta":
        # Two anti-collapse options for the beta variant (β² alone has a
        # trivial point-mass minimum, so one of these is needed to learn):
        #   * +lambda_ad=0.1      (external) AD anchor on top of β²
        #   * +gamma_logtrace=X   (intrinsic) -γ·mean log trace(S_r(x))
        # Both default to 0; pass at least one to actually train.
        # gamma_logtrace flows to BetaNumber via URJEPA's **variant_kwargs.
        #
        # +eigval_threshold=X (default 0): when > 0, swap the top-n
        # tangent selection for an adaptive rule -- keep λ_i with
        # λ_i > τ · trace(S) / D ("variance-share relative to uniform").
        # Sweep e.g. τ ∈ {0.5, 1.0, 2.0, 5.0} to find the right
        # tangent/normal separator for the dataset.
        #
        # +log_beta_eps=X (default 0): when > 0, replace β² with
        # log(β² + log_beta_eps) per anchor before the scale weighting.
        # Compresses dynamic range of β² across scales. Does NOT change
        # the collapse mode, so still requires lambda_ad or
        # gamma_logtrace for non-trivial training.
        return URJEPA(
            n=cfg.n, variant="beta", n_scales=cfg.get("n_scales", 5),
            anchors=cfg.get("anchors", None),
            lambda_ad=cfg.get("lambda_ad", 0.0),
            gamma_logtrace=cfg.get("gamma_logtrace", 0.0),
            eigval_threshold=cfg.get("eigval_threshold", 0.0),
            log_beta_eps=cfg.get("log_beta_eps", 0.0),
        )
    if name == "ur_cglt":
        return URJEPA(
            n=cfg.n, variant="cglt", n_scales=cfg.get("n_scales", 5),
            anchors=cfg.get("anchors", None),
            lambda_ad=cfg.get("lambda_ad", 0.1),
        )
    if name == "ur_cglt_deriv":
        # Eq.~(1.5) scale-derivative variant of UR-CGLT, in LOG form
        # (t · d/dt log theta_t = ⟨d²/t² - n⟩_w). Same AD-anchor pairing
        # as ur_cglt (theorem 1.2 in CGLT presupposes n-AD regularity
        # just as theorem 1.1 does).
        return URJEPA(
            n=cfg.n, variant="cglt_deriv", n_scales=cfg.get("n_scales", 5),
            anchors=cfg.get("anchors", None),
            lambda_ad=cfg.get("lambda_ad", 0.1),
        )
    if name == "ur_cglt_deriv_raw":
        # Literal Eq.~(1.5) of CGLT 2014, no log transformation:
        # Δ̃ = (1/(N t^n)) Σ_j exp(-d²_j/(2t²)) · (d²_j/t² - n).
        # The θ_t prefactor couples loss to local density; relies on
        # the AD anchor to push toward AD-regularity, which is the
        # precondition Theorem 1.2 needs.
        return URJEPA(
            n=cfg.n, variant="cglt_deriv_raw", n_scales=cfg.get("n_scales", 5),
            anchors=cfg.get("anchors", None),
            lambda_ad=cfg.get("lambda_ad", 0.1),
        )
    if name == "combined":
        # SIGReg + URJEPA(cglt) additive combination. Each component
        # is built with the same kwargs it'd see when used alone.
        return CombinedReg(
            sigreg=SIGReg(
                knots=cfg.get("knots", 17),
                num_slices=cfg.get("num_slices", 256),
            ),
            urjepa=URJEPA(
                n=cfg.n, variant="cglt",
                n_scales=cfg.get("n_scales", 5),
                anchors=cfg.get("anchors", None),
                lambda_ad=cfg.get("lambda_ad", 0.1),
            ),
            alpha_sigreg=cfg.get("alpha_sigreg", 1.0),
            alpha_ur=cfg.get("alpha_ur", 1.0),
            ur_scale=cfg.get("ur_scale", 1000.0),
        )
    raise ValueError(
        f"Unknown regularizer {name!r} "
        f"(use sigreg | ur_beta | ur_cglt | ur_cglt_deriv | "
        f"ur_cglt_deriv_raw | combined)"
    )


def schedule_n(cfg, epoch: int) -> int:
    """Return the target intrinsic dim ``n`` for this epoch.

    ``cfg.n_schedule``:
      * ``fixed`` (default): always ``cfg.n``.
      * ``step``: start at ``cfg.n``, drop by 1 every ``cfg.n_every`` epochs,
        floor at ``cfg.n_end``. Used by run_anneal_n.sbatch to discover
        the data-driven intrinsic dim.
    """
    sched = cfg.get("n_schedule", "fixed")
    if sched == "fixed":
        return int(cfg.n)
    if sched == "step":
        level = epoch // int(cfg.n_every)
        return max(int(cfg.n_end), int(cfg.n) - level)
    raise ValueError(f"Unknown n_schedule {sched!r} (use fixed | step)")


def set_target_n(reg: nn.Module, n: int) -> None:
    """Update the target intrinsic dim on whatever regularizer is in use.

    No-op for SIGReg (no n). For URJEPA, updates both the UR head (CGLT
    or β) and the AD anchor. For CombinedReg, recurses into the nested
    URJEPA.
    """
    inner = getattr(reg, "ur", None)
    if inner is not None and hasattr(inner, "n"):
        inner.n = int(n)
    ad = getattr(reg, "ad", None)
    if ad is not None and hasattr(ad, "n"):
        ad.n = int(n)
    # CombinedReg holds a nested URJEPA at .urjepa; recurse.
    nested = getattr(reg, "urjepa", None)
    if nested is not None:
        set_target_n(nested, n)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    rank, world_size, local_rank = setup_dist()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg.out_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Different seed offset per rank for augmentations / dataloader
    # randomness; the base seed is preserved for reproducibility.
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    use_wandb = bool(cfg.get("wandb", False)) and HAS_WANDB and is_main()
    if use_wandb:
        wandb.init(
            project=cfg.get("wandb_project", "UR-JEPA"),
            name=f"{cfg.regularizer}_s{seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(out_dir),
        )

    # Pretrain dataset switch: defaults to imagenette for backwards
    # compatibility with every existing sbatch. Pass +dataset=imagenet100
    # to swap. The online probe sizes itself from the registry's n_classes.
    dataset_name = cfg.get("dataset", "imagenette")
    if dataset_name not in PRETRAIN_DATASETS:
        raise ValueError(
            f"Unknown pretrain dataset {dataset_name!r}. "
            f"Choose from {list(PRETRAIN_DATASETS)}"
        )
    dataset_spec = PRETRAIN_DATASETS[dataset_name]
    n_classes = int(dataset_spec["n_classes"])

    img_size = int(cfg.get("img_size", 128))
    train_ds = HFDataset(dataset_name, dataset_spec["train_split"], V=cfg.V, img_size=img_size)
    test_ds = HFDataset(dataset_name, dataset_spec["val_split"], V=1, img_size=img_size)
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True,
        )
        train = DataLoader(
            train_ds, batch_size=cfg.bs, sampler=train_sampler,
            drop_last=True, num_workers=cfg.get("num_workers", 8),
            pin_memory=True,
        )
    else:
        train_sampler = None
        train = DataLoader(
            train_ds, batch_size=cfg.bs, shuffle=True,
            drop_last=True, num_workers=cfg.get("num_workers", 8),
        )
    # Test eval is rank-0 only (small dataset, no point splitting).
    test = DataLoader(test_ds, batch_size=256, num_workers=cfg.get("num_workers", 8))

    net = ViTEncoder(
        proj_dim=cfg.proj_dim,
        projector_norm=cfg.get("projector_norm", "bn"),
        backbone=cfg.get("backbone", "vit_s8_128"),
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
    ).to(device)
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, n_classes)).to(device)
    reg = build_regularizer(cfg).to(device)

    # Multi-GPU: convert BN to SyncBN so projector's BN sees gathered
    # statistics, then wrap in DDP.
    if world_size > 1:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[local_rank])
        probe = DDP(probe, device_ids=[local_rank])

    net_params = (net.module if isinstance(net, DDP) else net).parameters()
    probe_params = (probe.module if isinstance(probe, DDP) else probe).parameters()
    g1 = {"params": net_params, "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe_params, "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])

    # Gradient accumulation: take `accum_steps` micro-batches before each
    # optimizer step → effective batch size = accum_steps * cfg.bs without
    # the GPU-memory cost. Scheduler and warmup are in OPTIMIZER steps,
    # not micro-batches, so they match what large-batch training would do.
    accum_steps = max(1, int(cfg.get("accum_steps", 1)))
    opt_steps_per_epoch = max(1, len(train) // accum_steps)
    warmup_steps = opt_steps_per_epoch
    total_steps = opt_steps_per_epoch * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    # eta_min: literal 1e-3 to match LeJEPA MINIMAL.md
    # (https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md), which
    # hardcodes `CosineAnnealingLR(..., eta_min=1e-3)` regardless of the
    # initial lr. The LeJEPA README's verbal "decay to initial_lr / 1000"
    # description does NOT match their actual code; the code is what
    # produced their published 0.907 number, so we follow it literally.
    # Note: at cfg.lr <= 1e-3 the cosine annealing ramps UP from cfg.lr
    # to 1e-3; pass +eta_min=<smaller> in that regime.
    eta_min = float(cfg.get("eta_min", 1e-3))
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    # bf16 autocast doesn't need a loss scaler; keep enabled=False to
    # silence the warning that GradScaler emits under non-fp16 autocast.
    scaler = GradScaler(enabled=False)

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "seed": seed,
        "epochs": [],            # list of dicts: {epoch, reg_loss, inv_loss, probe_loss, test_acc}
        "start_time": time.time(),
    }

    best_acc = 0.0
    start_epoch = 0

    # ------------------------------------------------------------------
    # Resume from a prior checkpoint (optional)
    # ------------------------------------------------------------------
    # +resume_from=<path>: load net/probe/optimizer/scheduler/scaler state
    # from a prior ckpt-latest.pt (or ckpt-final.pt) and continue training
    # from the next epoch. Supports extending epochs (e.g., resuming a
    # 400-ep run at +epochs=800): the scheduler is rebuilt above with the
    # NEW cfg.epochs total, then advanced to the saved step count, so
    # the cosine spans 0..cfg.epochs and we land where we'd be in that
    # extended schedule. NB: at extension, the LR at resume may jump up
    # vs the original schedule's end-of-run LR -- this is the warm-restart
    # behavior of "the original epochs target was wrong, here's the right
    # schedule." If you want to continue with the original schedule
    # exactly, do not pass +resume_from with a changed +epochs.
    # +resume=true: auto-discover an existing ckpt in out_dir. Prefer
    # ckpt-final.pt (run completed) then fall back to ckpt-latest.pt
    # (run was in progress). If neither exists, fall through to
    # training-from-scratch silently. Explicit +resume_from=<path>
    # takes precedence (both can be set; the explicit path wins).
    resume_auto = bool(cfg.get("resume", False))
    resume_from = cfg.get("resume_from", None)
    if resume_auto and not resume_from:
        for cand_name in ("ckpt-final.pt", "ckpt-latest.pt"):
            cand = out_dir / cand_name
            if cand.exists():
                resume_from = str(cand)
                if is_main():
                    print(f"--- resume=true: discovered {cand}, resuming ---")
                break
        if resume_from is None and is_main():
            print(
                f"--- resume=true: no ckpt found in {out_dir}, "
                f"training from scratch ---"
            )

    if resume_from:
        if is_main() and not resume_auto:
            print(f"--- resuming from {resume_from} ---")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)

        save_net = net.module if isinstance(net, DDP) else net
        save_probe = probe.module if isinstance(probe, DDP) else probe

        save_net.load_state_dict(ckpt["net"])
        save_probe.load_state_dict(ckpt["probe"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        # Note: scheduler is NOT restored from state_dict. Instead we
        # advance the freshly-built scheduler (which already has
        # cfg.epochs total_steps baked in) to the saved step count.
        # This is what makes "extend epochs on resume" work cleanly.

        start_epoch = int(ckpt["epoch"]) + 1
        best_acc = float(ckpt.get("best_acc", 0.0))

        # Restore prior per-epoch trajectory. The new ckpt format
        # embeds a "results" dict; older ckpts (pre-2026-05-28) don't.
        # Fall back to reading the prior results.json from out_dir so
        # we can resume from those old checkpoints without losing the
        # 0..start_epoch epoch trajectory in the new results.json.
        prior_results = ckpt.get("results", None)
        if prior_results is None:
            prior_results_file = out_dir / "results.json"
            if prior_results_file.exists():
                try:
                    prior_results = json.loads(prior_results_file.read_text())
                    if is_main():
                        print(
                            f"--- ckpt has no embedded results dict; recovered "
                            f"{len(prior_results.get('epochs', []))} prior epochs "
                            f"from {prior_results_file} ---"
                        )
                except json.JSONDecodeError:
                    prior_results = None
        if prior_results is not None:
            results["epochs"] = prior_results.get("epochs", [])
            results["start_time"] = prior_results.get("start_time", results["start_time"])
            # Backfill best_acc from the prior trajectory if the ckpt
            # didn't store it explicitly (old format).
            if best_acc == 0.0:
                prior_best = max(
                    (ep.get("test_acc", 0.0) for ep in results["epochs"]),
                    default=0.0,
                )
                if prior_best > 0.0:
                    best_acc = float(prior_best)

        if start_epoch >= cfg.epochs:
            if resume_auto:
                # Idempotent re-run: rerunning a completed sbatch with
                # +resume=true is a no-op rather than a crash. We do
                # NOT rewrite results.json or ckpt-final.pt -- the
                # prior run's artifacts stand as the canonical record.
                if is_main():
                    print(
                        f"--- resume=true: ckpt is at epoch "
                        f"{ckpt['epoch']} >= cfg.epochs={cfg.epochs}; "
                        f"training already complete, exiting cleanly ---"
                    )
                if use_wandb:
                    wandb.finish()
                if dist.is_initialized():
                    dist.destroy_process_group()
                return
            raise ValueError(
                f"+resume_from checkpoint is at epoch {ckpt['epoch']} which is "
                f">= cfg.epochs={cfg.epochs}. Set +epochs larger than the "
                f"checkpoint epoch to extend training."
            )

        # Advance scheduler to current step. Built above with NEW cfg.epochs,
        # so this gives a coherent cosine over 0..cfg.epochs that we're now
        # `start_epoch * opt_steps_per_epoch` steps into.
        current_step = start_epoch * opt_steps_per_epoch
        for _ in range(current_step):
            scheduler.step()

        if is_main():
            print(
                f"--- resumed at epoch {start_epoch}/{cfg.epochs}, "
                f"lr={opt.param_groups[0]['lr']:.2e}, best_acc={best_acc:.4f}, "
                f"{len(results['epochs'])} epochs in prior results ---"
            )

    def _save_full_checkpoint(path: Path) -> None:
        """Full checkpoint with model + optimizer + scheduler + scaler state.

        Saved every ``cfg.ckpt_every`` epochs (default 50) AND at the end
        of training. Use ``+resume_from=<path>`` to continue training.
        """
        save_net_local = net.module if isinstance(net, DDP) else net
        save_probe_local = probe.module if isinstance(probe, DDP) else probe
        torch.save({
            "epoch": epoch,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "net": save_net_local.state_dict(),
            "probe": save_probe_local.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": best_acc,
            "results": results,
        }, path)

    ckpt_every = int(cfg.get("ckpt_every", 50))

    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # reshuffle for DistributedSampler

        # If n_schedule is active, update the regularizer's target dim
        # before the epoch starts. No-op for sigreg (no n).
        current_n = schedule_n(cfg, epoch)
        set_target_n(reg, current_n)

        net.train(); probe.train()
        epoch_logs = {"reg": [], "inv": [], "probe": []}
        opt.zero_grad()
        batch_iter = train if not is_main() else tqdm.tqdm(
            train, total=len(train), desc=f"ep{epoch} n={current_n}"
        )
        for micro_idx, (vs, y) in enumerate(batch_iter):
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                emb, proj = net(vs)

                # Per-rank invariance and probe losses: each rank sees
                # different images so these are inherently per-rank;
                # DDP averages the gradients automatically.
                inv_loss = (proj.mean(0) - proj).square().mean()
                yhat = probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y.repeat_interleave(cfg.V))

                # Regularizer: gather projector outputs across ranks so
                # SIGReg / CGLT see the true effective batch. This is the
                # fix that grad-accum couldn't provide — batch-statistic
                # losses can't be emulated by per-microbatch evaluation.
                # proj has shape (V, N_local, D); gather along the N axis.
                proj_gathered = all_gather_with_grad(proj, dim=1)
                # reg_scale rescales the regularizer's output before the
                # convex combo so UR-JEPA's raw loss (~10^5x smaller than
                # SIGReg's) can reach a comparable effective weight.
                reg_loss = reg(proj_gathered) * float(cfg.get("reg_scale", 1.0))

                urjepa_loss = reg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                # Divide by accum_steps so the accumulated grad matches a
                # single-step large-batch gradient.
                loss = (urjepa_loss + probe_loss) / accum_steps

            scaler.scale(loss).backward()

            # Optimizer step every accum_steps micro-batches.
            if (micro_idx + 1) % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()

            epoch_logs["reg"].append(reg_loss.item())
            epoch_logs["inv"].append(inv_loss.item())
            epoch_logs["probe"].append(probe_loss.item())

            if use_wandb:
                wandb.log({
                    "train/reg": reg_loss.item(),
                    "train/inv": inv_loss.item(),
                    "train/probe": probe_loss.item(),
                })

        # Eval — rank 0 only. Other ranks wait at the barrier below.
        acc = 0.0
        if is_main():
            net.eval(); probe.eval()
            correct = 0
            eval_net = net.module if isinstance(net, DDP) else net
            eval_probe = probe.module if isinstance(probe, DDP) else probe
            with torch.inference_mode():
                for vs, y in test:
                    vs = vs.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with autocast("cuda", dtype=torch.bfloat16):
                        correct += (eval_probe(eval_net(vs)[0]).argmax(1) == y).sum().item()
            acc = correct / len(test_ds)
            best_acc = max(best_acc, acc)
        if dist.is_initialized():
            dist.barrier()

        epoch_summary = {
            "epoch": epoch,
            "n_target": current_n,  # constant under fixed schedule; varies under step
            "reg_loss": float(sum(epoch_logs["reg"]) / len(epoch_logs["reg"])),
            "inv_loss": float(sum(epoch_logs["inv"]) / len(epoch_logs["inv"])),
            "probe_loss": float(sum(epoch_logs["probe"]) / len(epoch_logs["probe"])),
            "test_acc": acc,
        }
        results["epochs"].append(epoch_summary)
        if use_wandb:
            wandb.log({"test/acc": acc, "test/epoch": epoch})

        # Persist results every epoch so partially-completed runs are
        # still useful. Rank-0 only — other ranks have nothing to save.
        if is_main():
            (out_dir / "results.json").write_text(json.dumps(results, indent=2))

            # Periodic full checkpoint (model + optimizer + scheduler +
            # scaler) every ckpt_every epochs. Used for resume via
            # +resume_from=<path>. Always overwrite ckpt-latest.pt so
            # the resume target is unambiguous.
            if (epoch + 1) % ckpt_every == 0:
                _save_full_checkpoint(out_dir / "ckpt-latest.pt")

    if is_main():
        results["final_acc"] = results["epochs"][-1]["test_acc"]
        results["best_acc"] = best_acc
        results["wall_seconds"] = time.time() - results["start_time"]
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

        # Save final checkpoint (unwrap DDP first so the saved state dict
        # is directly loadable by a single-GPU run). Same full-state
        # format as ckpt-latest.pt so it's also usable as a resume target.
        _save_full_checkpoint(out_dir / "ckpt-final.pt")

    if use_wandb:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
