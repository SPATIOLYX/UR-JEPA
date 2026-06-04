"""
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026

Downstream linear-probe eval from a saved UR-JEPA / SIGReg checkpoint.

Loads ``ckpt-final.pt`` written by ``pretrain.py``, freezes the
backbone, attaches a fresh ``LayerNorm(D) -> Linear(D, num_classes)``
probe, trains it on a target dataset, and writes ``results.json`` with
per-epoch test accuracy.

Supported datasets (matches LeJEPA's evaluation matrix):
    aircraft     -- FGVC-Aircraft       (100 cls,  6,667 / 3,333)
    cars         -- StanfordCars        (196 cls,  8,144 / 8,041)
    cifar10      -- CIFAR-10            ( 10 cls, 50,000 /10,000)
    cifar100     -- CIFAR-100           (100 cls, 50,000 /10,000)
    dtd          -- DTD                 ( 47 cls,  1,880 / 1,880)
    flowers      -- Flowers-102         (102 cls,  1,020 / 6,149)
    food         -- Food-101            (101 cls, 75,750 /25,250)
    pets         -- Oxford-IIIT-Pet     ( 37 cls,  3,680 / 3,669)

Usage:
    python scripts/eval_downstream.py \\
        +ckpt=$SCRATCH/projects/UR-JEPA/runs/lejepa_exact/ur_cglt_s0/ckpt-final.pt \\
        +dataset=aircraft \\
        +seed=0 \\
        +out_dir=$SCRATCH/projects/UR-JEPA/runs/downstream_aircraft/ur_cglt_s0

Notes on protocol:
* The probe is fit on the **concatenation of CLS tokens from the last
  two transformer blocks** (768-d for ViT-S/8: 384 + 384), matching
  LeJEPA's published protocol. The CLS tokens are captured via forward
  hooks on ``backbone.blocks[-2]`` and ``backbone.blocks[-1]``.
* Backbone is frozen (``requires_grad=False``, ``eval()``), so BN
  running stats are not updated and dropout is off.
* The probe is ``LayerNorm(768) -> Linear(768, num_classes)``, trained
  with AdamW (wd=1e-6) + 1-epoch linear warmup + cosine annealing to
  1e-6, matching LeJEPA's verbal description.
* Train aug: RandomResizedCrop(128) + horizontal flip + normalize.
  Test: Resize(146) + CenterCrop(128) + normalize.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets as tvds
from torchvision.transforms import v2
from torch.amp import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import timm
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Dataset registry. Each entry: (torchvision class, n_classes, train_kwargs,
# test_kwargs). Datasets that use ``split=``/``train=`` differently are
# normalized here.
# ---------------------------------------------------------------------------
DATASETS = {
    "aircraft": dict(
        cls=tvds.FGVCAircraft, n_classes=100,
        train_kwargs={"split": "trainval", "annotation_level": "variant"},
        test_kwargs={"split": "test", "annotation_level": "variant"},
    ),
    "cars": dict(
        cls=tvds.StanfordCars, n_classes=196,
        train_kwargs={"split": "train"},
        test_kwargs={"split": "test"},
    ),
    "cifar10": dict(
        cls=tvds.CIFAR10, n_classes=10,
        train_kwargs={"train": True},
        test_kwargs={"train": False},
    ),
    "cifar100": dict(
        cls=tvds.CIFAR100, n_classes=100,
        train_kwargs={"train": True},
        test_kwargs={"train": False},
    ),
    "dtd": dict(
        cls=tvds.DTD, n_classes=47,
        train_kwargs={"split": "train"},
        test_kwargs={"split": "test"},
    ),
    "flowers": dict(
        cls=tvds.Flowers102, n_classes=102,
        train_kwargs={"split": "train"},
        test_kwargs={"split": "test"},
    ),
    "food": dict(
        cls=tvds.Food101, n_classes=101,
        train_kwargs={"split": "train"},
        test_kwargs={"split": "test"},
    ),
    "pets": dict(
        cls=tvds.OxfordIIITPet, n_classes=37,
        train_kwargs={"split": "trainval"},
        test_kwargs={"split": "test"},
    ),
}


def make_transforms(img_size: int = 128):
    """Returns (train_tf, test_tf). Image stats are ImageNet's, matching
    the pretraining preprocessing in pretrain.py."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = v2.Compose([
        v2.RGB(),  # FGVC has some grayscale-ish images; normalize to 3 channels
        v2.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    test_tf = v2.Compose([
        v2.RGB(),
        v2.Resize(int(img_size * 1.14)),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    return train_tf, test_tf


class TwoLayerCLSExtractor:
    """Callable wrapper around a timm ViT that returns the concatenation
    of the CLS tokens from the last two transformer blocks.

    LeJEPA's published linear-probe protocol uses ``concat(CLS_-2,
    CLS_-1)``; this class implements that via forward hooks so we can
    keep the timm model otherwise unmodified.

    Shape:
        - input ``x``: ``(B, 3, H, W)``
        - output:      ``(B, 2 * embed_dim)`` -- e.g. ``(B, 768)`` for ViT-S
    """

    def __init__(self, backbone: nn.Module):
        self.backbone = backbone
        self._captures: list[torch.Tensor] = []
        # Hooks fire after each block's forward; output is the post-residual
        # (B, seq_len, embed_dim) tensor. Index 0 in the seq dim is CLS for
        # timm ViTs (cls_token prepended in `forward_features`).
        backbone.blocks[-2].register_forward_hook(self._hook)
        backbone.blocks[-1].register_forward_hook(self._hook)

    def _hook(self, module, _input, output):
        # CLS at index 0; detach not needed (backbone is frozen).
        self._captures.append(output[:, 0, :])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._captures.clear()
        # Run the backbone's feature path; we don't need the classifier head
        # output, only the side effect of the hooks firing.
        _ = self.backbone.forward_features(x)
        feats = torch.cat(self._captures, dim=-1)   # (B, 2 * embed_dim)
        self._captures.clear()
        return feats

    def eval(self):
        self.backbone.eval()
        return self


def load_backbone_from_ckpt(ckpt_path: str, device: str):
    """Reconstruct the ViT backbone from a pretrain.py checkpoint
    and wrap it in a ``TwoLayerCLSExtractor``.

    The checkpoint stores the full ``ViTEncoder`` state dict under the
    ``net`` key plus the pretrain config under ``config``. We instantiate
    a matching timm ViT, load the backbone subset of the saved state
    dict, freeze, attach the two-layer CLS extractor, and return.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state = ckpt["net"]
    # Strip DDP "module." prefix if present.
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}
    # Keep only the backbone keys.
    backbone_state = {k[len("backbone."):]: v
                      for k, v in state.items() if k.startswith("backbone.")}

    backbone = timm.create_model(
        "vit_small_patch8_224",
        pretrained=False,
        num_classes=512,
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
        img_size=128,
    )
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"[warn] missing backbone keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected backbone keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    extractor = TwoLayerCLSExtractor(backbone)
    emb_dim = 2 * backbone.embed_dim   # 768 for ViT-S/8
    return extractor, emb_dim, cfg


@torch.inference_mode()
def evaluate(extractor, probe, loader, device):
    extractor.eval(); probe.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16):
            emb = extractor(x)
            logits = probe(emb.float())
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Backbone (frozen) + 2-layer CLS extractor -------------------------
    extractor, emb_dim, pretrain_cfg = load_backbone_from_ckpt(cfg.ckpt, device)

    # --- Dataset -----------------------------------------------------------
    dname = cfg.dataset
    if dname not in DATASETS:
        raise ValueError(f"Unknown dataset {dname!r}. Choose from {list(DATASETS)}")
    spec = DATASETS[dname]
    img_size = int(cfg.get("img_size", 128))
    train_tf, test_tf = make_transforms(img_size)
    # data_root default: prefer $SCRATCH/.cache/torchvision if SCRATCH is
    # set (e.g. on Anvil), else $HOME/.cache/torchvision. Override with
    # +data_root=<path>. On Anvil, $HOME has very tight quota and will
    # fail mid-download for the larger datasets (Aircraft, Food, Cars).
    import os
    scratch = os.environ.get("SCRATCH")
    default_root = (Path(scratch) / ".cache" / "torchvision") if scratch \
                   else (Path.home() / ".cache" / "torchvision")
    data_root = cfg.get("data_root", str(default_root))

    train_ds = spec["cls"](root=data_root, transform=train_tf,
                            download=True, **spec["train_kwargs"])
    test_ds = spec["cls"](root=data_root, transform=test_tf,
                           download=True, **spec["test_kwargs"])

    bs = int(cfg.get("bs", 256))
    nw = int(cfg.get("num_workers", 8))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              drop_last=True, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=True)

    # --- Probe -------------------------------------------------------------
    probe = nn.Sequential(
        nn.LayerNorm(emb_dim),
        nn.Linear(emb_dim, spec["n_classes"]),
    ).to(device)

    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("wd", 1e-6))
    epochs = int(cfg.get("epochs", 50))

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    warmup_steps = max(1, len(train_loader))
    total_steps = warmup_steps * epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    eta_min = float(cfg.get("probe_eta_min", 1e-6))
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "pretrain_config": pretrain_cfg,
        "dataset": dname,
        "n_classes": spec["n_classes"],
        "seed": seed,
        "epochs": [],
        "start_time": time.time(),
    }

    best_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        loss_sum = n_batches = 0
        for x, y in tqdm.tqdm(train_loader, total=len(train_loader),
                              desc=f"ep{epoch} {dname}"):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    emb = extractor(x)
                logits = probe(emb.float())
                loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            loss_sum += loss.item()
            n_batches += 1

        acc = evaluate(extractor, probe, test_loader, device)
        best_acc = max(best_acc, acc)
        results["epochs"].append({
            "epoch": epoch,
            "train_loss": loss_sum / max(n_batches, 1),
            "test_acc": acc,
        })
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    results["final_acc"] = results["epochs"][-1]["test_acc"] if results["epochs"] else float("nan")
    results["best_acc"] = best_acc
    results["wall_seconds"] = time.time() - results["start_time"]
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
