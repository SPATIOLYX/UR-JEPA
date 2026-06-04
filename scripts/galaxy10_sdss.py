"""Galaxy10 SDSS loader for UR-JEPA pretraining.
Author: Spatiolyx LLC, https://spatiolyx.ai
Date: May 30, 2026


The original Galaxy10 SDSS dataset used by LeJEPA's Table 5 has no
HuggingFace mirror; it's distributed as a single ~200MB HDF5 file
from Zenodo. This module:
  - Downloads ``Galaxy10.h5`` once to a configurable cache dir
  - Verifies SHA256 against the published hash
  - Provides a deterministic 50/50 stratified train/test split
    (``random_state=42``) approximating LeJEPA's 11,008-train Table 5 cell
    (their split is unpublished; this is our best guess based on their
    reported train count)
  - Exposes a ``torch.utils.data.Dataset`` with the same
    ``{"image": PIL.Image, "label": int}`` schema as our HuggingFace
    pretrain datasets, so it slots into ``HFDataset`` and the rest of
    ``pretrain.py`` with no special-casing downstream

Reference: https://astronn.readthedocs.io/en/latest/galaxy10sdss.html

Cache layout:
    $SCRATCH/.cache/galaxy10/Galaxy10.h5    (default on Anvil)
    $HOME/.cache/galaxy10/Galaxy10.h5       (fallback elsewhere)

CLI smoke test:
    python scripts/galaxy10_sdss.py
"""

import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


GALAXY10_URL = "https://zenodo.org/records/10844811/files/Galaxy10.h5"
GALAXY10_SHA256 = "969a6b1cefcc36e09fffa86febd2f699a4aa19b837ba0427f01b0bc6ded458af"
GALAXY10_TOTAL = 21785  # for sanity-check assertions


def _default_cache_dir() -> Path:
    """Default cache: ``$SCRATCH/.cache/galaxy10`` if SCRATCH is set
    (Anvil convention), else ``$HOME/.cache/galaxy10``."""
    scratch = os.environ.get("SCRATCH")
    base = Path(scratch) if scratch else Path.home()
    return base / ".cache" / "galaxy10"


def _verify_sha256(path: Path) -> bool:
    """Return True if the file's SHA256 matches GALAXY10_SHA256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest() == GALAXY10_SHA256


def ensure_galaxy10_h5(cache_dir: Path | None = None) -> Path:
    """Download ``Galaxy10.h5`` to ``cache_dir`` if missing, verify
    SHA256, and return the file path. Raises ``RuntimeError`` on hash
    mismatch (delete the corrupted file manually and retry)."""
    cache_dir = cache_dir or _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "Galaxy10.h5"
    if not path.exists():
        print(f"[galaxy10] downloading {GALAXY10_URL} -> {path}", flush=True)
        urlretrieve(GALAXY10_URL, path)
    if not _verify_sha256(path):
        raise RuntimeError(
            f"[galaxy10] SHA256 mismatch on {path}\n"
            f"  expected: {GALAXY10_SHA256}\n"
            f"Delete the file and re-download."
        )
    return path


def _load_split(
    cache_dir: Path | None = None,
    split: str = "train",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (images, labels) for a 50/50 stratified split of Galaxy10 SDSS.

    Returns:
        images: ``(N, 69, 69, 3)`` uint8 array
        labels: ``(N,)`` int array, 0..9
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    h5_path = ensure_galaxy10_h5(cache_dir)
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]   # (21785, 69, 69, 3) uint8
        labels = f["ans"][:]      # (21785,) uint8
    assert len(labels) == GALAXY10_TOTAL, (
        f"unexpected dataset size {len(labels)}; expected {GALAXY10_TOTAL}. "
        f"H5 file may be from a different Galaxy10 release."
    )
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.5, random_state=seed, stratify=labels
    )
    use_idx = train_idx if split == "train" else test_idx
    return images[use_idx], labels[use_idx].astype(np.int64)


class Galaxy10SDSS(Dataset):
    """Galaxy10 SDSS dataset with HF-compatible schema.

    Returns dicts ``{"image": PIL.Image (RGB, 69x69), "label": int}``
    on ``__getitem__``, matching ``datasets.load_dataset(...)`` output.

    Args:
        split: ``'train'`` (10,892 images) or ``'test'`` (10,893 images).
            Deterministic 50/50 stratified split; matches LeJEPA's
            Table 5 11,008-train cell approximately (their exact split
            is unpublished).
        cache_dir: directory holding ``Galaxy10.h5``. Defaults to
            ``$SCRATCH/.cache/galaxy10`` on Anvil, else
            ``$HOME/.cache/galaxy10``.
        seed: ``random_state`` for the stratified split (default 42).
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Path | None = None,
        seed: int = 42,
    ):
        self.split = split
        self.images, self.labels = _load_split(cache_dir, split, seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> dict:
        img = Image.fromarray(self.images[i])  # 69x69x3 uint8 -> PIL RGB
        return {"image": img, "label": int(self.labels[i])}


def _smoke_test():
    """CLI smoke test: download (if needed), load both splits, print shapes."""
    print("[smoke] cache dir:", _default_cache_dir())
    for split in ("train", "test"):
        ds = Galaxy10SDSS(split=split)
        sample = ds[0]
        print(
            f"[smoke] split={split}: N={len(ds)}, "
            f"image={sample['image'].size} mode={sample['image'].mode}, "
            f"label={sample['label']}, "
            f"label range=[{min(ds.labels)}, {max(ds.labels)}]"
        )
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
