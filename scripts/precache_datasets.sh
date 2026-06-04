#!/bin/bash
# Pre-cache pretraining datasets used by pretrain.py.
# Run ONCE on the login node before submitting the corresponding
# sbatches, so concurrent training tasks don't all try to download
# the dataset at the same time (causes ratelimit + concurrent-write
# corruption on the cache).
#
# Datasets cached:
#   * frgfm/imagenette        (1.3GB)   -- existing Imagenette-10 runs
#   * clane9/imagenet-100     (8.4GB)   -- run_inet100_*.sbatch
#   * Galaxy10 SDSS           (200MB)   -- Table 5 in-domain (Zenodo HDF5)
#   * Flowers102              (350MB)   -- Table 5 in-domain (torchvision)
#   * CIFAR-10                (170MB)   -- Table 5 in-domain (torchvision)
#   * CIFAR-100               (170MB)   -- Table 5 in-domain (torchvision)
#   * Food-101                ( ~5GB)   -- Table 5 in-domain (torchvision; LARGE)
#
# Usage on Anvil:
#   bash scripts/precache_datasets.sh
#
# Cache locations:
#   HuggingFace:  $HF_HOME (default $SCRATCH/.cache/huggingface)
#   Galaxy10:     $SCRATCH/.cache/galaxy10/Galaxy10.h5
#   torchvision:  $SCRATCH/.cache/torchvision/

set -euo pipefail

export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "HF_HOME=$HF_HOME"
echo "--- caching Imagenette-10 ---"
python -c "
from datasets import load_dataset
for split in ('train', 'validation'):
    ds = load_dataset('frgfm/imagenette', '160px', split=split)
    print(f'  imagenette {split}: {len(ds)} images')
"

echo "--- caching ImageNet-100 (clane9/imagenet-100, ~8.4GB) ---"
python -c "
from datasets import load_dataset
for split in ('train', 'validation'):
    ds = load_dataset('clane9/imagenet-100', split=split)
    print(f'  imagenet-100 {split}: {len(ds)} images')
"

echo "--- caching Galaxy10 SDSS (Zenodo, ~200MB) ---"
# Use the loader's ensure_galaxy10_h5() so the SHA256 check runs here
# rather than at the start of every training task. Cache dir defaults
# to $SCRATCH/.cache/galaxy10 when SCRATCH is set.
python -c "
import sys
sys.path.insert(0, 'scripts')
from galaxy10_sdss import ensure_galaxy10_h5, Galaxy10SDSS
path = ensure_galaxy10_h5()
print(f'  galaxy10 SDSS HDF5: {path}')
for split in ('train', 'test'):
    ds = Galaxy10SDSS(split=split)
    print(f'  galaxy10 {split}: {len(ds)} images')
"

echo "--- caching torchvision pretrain datasets ---"
# All torchvision datasets used by pretrain.py's pretrain runs.
# Cached under $SCRATCH/.cache/torchvision (matches what
# _load_torchvision expects). Sizes (after download):
#   Flowers102: ~350MB
#   CIFAR-10:   ~170MB
#   CIFAR-100:  ~170MB
#   Food-101:   ~5GB  <- LARGEST; ~10-20 min on a fast connection
python -c "
import os
from pathlib import Path
from torchvision.datasets import Flowers102, CIFAR10, CIFAR100, Food101
scratch = os.environ.get('SCRATCH')
root = str((Path(scratch) if scratch else Path.home()) / '.cache' / 'torchvision')
Path(root).mkdir(parents=True, exist_ok=True)

print('  Flowers102 ...')
for split in ('train', 'val', 'test'):
    ds = Flowers102(root=root, split=split, download=True)
    print(f'    flowers102 {split}: {len(ds)} images')

print('  CIFAR-10 ...')
for train in (True, False):
    ds = CIFAR10(root=root, train=train, download=True)
    print(f'    cifar10 train={train}: {len(ds)} images')

print('  CIFAR-100 ...')
for train in (True, False):
    ds = CIFAR100(root=root, train=train, download=True)
    print(f'    cifar100 train={train}: {len(ds)} images')

print('  Food-101 (~5GB, this can take 10-20 min) ...')
for split in ('train', 'test'):
    ds = Food101(root=root, split=split, download=True)
    print(f'    food101 {split}: {len(ds)} images')
"

echo "OK -- datasets cached"
