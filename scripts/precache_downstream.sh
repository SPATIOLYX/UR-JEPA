#!/bin/bash
# Pre-cache the torchvision datasets used by run_downstream_*.sbatch
# to $SCRATCH/.cache/torchvision. Run ONCE on the login node before
# submitting any Phase A sbatch.
#
# Why: $HOME on Anvil has a tight quota (typically 25GB) and large
# datasets like FGVC-Aircraft (~3GB after extraction) will fail
# mid-download with cryptic FileNotFoundError messages on missing
# image files. $SCRATCH has TB of room.
#
# Also: pre-caching here avoids the race where 5 concurrent sbatch
# tasks all try to download the same dataset simultaneously and
# corrupt each other's cache.
#
# Datasets cached:
#   aircraft  -- FGVC-Aircraft       (~3GB extracted)
#   cifar100  -- CIFAR-100           (~170MB)
#   dtd       -- Describable Textures (~600MB)
#
# Usage on Anvil:
#   bash scripts/precache_downstream.sh
#
# If recovering from a $HOME quota failure:
#   rm -rf $HOME/.cache/torchvision
#   bash scripts/precache_downstream.sh

set -euo pipefail

DATA_ROOT="${SCRATCH:-$HOME}/.cache/torchvision"
mkdir -p "$DATA_ROOT"

echo "Caching downstream datasets to: $DATA_ROOT"
echo "(disk usage so far: $(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1))"
echo

python <<PYEOF
from torchvision import datasets as tvds
import os, sys
root = "$DATA_ROOT"

# Each dataset's train + test split, downloaded if missing.
print("--- FGVC-Aircraft (trainval + test, ~3GB) ---", flush=True)
tvds.FGVCAircraft(root=root, split="trainval", annotation_level="variant", download=True)
tvds.FGVCAircraft(root=root, split="test",     annotation_level="variant", download=True)

print("--- CIFAR-100 (train + test, ~170MB) ---", flush=True)
tvds.CIFAR100(root=root, train=True,  download=True)
tvds.CIFAR100(root=root, train=False, download=True)

print("--- DTD (train + test, ~600MB) ---", flush=True)
tvds.DTD(root=root, split="train", download=True)
tvds.DTD(root=root, split="test",  download=True)
PYEOF

echo
echo "OK -- downstream datasets cached. Total size:"
du -sh "$DATA_ROOT" 2>/dev/null || true
