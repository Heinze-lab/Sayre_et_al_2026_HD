#!/usr/bin/env bash
# Download the training raw volume (NO_cube1.zarr) from the GitHub release and
# extract it into this data/ directory. The raw volume is ~185 MB and is
# distributed as a release asset (too large to commit to git).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SYN_REPO:-Heinze-lab/Sayre_et_al_2026_HD}"
TAG="${SYN_DATA_TAG:-synapse-data-NO_cube1}"

echo ">> Downloading NO_cube1.zarr.tar.gz from $REPO ($TAG) ..."
gh release download "$TAG" --repo "$REPO" -p 'NO_cube1.zarr.tar.gz' -D "$HERE" --clobber

echo ">> Extracting ..."
tar -xzf "$HERE/NO_cube1.zarr.tar.gz" -C "$HERE"
rm -f "$HERE/NO_cube1.zarr.tar.gz"
echo "Done -> $HERE/NO_cube1.zarr"
