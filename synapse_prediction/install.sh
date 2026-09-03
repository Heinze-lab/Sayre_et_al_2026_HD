#!/usr/bin/env bash
# Create the conda env and install the vendored (forked) packages.
# Usage: ./install.sh [env_name]   (default env name: synful)
set -euo pipefail

ENV_NAME="${1:-synful}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Creating conda env '$ENV_NAME' from environment.yml ..."
conda env create -n "$ENV_NAME" -f "$HERE/environment.yml"

# Resolve the new env's own python/pip by absolute path. Do NOT use
# `conda run` / bare pip here: if another env is currently active, those can
# leak and install into the wrong environment.
ENV_PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
ENV_PY="$ENV_PREFIX/bin/python"
if [ ! -x "$ENV_PY" ]; then
    echo "ERROR: could not locate python for env '$ENV_NAME' at $ENV_PY" >&2
    exit 1
fi
echo ">> Target env python: $ENV_PY"

echo ">> Installing vendored forks (no-deps; deps already satisfied above) ..."
# Order matters: gunpowder first, then funlib.* , then synful.
"$ENV_PY" -m pip install --no-deps --no-build-isolation \
    "$HERE/vendor/gunpowder" \
    "$HERE/vendor/funlib.persistence" \
    "$HERE/vendor/funlib.learn.tensorflow" \
    "$HERE/vendor/synful"

echo ">> Verifying imports ..."
"$ENV_PY" -c "
import gunpowder, gunpowder.nodes as gn, inspect
import synful.gunpowder
from funlib.persistence import graphs
import funlib.learn.tensorflow
sig = str(inspect.signature(gn.GraphSource.__init__))
assert 'graph_params' in sig, 'wrong gunpowder GraphSource: '+sig
print('OK — gunpowder(fork), synful, funlib.persistence, funlib.learn.tensorflow import cleanly')
"

echo ""
echo "Done. Activate with:  conda activate $ENV_NAME"
