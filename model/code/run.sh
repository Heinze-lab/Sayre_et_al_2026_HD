#!/usr/bin/env bash
set -ex

mkdir -p ../results/data/connectome
mkdir -p ../results/data/tuning
mkdir -p ../results/data/experiments
mkdir -p ../results/experiments

# Extrapolate fly connectome
if [[ ! -e ../results/extrapolate-fly.ipynb ]]; then
    uv run jupyter nbconvert \
    --ExecutePreprocessor.allow_errors=True \
    --ExecutePreprocessor.timeout=-1 \
    --FilesWriter.build_directory=../results \
    --to notebook \
    --execute extrapolate-fly.ipynb
fi

# Extrapolate bee connectome
if [[ ! -e ../results/extrapolate-bee.ipynb ]]; then
    uv run jupyter nbconvert \
    --ExecutePreprocessor.allow_errors=True \
    --ExecutePreprocessor.timeout=-1 \
    --FilesWriter.build_directory=../results \
    --to notebook \
    --execute extrapolate-bee.ipynb
fi
  
# Run parameter search
uv run tune.py

# Run experiments
uv run experiments.py