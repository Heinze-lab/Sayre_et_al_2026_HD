#!/usr/bin/env bash
set -ex

mkdir -p ../results/data/connectome
mkdir -p ../results/data/tuning
mkdir -p ../results/data/experiments
mkdir -p ../results/experiments
mkdir -p ../data/workspace

# Extrapolate fly connectome
if [[ ! -e ../results/extrapolate-fly.ipynb ]]; then
    jupyter nbconvert \
    --ExecutePreprocessor.allow_errors=True \
    --ExecutePreprocessor.timeout=-1 \
    --FilesWriter.build_directory=../results \
    --to notebook \
    --execute extrapolate-fly.ipynb
fi

# Extrapolate bee connectome
if [[ ! -e ../results/extrapolate-bee.ipynb ]]; then
    jupyter nbconvert \
    --ExecutePreprocessor.allow_errors=True \
    --ExecutePreprocessor.timeout=-1 \
    --FilesWriter.build_directory=../results \
    --to notebook \
    --execute extrapolate-bee.ipynb
fi

# Extrapolation figures
if [[ ! -e ../results/extrapolation-figures.ipynb ]]; then
    jupyter nbconvert \
    --ExecutePreprocessor.allow_errors=True \
    --ExecutePreprocessor.timeout=-1 \
    --FilesWriter.build_directory=../results \
    --to notebook \
    --execute extrapolation-figures.ipynb
fi
 
# Run parameter search
python3 tune.py

# Run experiments
python3 experiments.py