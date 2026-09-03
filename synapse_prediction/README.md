# Synful synapse-partner detection — training

Trains a 3D U-Net that predicts, per voxel, a **synaptic-site indicator** and a
**partner (post→pre) vector field** in EM volumes, following the
[synful](https://github.com/funkelab/synful) approach. This folder packages a
known-working version of the training pipeline plus everything needed to
recreate the environment and train from the provided ground-truth cube.

## What's here

```
environment.yml          conda env (Python 3.7 / TensorFlow 1.14 / CUDA 10.0)
install.sh               creates the env + installs the vendored forks
vendor/                  forked packages not on PyPI:
    gunpowder/             custom GraphSource(graph_params=[db,host], ...)
    synful/                AddPartnerVectorMap, IntensityScaleShiftClip, ...
    funlib.persistence/    MongoDbGraphProvider
    funlib.learn.tensorflow/  U-Net model builder (one compiled .so, linux/py3.7)
pipeline/
    train_gb_flex.py       training entry point
    generate_network_flex.py  (re)builds net.meta + net_config.json
    synapse_db_to_json.py  export a synapse MongoDB (syn.nodes/syn.edges) to JSON
    net.meta               network graph definition (structure only, no weights)
    net_config.json        tensor names + input/output shapes
    checkpoints/           net.meta + net_config.json (drop trained weights here)
data/
    ground_truth/          synapse GT graph dump (restored into MongoDB)
    download_data.sh       fetch the raw training volume from the release
    restore_gt.sh          load the GT graph into local MongoDB
full_params.example.json  example training config
```

The trained model and the raw training volume are distributed as **GitHub
release assets** (too large to commit); the tiny GT graph is committed under
`data/`.

## Requirements

- Linux x86-64, NVIDIA GPU with ~11 GB (developed on RTX 2080 Ti)
- NVIDIA driver supporting CUDA 10.0 (driver ≥ 410); `cudatoolkit`/`cudnn` come
  from conda, so no system CUDA install is needed.
- `conda` (miniconda/miniforge), the `gh` CLI (for release downloads)
- **MongoDB** running locally — training reads synapse ground-truth graphs from it.

## Install

```bash
./install.sh            # creates conda env "synful" and installs the vendored forks
conda activate synful
```

`install.sh` builds the env from `environment.yml`, `pip install`s the four
vendored forks with the target env's own pip, and verifies the imports — in
particular that the **forked** gunpowder `GraphSource(graph_params, graph,
graph_spec)` is on the path. The stock PyPI gunpowder is **incompatible** and
fails with `'list' object has no attribute 'directed'`.

## Get the trained model (optional)

The trained checkpoint (700k iterations) is a release asset. Download it if you
want to resume training from it or run it with your own inference tooling:

```bash
cd pipeline/checkpoints
gh release download synapse-model-700k --repo Heinze-lab/Sayre_et_al_2026_HD -p '*.tar.gz'
tar --strip-components=1 -xzf net_checkpoint_700000.tar.gz && rm net_checkpoint_700000.tar.gz
```

This drops `net_checkpoint_700000.*` + `net.meta` + `net_config.json` +
`checkpoint` into `pipeline/checkpoints/`. The network **graph** (`net.meta`) and
tensor-name/shape config (`net_config.json`) are already included, so training
finds the architecture even before you have weights.

## Get the training data

One ground-truth cube (`NO_cube1`) is provided (see [`data/README.md`](data/README.md)):

```bash
cd data
bash download_data.sh    # raw volume NO_cube1.zarr  (release asset, ~185 MB)
bash restore_gt.sh       # synapse GT graph -> local MongoDB
cd ..
```

`train_gb_flex.py` points at this cube via `data_dir_syns` / `syn_db_names` /
`in_rois` near the top of the file (data dir overridable with `SYN_DATA_DIR`).
Append matching entries to those three lists to train on more cubes.

## Configure

```bash
cp full_params.example.json full_params.json
```

- `augments.GPU_id` — which GPU to use.
- `train_params` — hyperparameters; these define the shipped `net.meta`, so don't
  change shapes unless you also regenerate the network
  (`python generate_network_flex.py ...`). `working_directory: "."` keeps
  `checkpoints/`, `tensorboard/`, `snapshot/` inside the run dir.

## Train

Needs `mongod` (localhost:27017) with the GT restored and the raw cube downloaded.
Run from inside `pipeline/` (so `./checkpoints/` and `./net_config.json` resolve):

```bash
cd pipeline
python train_gb_flex.py ../full_params.json
```

- **Resume**: if `checkpoints/checkpoint` points at existing `net_checkpoint_*`
  files, training continues from the latest.
- **From scratch**: with no `checkpoints/checkpoint` state file present, training
  initializes randomly and starts at iteration 0. To keep an existing model,
  train into a separate directory (`working_directory`) — a same-basename run
  overwrites `net_checkpoint_*` as it saves.

Checkpoints are written every 100k iterations; snapshots and TensorBoard logs go
to `snapshot/` and `tensorboard/`.

## Export a synapse database to JSON

`synapse_db_to_json.py` exports synapses from a MongoDB (collections
`syn.nodes` / `syn.edges`) to a JSON file (e.g. for viewing in neuroglancer):

```bash
python synapse_db_to_json.py to_json_config.json   # see to_json_config.example.json
```

## Troubleshooting

- **`Conv3DBackpropInputOpV2 only supports NDHWC on the CPU`** — the network is
  running on CPU. Either you didn't `conda activate` the env (so the conda CUDA
  libraries aren't found), or the CPU `tensorflow` package got installed
  alongside `tensorflow-gpu` and shadowed it. Fix: activate the env and keep
  **only** `tensorflow-gpu==1.14.0` (`pip uninstall tensorflow`). Verify with
  `python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"`.
- **`protobuf` / "generated code is out of date"** — TensorFlow 1.14 needs
  protobuf 3.x; the env pins `protobuf==3.20.3`.
- **`ModuleNotFoundError: neuroglancer` / `cloudvolume`** — only needed for the
  synapse-extraction / cloud-source stages; `synful.gunpowder` guards those
  imports, so training is unaffected.

> **comet_ml logging removed.** The upstream fork's training node created a comet
> experiment (with a hard-coded API key) on every run; that call was stripped
> from the vendored `gunpowder`. TensorBoard summaries are kept.
