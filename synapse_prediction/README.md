# Synful synapse-partner detection — train & predict

Trains and runs a 3D U-Net that predicts, per voxel, a **synaptic-site indicator**
and a **partner (post→pre) vector field** in EM volumes, following the
[synful](https://github.com/funkelab/synful) approach. This repo packages a
known-working version of the pipeline plus everything needed to recreate the
environment on another machine.

> **The trained model weights are not included.** You supply your own
> checkpoints (from training) or obtain them separately — see
> [Getting a model](#getting-a-model).

---

## What's here

```
environment.yml          conda env (Python 3.7 / TensorFlow 1.14 / CUDA 10.0)
install.sh               creates the env + installs the vendored forks
vendor/                  forked packages not available on PyPI:
    gunpowder/             custom GraphSource(graph_params=[db,host], ...)
    synful/                AddPartnerVectorMap, IntensityScaleShiftClip, ...
    funlib.persistence/    MongoDbGraphProvider
    funlib.learn.tensorflow/  U-Net model builder (one compiled .so, linux/py3.7)
pipeline/
    train_gb_flex.py       training entry point
    predict_scan.py        prediction entry point (single-process, portable)
    generate_network_flex.py  (re)builds net.meta + net_config.json
    net.meta               network graph definition (structure only, no weights)
    net_config.json        tensor names + input/output shapes
    checkpoints/           net.meta + net_config.json live here; weights go here too
full_params.example.json  example config for train + predict
```

## Requirements

- Linux x86-64, NVIDIA GPU with ~11 GB (developed on RTX 2080 Ti)
- NVIDIA driver supporting CUDA 10.0 (driver >= 410). `cudatoolkit`/`cudnn`
  themselves are installed by conda — no system CUDA needed.
- `conda` (miniconda/miniforge)
- **MongoDB** running locally — required for **training only** (synapse
  ground-truth graphs are read from Mongo). Prediction does **not** need Mongo.

## Install

```bash
./install.sh                # creates conda env "synful"
# or: ./install.sh myenvname
conda activate synful
```

`install.sh` creates the env from `environment.yml`, then `pip install`s the four
vendored forks (with `--no-deps`, since their dependencies come from the env),
and verifies the imports — in particular that the **forked** gunpowder
`GraphSource(graph_params, graph, graph_spec)` is the one on the path. The stock
PyPI gunpowder is **incompatible** and will fail with
`'list' object has no attribute 'directed'`.

## Getting a model

Weights are not distributed here. Options:

1. **Train your own** (below) — produces `checkpoints/net_checkpoint_<iter>.*`.
2. **Use existing checkpoints**: drop `net_checkpoint_<iter>.{index,meta,data-*}`
   into `pipeline/checkpoints/`. If they came from another machine, make sure
   `pipeline/checkpoints/checkpoint` (the TF state file) uses **relative**
   basenames, e.g.:
   ```
   model_checkpoint_path: "net_checkpoint_700000"
   all_model_checkpoint_paths: "net_checkpoint_700000"
   ```
   `predict_scan.py` builds the checkpoint path explicitly from `iteration`, so
   it works even without that state file; training uses it to decide whether to
   resume.

The network **graph** (`net.meta`) and tensor-name/shape config
(`net_config.json`) *are* included, so predict/train find the architecture even
before you have weights.

## Configure

Copy the example and edit it:

```bash
cp full_params.example.json full_params.json
```

Key blocks:
- `augments.GPU_id` — which GPU to use.
- `train_params` — hyperparameters; these define the shipped `net.meta`, so
  don't change shapes unless you also regenerate the network
  (`python generate_network_flex.py ...`). `working_directory: "."` keeps
  checkpoints/tensorboard/snapshot inside the run dir.
- `predict_params` — `raw_file` (input zarr), `raw_dataset`, `raw_offset`/
  `raw_size` (world units / nm; omit to predict the whole volume), `iteration`,
  `out_directory`, `out_filename`, `out_properties` (dtype/scale per output).

> **Training data wiring is currently hard-coded** near the top of
> `train_gb_flex.py`: `syn_db_names`, `in_rois`, and `data_dir_syns`. Edit those
> to point at your Mongo databases, ground-truth ROIs, and raw zarrs before
> training. (Prediction takes its input purely from `predict_params`.)

## Train

Run from inside `pipeline/` (so `./checkpoints/` and `./net_config.json`
resolve):

```bash
cd pipeline
python train_gb_flex.py ../full_params.json
```

- **Resume**: if `checkpoints/checkpoint` points at existing
  `net_checkpoint_*` files, training continues from the latest.
- **From scratch**: with no `checkpoints/checkpoint` state file present,
  training initializes randomly and starts at iteration 0. To keep an existing
  model, train into a separate directory (set `working_directory`) — a
  same-basename run will overwrite `net_checkpoint_*` as it saves.

Checkpoints are written every 100k iterations; snapshots and TensorBoard logs
go to `snapshot/` and `tensorboard/`.

## Predict

Single-process, no daisy/Mongo — portable and the recommended path:

```bash
cd pipeline
python predict_scan.py ../full_params.json
```

Reads the raw zarr, runs the network over the ROI in network-sized chunks
(`gunpowder.Scan`), and writes:

```
<out_directory>/<setup>/<out_filename>/volumes/pred_syn_indicator     (uint8, 1 ch)
<out_directory>/<setup>/<out_filename>/volumes/pred_partner_vectors   (int8/float32, 3 ch)
```

`pred_syn_indicator` is `sigmoid * 255` (so 128 ≈ p=0.5); `pred_partner_vectors`
stores post→pre offsets in nm scaled by the `scale` attribute on the dataset.

> A block-parallel `predict_blockwise.py` (daisy + Mongo, multi-GPU) also exists
> in the original pipeline but is **not** included as the default: the daisy
> scheduler/worker handshake is version-fragile across machines. `predict_scan.py`
> supersedes it for single-machine use.

## Sanity check

Over a region containing real synapses, `pred_syn_indicator` should span a wide
range (up to ~254) with a few percent of voxels above 128; over empty neuropil
it should be near zero. `pred_partner_vectors` is a dense field with magnitudes
on the order of the training `d_blob_radius` (~150 nm).

## Troubleshooting

Real snags hit while reproducing this env on a clean machine (all are handled by
`environment.yml`/`install.sh`, but here's what they look like if something drifts):

- **`Conv3DBackpropInputOpV2 only supports NDHWC on the CPU`** — the network is
  running on CPU. Two causes: (a) you didn't `conda activate` the env, so the
  conda CUDA libraries aren't found; (b) the CPU `tensorflow` package got
  installed alongside `tensorflow-gpu` and shadowed it. Fix: activate the env,
  and ensure **only** `tensorflow-gpu==1.14.0` is installed
  (`pip uninstall tensorflow`). Check with
  `python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"`.
- **`protobuf` / "generated code is out of date"** — TensorFlow 1.14 is
  incompatible with protobuf 4.x. The env pins `protobuf==3.20.3`; don't let
  something upgrade it.
- **`ModuleNotFoundError: neuroglancer` (or `cloudvolume`)** — these are only
  needed for the synapse-extraction / cloud-source stages, not train/predict.
  `synful.gunpowder` guards those imports, so this should not occur; if you use
  the extraction/eval stages, `pip install neuroglancer` separately.
- **`install.sh` installs into the wrong env** — if you run it with another
  conda env already active, avoid `conda run`; the script installs the vendored
  forks using the target env's own `python -m pip` by absolute path for exactly
  this reason.

> **comet_ml logging has been removed.** The upstream fork's training node
> created a comet experiment (with a hard-coded API key) on every run; that call
> was stripped from the vendored `gunpowder`. TensorBoard summaries are kept.
