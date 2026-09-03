# Training data — NO_cube1

One ground-truth cube is provided so you can train the pipeline end to end.

| Piece | Where it lives | Size |
|---|---|---|
| Raw EM volume `NO_cube1.zarr` (`raw`, uint8, 200×1000×1000, 50×10×10 nm) | GitHub **release** asset `NO_cube1.zarr.tar.gz` | ~185 MB |
| Synapse ground-truth graph (570 nodes / 285 pre→post edges) | `ground_truth/synapses_megalopta_NO1_cube_gb.archive.gz` (in this repo) | 11 KB |

The raw volume is too large to commit, so it's a release asset; the GT graph is
tiny and lives in the repo.

## Get the data

```bash
# 1. Raw volume  -> data/NO_cube1.zarr   (needs the gh CLI, authenticated)
bash download_data.sh

# 2. GT graph    -> local MongoDB db 'synapses_megalopta_NO1_cube_gb'
#    (needs mongorestore and a running mongod on localhost:27017)
bash restore_gt.sh
```

Overrides (env vars): `SYN_REPO`, `SYN_DATA_TAG` for the download; `SYN_DB_HOST`
for the restore.

## How training uses it

`pipeline/train_gb_flex.py` points at this cube:

```python
data_dir_base = os.environ.get('SYN_DATA_DIR', '<repo>/data')
in_rois       = [gp.Roi([207300, 126989, 256580], [5000]*3)]   # world nm
data_dir_syns = [<data_dir>/NO_cube1.zarr]
syn_db_names  = ['synapses_megalopta_NO1_cube_gb']
```

- `data_dir_syns[i]` — raw zarr for cube *i*
- `syn_db_names[i]`  — MongoDB database with that cube's GT graph
- `in_rois[i]`       — the ROI (world units / nm) the GT covers

To train on additional cubes, append matching entries to all three lists (and
provide each cube's raw zarr + restored GT database).
