# Deep conservation of head direction circuits

This repository contains the analysis code for Sayre et al. (2026), **“Deep
conservation of head direction circuits in bees, ants and flies.”** It is laid
out as a [Code Ocean](https://codeocean.com/) capsule and can also be run with
Docker or a local Python installation.


| Path | Contents |
| --- | --- |
| `code/analysis/` | Notebooks used for the paper's analyses and figures |
| `code/syntables/` | Notebooks for importing bee and fly synapse tables |
| `code/utils/` | Data conversion and visualization utilities |
| `environment/` | Python dependencies and container |
| `results/` | Output written by the capsule run script |

Synapse tables are kept as capsule data rather than in
Git; see [`code/syntables/README.md`](code/syntables/README.md)


### Python virtual environment

```bash
python3.8 -m venv .hd_venv
source .hd_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
python -m ipykernel install --user --name sayre-hd --display-name "Sayre HD"
./code/run
```

## Credentials and external services

Some data-preparation and utility notebooks access CAVE, CATMAID, neuPrint,
Google Sheets, or SeaTable to import data. These are **not needed by the environment
t** and analysis from the available / archived tables does not require those services.

## License

The code is released under the [MIT License](LICENSE). Data obtained from an
external service remains subject to that service's terms and the source data's
license.
