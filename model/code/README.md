# Computational modelling

This directory contains code to extrapolate from the synaptic resolution ROI to a full connectivity matrix of the head direction circuit,
as well as a computational model of a rate-based neural network with connectivity corresponding to the extrapolated connectivity matrix.


## Running

Dependencies and environment setup is handled by [uv](https://docs.astral.sh/uv/);
see `pyproject.toml` for dependency details.
The directory is set up as a CodeOcean capsule. To install dependencies and
perform a reproducible run locally, use

    uv run bash run.sh

Output artifacts will be placed in `../results`.

Alternatively, create a virtual environment using

    uv venv

to run the notebooks manually.

Tested on Ubuntu 24.04.2 LTS.


## Extrapolation

Extrapolation from the synaptic resolution ROI is done by modelling the neurons' input and output
regions as 'arborization kernels', i.e. the density of synapses as a function of the columnar distance from the
center of a cell type's projection. The connection strength between two cell types is then
computed based on the overlap of the cell types' arborization kernels. These arborization kernels are
constructed by least-squares optimization where the residuals are based on how closely they reconstruct
the ROI data. See notebooks `extrapolate-bee.ipynb` and `extrapolate-fly.ipynb` (for validation).
For a more detailed explanation, see the methods section of the manuscript.


## Parameter tuning
Based on the connectivity matrices, simple rate-based models are constructed.
The effective weights contributions of an excitatory or inhibitory synapse are treated as
free parameters. To explore this parameter space, a grid search is performed by `tune.py`.
The best parameter choice for each model is taken to be where the maximum drift rate
along the approximate ring attractor manifold is minimal, under the constraint that
unilateral PEN inhibition results in constant clockwise or counterclockwise rotation.

To increase the resolution of the grid search, edit the `RESOLUTION` value in `tune.py`. The value used for the manuscript was 100 samples along each axis.


## Figures

The simulation experiments on the tuned model are found in `experiments.py`,
and plots used for the manuscript are written to `../results/experiments`.


## License

Licensed under the [MIT license](LICENSE.md).
