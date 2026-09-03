from __future__ import absolute_import
# Core nodes needed for training/prediction.
from .add_partner_vector_map import AddPartnerVectorMap
from .intensity_scale_shift_clip import IntensityScaleShiftClip
from .upsample import UpSample

# Optional nodes: each pulls extra dependencies (neuroglancer, cloudvolume, ...)
# only needed for the synapse-extraction / cloud-source stages. Guard them so
# `import synful.gunpowder` works with just the core train/predict deps.
try:
    from .hdf5_points_source import Hdf5PointsSource
except ImportError:
    Hdf5PointsSource = None
try:
    from .extract_synapses import ExtractSynapses
except ImportError:
    ExtractSynapses = None
try:
    from .cloud_volume_source import CloudVolumeSource
except ImportError:
    CloudVolumeSource = None
