from __future__ import print_function

"""Single-process, daisy-free synapse prediction.

Runs the trained network over a raw volume using gunpowder's Scan node instead
of the daisy block-wise scheduler. No MongoDB, no worker subprocesses -- just
reads a raw zarr, predicts, and writes a prediction zarr. This is the portable
"runs anywhere on one GPU" path.

Usage:
    python predict_scan.py full_params.json

Reads the same `predict_params` / `train_params` blocks as predict_blockwise.py,
so it is a drop-in for the single-machine case.
"""

import json
import logging
import os
import sys

import numpy as np
import gunpowder as gp
import daisy

from synful.gunpowder import IntensityScaleShiftClip
import tensorflow as tf

logger = logging.getLogger(__name__)


def predict_scan(predict_params, train_params):
    iteration = predict_params['iteration']
    raw_file = os.path.abspath(predict_params['raw_file'])
    raw_dataset = predict_params['raw_dataset']
    raw_offset = predict_params.get('raw_offset', None)
    raw_size = predict_params.get('raw_size', None)
    configname = predict_params.get('configname', 'net')
    out_properties = predict_params.get('out_properties', {})

    out_directory = predict_params['out_directory']
    out_filename = predict_params['out_filename']
    setup = predict_params.get('setup', 'net')
    out_file = os.path.abspath(os.path.join(out_directory, setup, out_filename))

    setup_dir = os.path.abspath('./checkpoints')

    with open(os.path.join(setup_dir, '{}_config.json'.format(configname))) as f:
        net_config = json.load(f)

    # Determine total ROI from the raw source (world units / nm).
    source = daisy.open_ds(raw_file, raw_dataset)
    voxel_size = gp.Coordinate(source.voxel_size)
    if raw_offset is not None and raw_size is not None:
        roi = daisy.Roi(tuple(raw_offset), tuple(raw_size)).snap_to_grid(
            source.voxel_size)
        source = source[roi]
    output_roi = gp.Roi(tuple(source.roi.get_begin()),
                        tuple(source.roi.get_shape()))

    input_shape = gp.Coordinate(net_config['input_shape'])
    output_shape = gp.Coordinate(net_config['output_shape'])
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    context = (input_size - output_size) / 2
    input_roi = output_roi.grow(context, context)

    print('Raw file        : %s' % raw_file)
    print('Output file     : %s' % out_file)
    print('Voxel size      : %s' % (voxel_size,))
    print('Total output ROI: %s' % (output_roi,))
    print('Total input  ROI: %s' % (input_roi,))
    print('Block in/out    : %s / %s' % (input_size, output_size))

    raw = gp.ArrayKey('RAW')
    pred_postpre_vectors = gp.ArrayKey('PRED_POSTPRE_VECTORS')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')

    d_property = out_properties.get('pred_partner_vectors', None)
    m_property = out_properties.get('pred_syn_indicator_out', None)

    # Pre-create the output datasets at the correct ROI / dtype / channels so
    # ZarrWrite writes into an existing, correctly-sized array.
    out_specs = {
        pred_post_indicator: (
            'volumes/pred_syn_indicator',
            (m_property or {}).get('dtype', 'uint8'),
            net_config['outputs']['pred_syn_indicator_out']['out_dims'],
        ),
        pred_postpre_vectors: (
            'volumes/pred_partner_vectors',
            (d_property or {}).get('dtype', 'float32'),
            net_config['outputs']['pred_partner_vectors']['out_dims'],
        ),
    }
    for key, (ds_name, dtype, out_dims) in out_specs.items():
        ds = daisy.prepare_ds(
            out_file, ds_name, daisy.Roi(tuple(output_roi.get_begin()),
                                        tuple(output_roi.get_shape())),
            source.voxel_size, dtype,
            write_size=tuple(output_size), num_channels=out_dims,
            compressor={'id': 'gzip', 'level': 5})
        scale = (m_property if key == pred_post_indicator else d_property)
        if scale is not None and 'scale' in scale:
            ds.data.attrs['scale'] = scale['scale']

    # Prediction pipeline (identical network I/O to predict.py).
    pipeline = gp.ZarrSource(
        raw_file,
        datasets={raw: raw_dataset},
        array_specs={raw: gp.ArraySpec(interpolatable=True)})
    pipeline += gp.Pad(raw, size=None)
    pipeline += gp.Normalize(raw)
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += gp.tensorflow.Predict(
        os.path.join(setup_dir, train_params['config_name'] +
                    '_checkpoint_%d' % iteration),
        inputs={net_config['raw']: raw},
        outputs={
            net_config['pred_syn_indicator_out']: pred_post_indicator,
            net_config['pred_partner_vectors']: pred_postpre_vectors,
        },
        graph=os.path.join(setup_dir, '{}.meta'.format(configname)))

    # Post-processing: same order/semantics as predict.py.
    d_scale = train_params.get('d_scale', None)
    if d_scale is not None and d_scale != 1:
        pipeline += gp.IntensityScaleShift(pred_postpre_vectors, 1. / d_scale, 0)
    if m_property is not None and m_property.get('scale', 1) != 1:
        pipeline += gp.IntensityScaleShift(
            pred_post_indicator, m_property['scale'], 0)
    if d_property is not None and 'scale' in d_property:
        pipeline += gp.IntensityScaleShift(
            pred_postpre_vectors, d_property['scale'], 0)
    if d_property is not None and d_property.get('dtype') == 'int8':
        pipeline += IntensityScaleShiftClip(
            pred_postpre_vectors, 1, 0, clip=(-128, 127))

    pipeline += gp.ZarrWrite(
        dataset_names={
            pred_post_indicator: 'volumes/pred_syn_indicator',
            pred_postpre_vectors: 'volumes/pred_partner_vectors',
        },
        output_filename=out_file)

    # Scan over the whole ROI in network-sized chunks.
    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred_post_indicator, output_size)
    scan_request.add(pred_postpre_vectors, output_size)
    pipeline += gp.Scan(scan_request)

    full_request = gp.BatchRequest()
    full_request[raw] = gp.ArraySpec(roi=input_roi)
    full_request[pred_post_indicator] = gp.ArraySpec(roi=output_roi)
    full_request[pred_postpre_vectors] = gp.ArraySpec(roi=output_roi)

    print("Starting prediction (Scan)...")
    with gp.build(pipeline):
        pipeline.request_batch(full_request)
    print("Prediction finished -> %s" % out_file)
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open(sys.argv[1], 'r') as f:
        params = json.load(f)

    # Select the GPU here; let gunpowder's Predict node own the TF session
    # (creating one here would leave the graph on CPU, where the NCHW
    # transposed-convs are unsupported).
    gpu_id = str(params['augments']['GPU_id'])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    predict_scan(params['predict_params'], params['train_params'])
