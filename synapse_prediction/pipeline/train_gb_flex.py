from __future__ import print_function
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import subprocess

import json
import math
import os
import pdb
import sys
import logging

try:
    import absl.logging

    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

import gunpowder as gp
import gunpowder.nodes as gn
import numpy as np
import daisy
from generate_network_flex import mknet
from synful.gunpowder import AddPartnerVectorMap
from funlib.persistence import graphs as grs
import tensorflow as tf

# CREMI specific, download data from: www.cremi.org
#data_dir = '../../../../../data/cremi/'
#data_dir_syn = data_dir
#samples = [
#    'sample_A_padded_20160501',
#    'sample_B_padded_20160501',
#    'sample_C_padded_20160501'
#]
#cremi_roi = gp.Roi(np.array((1520, 3644, 3644)), np.array((5000, 5000, 5000)))

#in_roi = gp.Roi([207300,126980,256570], [5000,5000,5000])
#data_dir_syn = '/home/griffin/github/synapse-detection/make_train/example_in_and_out/test_n.h5'

syn_db_host = 'mongodb://localhost:27017'

# --- Training data ---------------------------------------------------------
# One ground-truth cube ships with this repo (see ../data/README.md):
#   * raw volume  NO_cube1.zarr  -> download via data/download_data.sh
#   * synapse GT graph in MongoDB -> restore via data/restore_gt.sh
# Each index below pairs a raw zarr, the MongoDB database holding its GT graph,
# and the ROI (world units / nm) that the GT covers. To train on more cubes,
# append matching entries to all three lists.
data_dir_base = os.environ.get(
    'SYN_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))

in_rois       = [gp.Roi([207300, 126989, 256580], [5000] * 3)]
data_dir_syns = [os.path.join(data_dir_base, 'NO_cube1.zarr')]
syn_db_names  = ['synapses_megalopta_NO1_cube_gb']

def create_source(sample, raw, graphsyn, dummypostsyn, parameter, 
                  data_dir_syn, syn_db_name, in_roi):
                  #, dummypostsyn, gt_neurons):

    gr_prov = grs.MongoDbGraphProvider(
                    syn_db_name,
                    host = syn_db_host,
                    mode = 'r+',
                    directed = True,
                    edges_collection = 'edges_pre_post_syn',
                    position_attribute = ['center_z', 'center_y', 'center_x'])
    # NOTE: the custom (forked) gunpowder GraphSource used by this project takes
    # [db_name, host] and constructs the MongoDbGraphProvider internally.
    gr_prov = [syn_db_name, syn_db_host]
    data_sources = tuple(
        (gn.GraphSource(
                gr_prov,
		graph=graphsyn,
                graph_spec = gp.GraphSpec(roi=in_roi, directed=True)
            ),
            gn.GraphSource(
                gr_prov,
                graph=dummypostsyn,
                graph_spec = gp.GraphSpec(roi=in_roi, directed=True)
            ),
            gn.ZarrSource(
                data_dir_syn,
                datasets={
                    raw: 'raw'},
		array_specs={
                    raw: gp.ArraySpec(interpolatable=True)}
		)
        )
    )
#    print([x.spec for x in data_sources])
    source_pip = data_sources + gp.MergeProvider() + gp.RandomLocation(ensure_nonempty=dummypostsyn,
                                 p_nonempty=parameter['reject_probability'])
    return source_pip


def build_pipeline(parameter, augment=True):
    augments = parameter['augments']
    parameter = parameter['train_params']

    # Resolve the working directory (holds checkpoints/, tensorboard/, snapshot/).
    # The configured path may be a stale absolute path from another machine; fall
    # back to the current directory so the setup is portable across machines.
    working_directory = parameter.get('working_directory') or '.'
    if not os.path.isdir(working_directory):
        logging.warning(
            "configured working_directory %r not found; using current directory %r",
            working_directory, os.getcwd())
        working_directory = os.getcwd()
    working_directory = os.path.abspath(working_directory)

    voxel_size = gp.Coordinate(parameter['voxel_size'])

    # Array Specifications.
    raw = gp.ArrayKey('RAW')
    #gt_neurons = gp.ArrayKey('GT_NEURONS')
    gt_postpre_vectors = gp.ArrayKey('GT_POSTPRE_VECTORS')
    gt_post_indicator = gp.ArrayKey('GT_POST_INDICATOR')
    post_loss_weight = gp.ArrayKey('POST_LOSS_WEIGHT')
    vectors_mask = gp.ArrayKey('VECTORS_MASK')

    pred_postpre_vectors = gp.ArrayKey('PRED_POSTPRE_VECTORS')
    pred_post_indicator = gp.ArrayKey('PRED_POST_INDICATOR')

    grad_syn_indicator = gp.ArrayKey('GRAD_SYN_INDICATOR')
    grad_partner_vectors = gp.ArrayKey('GRAD_PARTNER_VECTORS')

    # Points specifications
    dummypostsyn = gp.GraphKey('DUMMYPOSTSYN')
    #postsyn = gp.GraphKey('POSTSYN')
    #presyn = gp.GraphKey('PRESYN')
    graphsyn = gp.GraphKey('GRAPHSYN')
    trg_context = 140  # AddPartnerVectorMap context in nm - pre-post distance
    #print(os.path.abspath('./'+parameter['config_name']+'_config.json'))
    with open(os.path.abspath('./'+parameter['config_name']+'_config.json'), 'r') as f: #need net config json in the parameter file
        net_config = json.load(f)

    input_size = gp.Coordinate(net_config['input_shape']) * voxel_size
    output_size = gp.Coordinate(net_config['output_shape']) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    #request.add(gt_neurons, output_size)
    request.add(gt_postpre_vectors, output_size)
    request.add(gt_post_indicator, output_size)
    request.add(post_loss_weight, output_size)
    request.add(vectors_mask, output_size)
    request.add(dummypostsyn, output_size)

    #print("\nTHIS IS REQUEST SPEC\n")
    for (key, request_spec) in request.items():
        print(key)
        print(request_spec.roi)
        request_spec.roi.contains(request_spec.roi)
    # slkfdms

    snapshot_request = gp.BatchRequest({
        pred_post_indicator: request[gt_postpre_vectors],
        pred_postpre_vectors: request[gt_postpre_vectors],
        grad_syn_indicator: request[gt_postpre_vectors],
        grad_partner_vectors: request[gt_postpre_vectors],
        vectors_mask: request[gt_postpre_vectors]
    })

    postsyn_rastersetting = gp.RasterizationSettings(
        radius=parameter['blob_radius'],
        #mask=gt_neurons,
        mode=parameter['blob_mode'])
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #if gpus:
  # Restrict TensorFlow to only use the first GPU
    #    gpu_id = int(augments['GPU_id'])
    #    try:
    #        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
    #    except RuntimeError as e:
    # Visible devices must be set at program startup
    #       print(e)
    #with tf.device('/gpu:0'):
    pipeline = tuple([create_source(s, raw,
                                    graphsyn, dummypostsyn,
                                    parameter, data_dir_syns[s],
                                    syn_db_names[s], in_rois[s]) for s in
                      range(len(in_rois))])
    print(pipeline)
    pipeline += gp.RandomProvider()
    pipeline += gp.Normalize(raw)
    if augment:
        pipeline += gp.ElasticAugment(augments['elastic_spacing'], #need to change this if different 
                                      augments['elastic_jitter'],
                                      [0, math.pi / 2.0],
                                      prob_slip=augments['elastic_prob_slip'],
                                      prob_shift=augments['elastic_prob_shift'],
                                      max_misalign=10,
                                      subsample=8)
        pipeline += gp.SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
        pipeline += gp.IntensityAugment(raw, augments['intensity_scale_min'], augments['intensity_scale_max'],
                                             augments['intensity_shift_min'], augments['intensity_shift_max'],
                                            z_section_wise=True)
        pipeline += gp.NoiseAugment(raw)
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += gp.RasterizeGraph(graphsyn, gt_post_indicator,
                                   gp.ArraySpec(voxel_size=voxel_size,
                                                dtype=np.int32),
                                   postsyn_rastersetting)
    spec = gp.ArraySpec(voxel_size=voxel_size)
        #print(voxel_size)
    pipeline += AddPartnerVectorMap(
        src_points=graphsyn,
        array=gt_postpre_vectors,
        radius=parameter['d_blob_radius'],
        trg_context=trg_context,  # enlarge
        array_spec=spec,
#            mask=gt_neurons
        pointmask=vectors_mask
    )
    pipeline += gp.BalanceLabels(labels=gt_post_indicator,
                                     scales=post_loss_weight,
                                     slab=(-1, -1, -1),
                                     clipmin=parameter['cliprange'][0],
                                     clipmax=parameter['cliprange'][1])
    if parameter['d_scale'] != 1:
        pipeline += gp.IntensityScaleShift(gt_postpre_vectors,
                                           scale=parameter['d_scale'], shift=0)
    pipeline += gp.PreCache(
        cache_size=parameter['cache_size'], #gimme cache size
        num_workers=parameter['num_workers']) #gimme num workers
    #subprocess.call(["cp", "*.meta", parameter['chkpt_loc']])
    pipeline += gp.tensorflow.Train(
            working_directory+'/checkpoints/'+parameter['chkpt_loc'], #need to provide checkpoint location
            optimizer=net_config['optimizer'],
            loss=net_config['loss'],
            summary=net_config['summary'],
            log_dir=working_directory+'/tensorboard/',
            save_every=100000,  # 10000
            log_every=100,
            inputs={
                net_config['raw']: raw,
                net_config['gt_partner_vectors']: gt_postpre_vectors,
                net_config['gt_syn_indicator']: gt_post_indicator,
                net_config['vectors_mask']: vectors_mask,
                # Loss weights --> mask
                net_config['indicator_weight']: post_loss_weight,  # Loss weights
            },
            outputs={
                net_config['pred_partner_vectors']: pred_postpre_vectors,
                net_config['pred_syn_indicator']: pred_post_indicator,
            },
            gradients={
                net_config['pred_partner_vectors']: grad_partner_vectors,
                net_config['pred_syn_indicator']: grad_syn_indicator,
            },
        )
        # Visualize.

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += gp.Snapshot({
            raw: 'volumes/raw',
    #        gt_neurons: 'volumes/labels/neuron_ids',
            gt_post_indicator: 'volumes/gt_post_indicator',
            gt_postpre_vectors: 'volumes/gt_postpre_vectors',
            pred_postpre_vectors: 'volumes/pred_postpre_vectors',
            pred_post_indicator: 'volumes/pred_post_indicator',
            post_loss_weight: 'volumes/post_loss_weight',
            grad_syn_indicator: 'volumes/post_indicator_gradients',
            grad_partner_vectors: 'volumes/partner_vectors_gradients',
            vectors_mask: 'volumes/vectors_mask'
        },
            every=50000,
            output_filename='batch_{iteration}.hdf',
            output_dir = working_directory+'/snapshot/',
            compression_type='gzip',
            additional_request=snapshot_request)
    pipeline += gp.PrintProfilingStats(every=100)
    print("Starting training...")
    max_iteration = parameter['max_iteration']
    with gp.build(pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)


if __name__ == "__main__":
    # Set to DEBUG to increase verbosity for
    # everything. logging.INFO --> logging.DEBUG
    # Example of how to only increase verbosity for specific python modules.
    logging.getLogger('gunpowder.nodes.rasterize_points').setLevel(
        logging.INFO)
    logging.getLogger('synful.gunpowder.hdf5_points_source').setLevel(
        logging.INFO)
    logging.basicConfig(level=logging.INFO)
    param = sys.argv[1]
    #augs = sys.argv[2]
    with open(param, 'r') as f: #this may need to be changed
        parameter = json.load(f)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(parameter['augments']['GPU_id'])

    #gpus = tf.config.experimental.get_visible_devices('GPU')
    #if gpus:
        #for gpu in gpus[:1]:
    #tf.config.experimental.set_memory_growth(tf.device('/gpu:0'),True)
    with tf.device('/gpu:'+str(parameter['augments']['GPU_id'])):
 #       config = tf.ConfigProto(device_count={'GPU':int(augments['GPU_id'])})
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #print(config)
        sess = tf.Session(config=config)
        build_pipeline(parameter, augment=True)

#    build_pipeline(parameter, augment=True)

